"""Reconstruct a scene from a few RGB images with MapAnything and render it along a camera path
between the first and last image, to be cleaned up by FixAnything.

Example:
    python scripts/run_mapanything.py --images examples/microkitchen_2views/images --output_dir outputs/microkitchen_2views
    python scripts/run_inference.py --input outputs/microkitchen_2views/rendered.mp4 --output_dir outputs/microkitchen_2views

The rendering follows the FixAnything data generation: MapAnything runs at its native resolution
(518 px on the longest side, e.g. 518x294 for 16:9 images) and the reconstruction (mesh or point cloud)
is rendered into a larger viewport of the same aspect ratio (default height 480 -> 846x480) by scaling
the predicted intrinsics. The first and last frames are replaced by the input images, crop-resized to
the render size, so they are pixel-aligned with the rendered geometry.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import warnings

import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation, Slerp

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.viz import predictions_to_glb

from fixanything.data import list_frame_names, save_video


class SimpleRenderer:
    """Minimal pyrender-based mesh/point renderer (OpenCV camera convention)."""

    def __init__(self, height=480, width=640, point_size=2):
        self.renderer = pyrender.OffscreenRenderer(width, height, point_size=point_size)
        self.scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 1.0]))
        self._camera_node = None

    def init_meshes(self, meshes):
        for mesh in meshes:
            self.scene.add(mesh)

    def render(self, height, width, K, pose, render_flags=None):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width

        if self._camera_node is not None:
            self.scene.remove_node(self._camera_node)

        cam = pyrender.IntrinsicsCamera(
            cx=K[0, 2], cy=K[1, 2], fx=K[0, 0], fy=K[1, 1], zfar=2000
        )
        t = np.pi
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        fixed_pose = pose @ axis_transform

        self._camera_node = self.scene.add(cam, pose=fixed_pose)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if render_flags is not None:
                color, _ = self.renderer.render(self.scene, render_flags)
            else:
                color, _ = self.renderer.render(self.scene)
        return color


def run_mapanything(image_paths, model):
    """MapAnything inference from RGB only. Returns per-view world points, colors, masks, poses, intrinsics."""
    views = load_images(image_paths)   # resized to MapAnything's fixed resolution buckets (518 px longest side)
    outputs = model.infer(
        views,
        memory_efficient_inference=False,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    world_points, images, masks, poses, intrinsics = [], [], [], [], []
    for pred in outputs:
        depthmap = pred["depth_z"][0].squeeze(-1)
        K = pred["intrinsics"][0]
        c2w = pred["camera_poses"][0]
        pts3d, valid_mask = depthmap_to_world_frame(depthmap, K, c2w)
        world_points.append(pts3d.cpu().numpy())
        images.append(pred["img_no_norm"][0].cpu().numpy())   # HxWx3 float in [0, 1]
        masks.append(valid_mask.cpu().numpy())
        poses.append(c2w.cpu().numpy())
        intrinsics.append(K.cpu().numpy())
    return world_points, images, masks, poses, intrinsics


def median_scene_depth(world_points, masks, poses):
    """Median distance of the reconstructed points from the midpoint of the camera path."""
    pts = np.concatenate([p.reshape(-1, 3)[m.reshape(-1)] for p, m in zip(world_points, masks)])
    center = 0.5 * (poses[0][:3, 3] + poses[-1][:3, 3])
    return float(np.median(np.linalg.norm(pts - center, axis=1)))


def interpolate_cameras(pose_a, pose_b, K_a, K_b, num_frames, push_distance=0.0):
    """Camera path from camera a to camera b: slerp on rotation, lerp on translation and intrinsics.
    With `push_distance` > 0 the camera additionally dollies forward mid-path and back (a sine bump
    along the mean viewing direction that vanishes at both endpoints)."""
    t = np.linspace(0.0, 1.0, num_frames)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]])))
    rotations = slerp(t).as_matrix()
    forward = pose_a[:3, 2] + pose_b[:3, 2]
    forward = forward / np.linalg.norm(forward)
    poses, intrinsics = [], []
    for i in range(num_frames):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotations[i]
        pose[:3, 3] = ((1 - t[i]) * pose_a[:3, 3] + t[i] * pose_b[:3, 3]
                       + push_distance * np.sin(np.pi * t[i]) * forward)
        poses.append(pose)
        intrinsics.append(((1 - t[i]) * K_a + t[i] * K_b).astype(np.float32))
    return poses, intrinsics


def push_in_cameras(pose_a, pose_b, K_a, K_b, num_frames, world_points, masks, push_scale=0.22):
    """Interpolated camera path that additionally dollies toward the scene mid-path and back,
    by `push_scale` x the median scene depth."""
    push_distance = push_scale * median_scene_depth(world_points, masks, [pose_a, pose_b])
    return interpolate_cameras(pose_a, pose_b, K_a, K_b, num_frames, push_distance=push_distance)


def build_scene_meshes(world_points, images, masks, render_mesh):
    """pyrender meshes of the reconstruction: a triangle mesh per input view, or colored point clouds."""
    if render_mesh:
        predictions = {
            "world_points": np.stack(world_points, axis=0),
            "images": np.stack(images, axis=0).copy(),
            "final_masks": np.stack(masks, axis=0),
        }
        scene_3d = predictions_to_glb(predictions, as_mesh=True)
        meshes = [pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for geom in scene_3d.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    else:
        scene_3d = None
        meshes = []
        for pts, img, mask in zip(world_points, images, masks):
            vertices = pts.reshape(-1, 3)[mask.reshape(-1)]
            colors = (img.reshape(-1, 3)[mask.reshape(-1)] * 255).astype(np.uint8)
            meshes.append(pyrender.Mesh.from_points(vertices, colors))
    return meshes, scene_3d


def render_path(meshes, poses, intrinsics, processed_size, render_size, point_size=None):
    """Render the reconstruction along `poses` into a (render_h, render_w) viewport, scaling the
    intrinsics from MapAnything's processed resolution."""
    pcd_h, pcd_w = processed_size
    render_h, render_w = render_size
    scale_x, scale_y = render_w / pcd_w, render_h / pcd_h
    if abs(pcd_w / pcd_h - render_w / render_h) > 0.02:
        print(f"WARNING: render aspect ratio ({render_w}x{render_h}) differs from MapAnything's ({pcd_w}x{pcd_h}); "
              "rendered geometry and the replaced input frames may not align.")
    if point_size is None:
        point_size = max(1, int(round(2 * scale_x)))
    print(f"Rendering at {render_w}x{render_h} (MapAnything resolution {pcd_w}x{pcd_h}, scale {scale_x:.2f}x, point_size {point_size})")

    renderer = SimpleRenderer(height=render_h, width=render_w, point_size=point_size)
    renderer.init_meshes(meshes)
    render_flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.FLAT

    frames = []
    for pose, K in tqdm(list(zip(poses, intrinsics)), desc="Rendering"):
        K = K.copy()
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        K[0, 2] += 0.5
        K[1, 2] += 0.5
        frames.append(Image.fromarray(renderer.render(render_h, render_w, K, pose, render_flags).astype(np.uint8)))
    renderer.renderer.delete()
    return frames


def parse_args():
    p = argparse.ArgumentParser(description="MapAnything reconstruction + rendering for FixAnything")
    p.add_argument("--images", type=str, required=True,
                   help="Folder with the input RGB images (naturally sorted; the path goes from the first to the last).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=61, help="Frames to render (FixAnything uses 61).")
    p.add_argument("--render", type=str, default="mesh", choices=["mesh", "pcd"],
                   help="Render the reconstruction as a triangle mesh or as a point cloud.")
    p.add_argument("--trajectory", type=str, default="interpolate", choices=["interpolate", "push_in"],
                   help="Camera path: straight interpolation between the first and last view, or push_in "
                        "(additionally dolly toward the scene mid-path and back).")
    p.add_argument("--render_height", type=int, default=480,
                   help="Viewport height; the width follows MapAnything's aspect ratio (846 for 16:9).")
    p.add_argument("--point_size", type=int, default=None, help="Point size for --render pcd (default: auto).")
    p.add_argument("--model", type=str, default="facebook/map-anything",
                   help="MapAnything checkpoint (facebook/map-anything or facebook/map-anything-apache).")
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    image_paths = [os.path.join(args.images, n) for n in list_frame_names(args.images)]
    if len(image_paths) < 2:
        raise SystemExit(f"Need at least 2 images in {args.images}, found {len(image_paths)}")
    print(f"Input images ({len(image_paths)}): {[os.path.basename(p) for p in image_paths]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(args.model).to(device)
    world_points, images, masks, poses, intrinsics = run_mapanything(image_paths, model)
    processed_size = world_points[0].shape[:2]
    print(f"MapAnything resolution: {processed_size[1]}x{processed_size[0]}")

    # Camera path between the first and last input view, rendered at a higher resolution.
    render_size = (args.render_height, 2 * int(round(args.render_height * processed_size[1] / processed_size[0] / 2)))  # even width for H.264
    if args.trajectory == "push_in":
        path_poses, path_intrinsics = push_in_cameras(poses[0], poses[-1], intrinsics[0], intrinsics[-1],
                                                      args.num_frames, world_points, masks)
    else:
        path_poses, path_intrinsics = interpolate_cameras(poses[0], poses[-1], intrinsics[0], intrinsics[-1], args.num_frames)
    meshes, scene_3d = build_scene_meshes(world_points, images, masks, render_mesh=(args.render == "mesh"))
    frames = render_path(meshes, path_poses, path_intrinsics, processed_size, render_size, point_size=args.point_size)

    # The first / last frames are the input images themselves (clean frames for FixAnything).
    for i, p in ((0, image_paths[0]), (len(frames) - 1, image_paths[-1])):
        frames[i] = crop_resize_if_necessary(Image.open(p).convert("RGB"), resolution=render_size[::-1])[0]

    os.makedirs(args.output_dir, exist_ok=True)
    save_video(frames, os.path.join(args.output_dir, "rendered.mp4"), fps=args.fps)
    if scene_3d is not None:
        scene_3d.export(os.path.join(args.output_dir, "reconstruction.glb"))
    print(f"Saved {len(frames)} rendered frames to {args.output_dir}/rendered.mp4")
    print(f"Next: python scripts/run_inference.py --input {args.output_dir}/rendered.mp4 --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
