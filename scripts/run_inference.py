"""Run FixAnything on a rendered video (a folder of frames or a video file).

Example:
    python scripts/run_inference.py --input examples/dl3dv_3dgs/input.mp4 --output_dir outputs/dl3dv_3dgs

`load_pipeline()` and `fix_video()` below are the whole inference API; the evaluation script imports them from here.
"""
import argparse
import glob
import os

import torch
from diffsynth.utils import ModelConfig

from fixanything.pipelines import WanVideoPipeline
from fixanything.data import load_frames, save_video, crop_and_resize, side_by_side

# Base model (same layout on ModelScope and Hugging Face).
WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN_MODEL_FILES = [
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
]
WAN_TOKENIZER_FILES = "google/*"

DEFAULT_PROMPT = "A clean, high-quality, photorealistic video with sharp details, smooth motion, and natural lighting."
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def _model_config(model_dir, pattern, download_source="huggingface", **kwargs):
    # Skip the remote download check when the files are already there (so this also works offline).
    present = len(glob.glob(os.path.join(model_dir, WAN_MODEL_ID, pattern))) > 0
    return ModelConfig(
        model_id=WAN_MODEL_ID, origin_file_pattern=pattern, local_model_path=model_dir,
        download_resource=download_source, skip_download=present, **kwargs,
    )


def load_pipeline(lora_path, model_dir="checkpoints", device="cuda", torch_dtype=torch.bfloat16,
                  lora_alpha=1.0, download_source="huggingface"):
    """Build the Wan2.1-I2V-14B pipeline, apply the FixAnything LoRA and enable VRAM offloading.

    Base-model files are looked up in `<model_dir>/Wan-AI/Wan2.1-I2V-14B-480P/` and downloaded
    from `download_source` ("huggingface" or "modelscope") if missing.
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"FixAnything LoRA not found: {lora_path}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype, device=device,
        model_configs=[_model_config(model_dir, p, download_source, offload_device="cpu") for p in WAN_MODEL_FILES],
        tokenizer_config=_model_config(model_dir, WAN_TOKENIZER_FILES, download_source),
        redirect_common_files=False,
    )
    pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    pipe.enable_vram_management()
    return pipe


def fix_video(
    pipe, frames, clean_frame_indices=None,
    num_frames=61, num_repeat_last=4, height=480, width=832,
    seed=1, num_inference_steps=10, cfg_scale=5.0,
    prompt=DEFAULT_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT, tiled=True,
):
    """Refine a rendered video.

    Args:
        frames: list of PIL images (the degraded rendering); the first `num_frames` are used.
        clean_frame_indices: indices of input frames that are already clean (e.g. rendered at the input
            views); the model keeps them and refines the rest. List of ints or whitespace-separated
            string. Default: first and last frame. Pass [] to refine every frame.
    The model runs on `num_frames + num_repeat_last` frames (the last frame is repeated, as in training)
    and the output is cropped back to `num_frames`.
    Returns:
        (generated_frames, input_frames): two lists of `num_frames` PIL images of size (width, height).
    """
    if len(frames) < num_frames:
        raise ValueError(f"Need {num_frames} input frames, got {len(frames)}.")
    if len(frames) > num_frames:
        print(f"[fix_video] Using the first {num_frames} of {len(frames)} frames.")
    frames = [crop_and_resize(f, height, width) for f in frames[:num_frames]]
    ref_video = frames + [frames[-1]] * num_repeat_last

    if clean_frame_indices is None:
        clean_frame_indices = [0, num_frames - 1]
    elif isinstance(clean_frame_indices, str):
        clean_frame_indices = [int(x) for x in clean_frame_indices.split()]
    clean = set(clean_frame_indices)
    if num_frames - 1 in clean:   # the repeats of the last frame inherit its flag
        clean.update(range(num_frames, num_frames + num_repeat_last))

    gen_video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=ref_video[0],
        reference_video=ref_video,
        clean_frame_indices=sorted(clean),
        num_frames=num_frames + num_repeat_last, height=height, width=width,
        seed=seed, tiled=tiled,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )
    return gen_video[:num_frames], ref_video[:num_frames]


def parse_args():
    p = argparse.ArgumentParser(description="FixAnything inference")
    p.add_argument("--input", type=str, required=True,
                   help="Rendered video file, or a folder of frames.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_path", type=str, default="checkpoints/fixanything_lora.safetensors")
    p.add_argument("--model_dir", type=str, default="checkpoints",
                   help="Folder containing Wan-AI/Wan2.1-I2V-14B-480P/ (downloaded there if missing).")
    p.add_argument("--clean_frame_indices", type=str, default=None,
                   help='Space-separated indices of input frames to keep as-is (default: first and last, "0 60"). '
                        'Use "" to refine every frame.')
    p.add_argument("--num_frames", type=int, default=61, help="Input frames used (the first ones).")
    p.add_argument("--num_repeat_last", type=int, default=4,
                   help="The last frame is repeated this many times for the model (as in training); the output is cropped back.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    frames, _ = load_frames(args.input, args.height, args.width, num_frames=args.num_frames)
    print(f"Loaded {len(frames)} frames from {args.input}")
    clean = None if args.clean_frame_indices is None else [int(x) for x in args.clean_frame_indices.split()]

    pipe = load_pipeline(args.lora_path, args.model_dir)
    gen_video, ref_video = fix_video(
        pipe, frames, clean_frame_indices=clean,
        num_frames=args.num_frames, num_repeat_last=args.num_repeat_last,
        height=args.height, width=args.width,
        seed=args.seed, num_inference_steps=args.num_inference_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_video(gen_video, os.path.join(args.output_dir, "generated.mp4"), fps=args.fps)
    save_video(ref_video, os.path.join(args.output_dir, "input.mp4"), fps=args.fps)
    save_video(side_by_side(ref_video, gen_video), os.path.join(args.output_dir, "side_by_side.mp4"), fps=args.fps)
    print(f"Saved results to {args.output_dir}/ (generated.mp4, input.mp4, side_by_side.mp4 = input | generated)")


if __name__ == "__main__":
    main()
