"""Frame I/O: loading rendered videos, resizing, and saving results."""
import os
import re
import imageio
import numpy as np
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def natural_sort_key(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', s)]


def list_frame_names(folder, num_frames=None):
    """Image file names in `folder`, naturally sorted (frame_2 < frame_10), truncated to `num_frames`."""
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    files.sort(key=natural_sort_key)
    return files if num_frames is None else files[:num_frames]


def crop_and_resize(image, height, width):
    """Resize (bilinear) so the image covers (height, width), then center-crop to exactly that size.

    This is the resizing used for FixAnything training and evaluation.
    """
    if image.size == (width, height):
        return image
    image_width, image_height = image.size
    scale = max(width / image_width, height / image_height)
    new_width = round(image_width * scale)
    new_height = round(image_height * scale)
    image = image.resize((new_width, new_height), Image.BILINEAR)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def load_frames(path, height=None, width=None, num_frames=None):
    """Load RGB frames from an image folder or a video file.

    Frames are optionally resized with `crop_and_resize` to (height, width) and truncated to `num_frames`.
    Returns (frames, names): a list of PIL images and their file names (synthetic names for video files).
    """
    if os.path.isdir(path):
        names = list_frame_names(path, num_frames)
        frames = [Image.open(os.path.join(path, n)).convert("RGB") for n in names]
    else:
        reader = imageio.get_reader(path)
        frames = []
        for i, frame in enumerate(reader):
            if num_frames is not None and i >= num_frames:
                break
            frames.append(Image.fromarray(np.array(frame)).convert("RGB"))
        reader.close()
        names = [f"frame_{i:05d}.png" for i in range(len(frames))]
    if height is not None and width is not None:
        frames = [crop_and_resize(f, height, width) for f in frames]
    return frames, names


def save_frames(frames, folder, names=None):
    os.makedirs(folder, exist_ok=True)
    if names is None:
        names = [f"frame_{i:05d}.png" for i in range(len(frames))]
    for frame, name in zip(frames, names):
        frame.save(os.path.join(folder, name))


def side_by_side(frames_a, frames_b):
    """Concatenate two frame lists horizontally (for input | output videos)."""
    out = []
    for a, b in zip(frames_a, frames_b):
        canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        out.append(canvas)
    return out


def save_video(frames, path, fps=15, quality=9):
    # macro_block_size=2: keep the frame size (imageio's default pads to multiples of 16); H.264 needs even sizes
    writer = imageio.get_writer(path, fps=fps, quality=quality, macro_block_size=2)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
