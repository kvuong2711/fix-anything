"""Download the FixAnything checkpoints: the Wan2.1-I2V-14B-480P base model and the FixAnything LoRA.

Example:
    python scripts/download_models.py --model_dir checkpoints
"""
import argparse

from huggingface_hub import hf_hub_download
from diffsynth.utils import ModelConfig
from run_inference import WAN_MODEL_ID, WAN_MODEL_FILES, WAN_TOKENIZER_FILES  # scripts/ is on sys.path when this script is run

LORA_REPO, LORA_FILE = "kvuong2711/fix-anything", "fixanything_lora.safetensors"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="checkpoints")
    p.add_argument("--source", type=str, default="huggingface", choices=["huggingface", "modelscope"],
                   help="Where to download the Wan2.1 base model from.")
    args = p.parse_args()
    for pattern in WAN_MODEL_FILES + [WAN_TOKENIZER_FILES]:
        ModelConfig(model_id=WAN_MODEL_ID, origin_file_pattern=pattern,
                    local_model_path=args.model_dir, download_resource=args.source).download_if_necessary()
    hf_hub_download(LORA_REPO, LORA_FILE, local_dir=args.model_dir)
    print(f"Base model in {args.model_dir}/{WAN_MODEL_ID}/, LoRA at {args.model_dir}/{LORA_FILE}")


if __name__ == "__main__":
    main()
