from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavseg.data import IMAGENET_MEAN, IMAGENET_STD, indexed_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-ID masks with one SegFormer model.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="nvidia/mit-b0")
    parser.add_argument("--num-classes", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for inference")
    device = torch.device("cuda")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=args.num_classes, ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for name, path in tqdm(indexed_files(args.images).items(), desc="predict"):
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        pixels = ((torch.from_numpy(array) - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(pixel_values=pixels).logits
            logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)[0].byte().cpu().numpy()
        Image.fromarray(prediction, mode="L").save(output / f"{name}.png")


if __name__ == "__main__":
    main()

