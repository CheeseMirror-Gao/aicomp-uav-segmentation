from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def indexed_files(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate filename stem {path.stem!r} in {root}")
            result[path.stem] = path
    return result


def paired_samples(image_root: str | Path, mask_root: str | Path) -> list[tuple[Path, Path]]:
    images = indexed_files(image_root)
    masks = indexed_files(mask_root)
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise ValueError("No image/mask pairs found by matching filename stems")
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files: missing_masks={len(missing_masks)}, missing_images={len(missing_images)}"
        )
    return [(images[key], masks[key]) for key in common]


def split_samples(
    samples: list[tuple[Path, Path]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if len(samples) < 2:
        raise ValueError("At least two paired samples are required for train/validation split")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
    return shuffled[val_size:], shuffled[:val_size]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        crop_size: int,
        training: bool,
        num_classes: int = 9,
    ) -> None:
        self.samples = samples
        self.crop_size = crop_size
        self.training = training
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        size = min(self.crop_size, width, height)
        if self.training:
            left = random.randint(0, width - size)
            top = random.randint(0, height - size)
        else:
            left = (width - size) // 2
            top = (height - size) // 2
        box = (left, top, left + size, top + size)
        image, mask = image.crop(box), mask.crop(box)
        if size != self.crop_size:
            image = image.resize((self.crop_size, self.crop_size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.crop_size, self.crop_size), Image.Resampling.NEAREST)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        with Image.open(mask_path) as loaded:
            mask = loaded.copy()
        if image.size != mask.size:
            raise ValueError(f"Size mismatch for {image_path.name}: {image.size} vs {mask.size}")

        image, mask = self._crop(image, mask)
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        invalid = (mask_array < 0) | (mask_array >= self.num_classes)
        if invalid.any():
            values = np.unique(mask_array[invalid]).tolist()
            raise ValueError(f"Unexpected class IDs in {mask_path}: {values}")

        pixels = (torch.from_numpy(image_array) - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "pixel_values": pixels,
            "labels": torch.from_numpy(mask_array.copy()).long(),
            "name": image_path.stem,
        }

