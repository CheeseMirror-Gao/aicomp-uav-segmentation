from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavseg.data import SegmentationDataset, paired_samples, split_samples
from uavseg.metrics import confusion_matrix, iou_from_confusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single SegFormer model for UAV segmentation.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="outputs/segformer_b0")
    parser.add_argument("--model", default="nvidia/mit-b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--ignore-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device, num_classes: int, ignore_index: int) -> dict:
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    for batch in tqdm(loader, desc="validate", leave=False):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(pixel_values=pixels).logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
        total_loss += float(loss)
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        for prediction, target in zip(predictions, targets):
            matrix += confusion_matrix(prediction, target, num_classes, ignore_index)
    miou, per_class = iou_from_confusion(matrix, ignore_index)
    return {
        "loss": total_loss / max(1, len(loader)),
        "miou": miou,
        "per_class_iou": per_class,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training baseline")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    samples = paired_samples(args.images, args.masks)
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    print(f"samples: train={len(train_samples)}, val={len(val_samples)}")
    train_data = SegmentationDataset(train_samples, args.crop_size, True, args.num_classes)
    val_data = SegmentationDataset(val_samples, args.crop_size, False, args.num_classes)
    loader_options = dict(num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, best_miou = 1, -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["best_miou"]

    history_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(pixel_values=pixels).logits
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, labels, ignore_index=args.ignore_index)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss):.4f}")

        metrics = validate(model, val_loader, device, args.num_classes, args.ignore_index)
        metrics.update(epoch=epoch, train_loss=running_loss / max(1, len(train_loader)))
        print(json.dumps(metrics, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        state = {
            "epoch": epoch,
            "best_miou": max(best_miou, metrics["miou"]),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        torch.save(state, output / "last.pt")
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            state["best_miou"] = best_miou
            torch.save(state, output / "best.pt")
            print(f"new best mIoU: {best_miou:.6f}")


if __name__ == "__main__":
    main()

