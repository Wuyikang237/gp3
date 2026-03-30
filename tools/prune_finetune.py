import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


def model_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach()
            total += w.numel()
            zeros += (w == 0).sum().item()
    return 0.0 if total == 0 else zeros / total


def prune_model(model: nn.Module, amount: float = 0.3, min_channels: int = 16) -> int:
    params_to_prune = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.in_channels >= min_channels and m.out_channels >= min_channels:
                params_to_prune.append((m, "weight"))

    if not params_to_prune:
        raise RuntimeError("No Conv2d layers found for pruning.")

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Remove pruning re-parameterization so checkpoint can be used normally.
    for module, _ in params_to_prune:
        prune.remove(module, "weight")

    return len(params_to_prune)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Magnitude pruning + fine-tuning for custom YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="Path to teacher or baseline best.pt")
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml path")
    parser.add_argument("--prune", type=float, default=0.3, help="Global pruning ratio, e.g. 0.3")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tune epochs after pruning")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/healthcheck")
    parser.add_argument("--name", type=str, default="prune_finetune")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    yolo = YOLO(str(weights))
    model = yolo.model
    for p in model.parameters():
        p.requires_grad_(True)

    before = model_sparsity(model)
    n_layers = prune_model(model, amount=args.prune)
    after = model_sparsity(model)

    print(f"Pruned layers: {n_layers}")
    print(f"Conv sparsity before: {before:.4f}")
    print(f"Conv sparsity after : {after:.4f}")

    overrides = {
        "model": "yolov8n.pt",  # trainer bootstrap only
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }

    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model.to(trainer.device)
    trainer.train()


if __name__ == "__main__":
    main()
