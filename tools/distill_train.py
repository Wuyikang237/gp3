import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.fasternet_simAM_yolo import FasterNetYOLO


class DistillFasterNetYOLO(FasterNetYOLO):
    def __init__(self, teacher_model: nn.Module, kd_alpha: float = 0.5, kd_temp: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        # Keep teacher outside nn.Module registration to prevent optimizer/freeze logic from touching it.
        self._teacher_holder = [teacher_model]
        self.kd_alpha = kd_alpha
        self.kd_temp = kd_temp

        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_model.eval()

    @property
    def teacher_model(self) -> nn.Module:
        return self._teacher_holder[0]

    @staticmethod
    def _flatten_tensors(preds):
        out = []
        if torch.is_tensor(preds):
            return [preds]
        if isinstance(preds, dict):
            for v in preds.values():
                out.extend(DistillFasterNetYOLO._flatten_tensors(v))
            return out
        if isinstance(preds, (list, tuple)):
            for v in preds:
                out.extend(DistillFasterNetYOLO._flatten_tensors(v))
            return out
        return out

    def _kd_loss(self, student_preds, teacher_preds):
        s_list = self._flatten_tensors(student_preds)
        t_list = self._flatten_tensors(teacher_preds)
        if not s_list or not t_list:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        kd = torch.tensor(0.0, device=s_list[0].device)
        n = min(len(s_list), len(t_list))
        for i in range(n):
            s, t = s_list[i], t_list[i]
            if s.shape == t.shape:
                kd = kd + F.mse_loss(s / self.kd_temp, t.detach() / self.kd_temp)
        return kd * (self.kd_temp**2)

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            if self.criterion is None:
                from ultralytics.utils.loss import v8DetectionLoss

                self.criterion = v8DetectionLoss(self)

            img = x["img"]
            if next(self.teacher_model.parameters()).device != img.device:
                self.teacher_model = self.teacher_model.to(img.device)

            student_preds = self._forward_once(img)
            det_loss, loss_items = self.criterion(student_preds, x)

            with torch.no_grad():
                if hasattr(self.teacher_model, "_forward_once"):
                    teacher_preds = self.teacher_model._forward_once(img)
                else:
                    # Fallback for other detect models.
                    old_mode = self.teacher_model.training
                    self.teacher_model.train()
                    teacher_preds = self.teacher_model(img)
                    self.teacher_model.train(old_mode)

            kd_loss = self._kd_loss(student_preds, teacher_preds)
            total_loss = det_loss + self.kd_alpha * kd_loss
            return total_loss, loss_items

        return self._forward_once(x)


def parse_depths(raw: str):
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("--depths must have 4 comma-separated integers, e.g. 1,2,8,2")
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge distillation training for custom FasterNet YOLO")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher checkpoint path")
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml path")
    parser.add_argument("--student-ckpt", type=str, default="", help="Optional student checkpoint to warm start")
    parser.add_argument("--nc", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depths", type=str, default="1,2,8,2")
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--kd-alpha", type=float, default=0.5, help="Weight for KD loss")
    parser.add_argument("--kd-temp", type=float, default=1.0, help="Temperature for KD loss")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/healthcheck")
    parser.add_argument("--name", type=str, default="distill_train")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    teacher = YOLO(str(teacher_path)).model
    student = DistillFasterNetYOLO(
        teacher_model=teacher,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        nc=args.nc,
        embed_dim=args.embed_dim,
        depths=parse_depths(args.depths),
        mlp_ratio=args.mlp_ratio,
    )

    if args.student_ckpt:
        ckpt_path = Path(args.student_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {ckpt_path}")
        warm = YOLO(str(ckpt_path)).model
        missing, unexpected = student.load_state_dict(warm.state_dict(), strict=False)
        print(f"Warm start loaded. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

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
    trainer.model = student.to(trainer.device)
    trainer.train()


if __name__ == "__main__":
    main()
