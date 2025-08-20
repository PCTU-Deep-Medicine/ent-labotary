import argparse
from typing import Dict, List

import numpy as np
import pytorch_lightning as L
import timm
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

# ==== IMPORT của bạn (chỉnh lại đường dẫn cho đúng) ====
from src.datamodule.ent import ENTDataModule
from src.utils.metrics import MetricsManager  # <- class bạn đã đưa ở trên

# Các metric macro sẽ lấy từ trainer.test() (đã log trong MetricsManager)
METRIC_KEYS = [
    "test/macro/accuracy",
    "test/macro/precision",
    "test/macro/recall",
    "test/macro/f1",
    "test/macro/specificity",
    "test/macro/auroc",
]


# ------------------- LightningModule bọc backbone để test -------------------
class LitEvalModule(L.LightningModule):
    def __init__(self, backbone: torch.nn.Module, num_classes: int):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])  # tiện log
        self.backbone = backbone
        self.num_classes = num_classes
        self.crit = torch.nn.CrossEntropyLoss()
        self.metrics = MetricsManager(num_classes)

    def forward(self, x):
        return self.backbone(x)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        loss = self.crit(logits, y)

        # cập nhật metrics
        self.metrics.update(preds, probs, y)

        # (tuỳ) log loss
        self.log("test/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        # log macro/per-class + hình (ROC + Confusion matrix)
        self.metrics.compute_and_log(
            logger=self.logger,
            epoch=getattr(self, "current_epoch", 0),
            phase="test",
            log_fn=self.log,
            include_plots=True,
        )


# ------------------- Helper: load state_dict từ .ckpt/.pth -------------------
def load_state_dict_into_backbone(backbone: torch.nn.Module, ckpt_path: str | None):
    """Nạp weights từ .ckpt (Lightning) / .pth (state_dict).
    - Tự bỏ các prefix: 'model.', 'backbone.', 'net.', 'encoder.', 'module.'.
    - Giữ lại chỉ những key trùng tên trong backbone.
    - strict=False để bỏ qua head không khớp (fc/...).
    """
    if not ckpt_path:
        return backbone

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )

    # Bóc prefix nhiều lần cho tới khi không còn prefix nào khớp
    PREFIXES = ["model.", "backbone.", "net.", "encoder.", "module."]

    def strip_prefix(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for pf in PREFIXES:
                if k.startswith(pf):
                    k = k[len(pf) :]
                    changed = True
        return k

    stripped = {strip_prefix(k): v for k, v in state_dict.items()}

    # Chỉ giữ key có trong backbone (tránh unexpected)
    model_keys = set(backbone.state_dict().keys())
    filtered = {k: v for k, v in stripped.items() if k in model_keys}

    # Nạp
    missing, unexpected = backbone.load_state_dict(filtered, strict=True)
    print(
        f"[load] kept={len(filtered)} | missing={len(missing)} | unexpected={len(unexpected)}"
    )
    if missing:
        print("  - missing (thường là fc khác số lớp):", missing[:10], "...")
    if unexpected:
        print("  - unexpected:", unexpected[:10], "...")
    return backbone


# ------------------- Chạy 1 seed -------------------
def run_one_seed(
    seed: int,
    split_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    ckpt_path: str,
    log_to_wandb: bool = False,
    wandb_project: str = "ent-eval",
    wandb_entity: str = None,
    wandb_run_name: str = None,
) -> Dict[str, float]:
    """Chạy test cho 1 seed, trả về dict metric."""
    # 1) Seed
    L.seed_everything(seed, workers=True)

    # 2) DataModule
    dm = ENTDataModule(
        split_root=split_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    dm.setup("test")

    # 3) Backbone timm + nạp weight
    #    (đổi 'resnet50' nếu bạn dùng model khác; nên để num_classes=dm.num_classes)
    backbone = timm.create_model("resnet50", pretrained=False, num_classes=12)
    backbone = load_state_dict_into_backbone(backbone, ckpt_path)

    # 4) Bọc LightningModule
    model = LitEvalModule(backbone=backbone, num_classes=12)

    # 5) Logger (tuỳ chọn)
    logger = None
    if log_to_wandb:
        logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"test-seed-{seed}",
            group="eval-5seeds",
            job_type="test",
            reinit=True,
        )

    # 6) Trainer
    trainer = L.Trainer(
        logger=logger,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        inference_mode=True,
        accelerator="auto",
        devices="auto",
    )

    # 7) Test
    results: List[Dict[str, float]] = trainer.test(
        model=model, datamodule=dm, verbose=False
    )

    # 8) Gom metrics
    result = results[0] if results else {}
    out = {}
    for k in METRIC_KEYS:
        out[k] = float(result[k]) if k in result else float("nan")

    # đóng W&B (nếu log từng seed)
    if logger is not None:
        logger.experiment.finish()

    return out


# ------------------- main: chạy nhiều seed + aggregate -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_root",
        type=str,
        default="data/12endo",
        help="Thư mục data đã split (train/val/test)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Đường dẫn .ckpt (Lightning) hoặc .pth (state_dict) để load model",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--wandb_project", type=str, default="ent-eval")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Tên run W&B để log tổng hợp. Nếu bỏ qua, không log gì.",
    )
    args = parser.parse_args()

    # Chạy các seed
    seed_metrics: Dict[int, Dict[str, float]] = {}
    for s in args.seeds:
        m = run_one_seed(
            seed=s,
            split_root=args.split_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            ckpt_path=args.ckpt_path,
            log_to_wandb=bool(args.wandb_run_name),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=f"test-seed-{s}",
        )
        seed_metrics[s] = m

    # Tính mean / std
    agg_mean, agg_std = {}, {}
    for k in METRIC_KEYS:
        vals = np.array([seed_metrics[s][k] for s in args.seeds], dtype=float)
        agg_mean[k] = float(np.nanmean(vals))
        agg_std[k] = float(np.nanstd(vals, ddof=1))  # sample std

    # In console
    print("==== Per-seed metrics ====")
    for s in args.seeds:
        row = ", ".join(
            [f"{k.split('/')[-1]}={seed_metrics[s][k]:.4f}" for k in METRIC_KEYS]
        )
        print(f"seed {s}: {row}")

    print("\n==== Aggregate (mean ± std) ====")
    for k in METRIC_KEYS:
        print(f"{k}: {agg_mean[k]:.4f} ± {agg_std[k]:.4f}")

    # W&B run tổng hợp (một run duy nhất)
    if args.wandb_run_name:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type="aggregate",
            config={
                "seeds": args.seeds,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "image_size": args.image_size,
                "ckpt_path": args.ckpt_path,
            },
            reinit=True,
        )

        # Log aggregate
        wandb.log({f"aggregate/mean/{k}": v for k, v in agg_mean.items()}, commit=False)
        wandb.log({f"aggregate/std/{k}": v for k, v in agg_std.items()}, commit=True)

        # Log bảng per-seed
        table_cols = ["seed"] + [k.replace("test/macro/", "") for k in METRIC_KEYS]
        table_rows = []
        for s in args.seeds:
            row = [s] + [seed_metrics[s][k] for k in METRIC_KEYS]
            table_rows.append(row)
        table = wandb.Table(columns=table_cols, data=table_rows)
        wandb.log({"aggregate/per_seed_table": table})

        run.finish()


if __name__ == "__main__":
    main()
