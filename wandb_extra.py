import json
from pathlib import Path

import wandb
import torch


ENTITY = "my_team_projects"
PROJECT = "PASTIS"
RESULTS_ROOT = Path("./results")

NEW_PREFIX = "target_2025_from_2020_2024_128_"

OLD_RUN_NAMES = {
    "2020_128_mixup_temporal_20epochs",
    "2020_128_mixup_bands_20epochs",
    "2020_128_mixup_pixels_20epochs",
    "2020_128_mixup_pixels_50epochs",
    "2020_128_mixup_bands_50epochs",
    "2020_128_mixup_pixels_20epochs_pretrained",
    "2020_128_mixup_bands_20epochs_pretrained",
    "2020_128_mixup_temporal_20epochs_pretrained",
    "2020_128_mixup_labels_50epochs_pretrained",
    "2020_128_mixup_labels_20epochs_pretrained",
    "2020_128_mixup_labels_20epochs",
    "2020_128_mixup_labels_50epochs",
    "2020_128_pretrained_pastis_freeze_encoder_temporal",
    "2020_128_pretrained_pastis_freeze_encoder_spatial",
    "2020_128_pretrained_pastis_freeze_encoder",
    "2020_128_pretrained_pastis",
    "2020_128_scratch",
    "baseline_128",
}


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

updated = 0
skipped = 0

for run in runs:
    run_name = run.name

    is_new_experiment = run_name.startswith(NEW_PREFIX)
    is_old_experiment = run_name in OLD_RUN_NAMES

    if not (is_new_experiment or is_old_experiment):
        continue

    run_dir = RESULTS_ROOT / run_name

    if not run_dir.exists():
        print(f"[SKIP] {run_name}: nu gasesc folder local {run_dir}")
        skipped += 1
        continue

    val_metrics = load_json(run_dir / "val_metrics.json")
    overall = load_json(run_dir / "overall.json")
    ckpt_path = run_dir / "model.pth.tar"

    if val_metrics is None and overall is None:
        print(f"[SKIP] {run_name}: lipsesc val_metrics.json / overall.json")
        skipped += 1
        continue

    update = {}

    if val_metrics is not None:
        if "val_accuracy" in val_metrics:
            update["best_val_accuracy"] = float(val_metrics["val_accuracy"])
        if "val_IoU" in val_metrics:
            update["best_val_IoU"] = float(val_metrics["val_IoU"])
        if "val_loss" in val_metrics:
            update["best_val_loss"] = float(val_metrics["val_loss"])

    if overall is not None:
        if "Accuracy" in overall:
            update["best_overall_accuracy"] = float(overall["Accuracy"])
        if "MACRO_IoU" in overall:
            update["best_overall_MACRO_IoU"] = float(overall["MACRO_IoU"])

    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "epoch" in ckpt:
                update["best_epoch"] = int(ckpt["epoch"])
            if "best_mIoU" in ckpt:
                update["best_checkpoint_mIoU"] = float(ckpt["best_mIoU"])
        except Exception as e:
            print(f"[WARN] {run_name}: nu pot citi checkpointul: {e}")

    if not update:
        print(f"[SKIP] {run_name}: nu am gasit metrici de pus")
        skipped += 1
        continue

    print(f"\n[UPDATE] {run_name}")
    for k, v in update.items():
        print(f"  {k}: {v}")
        run.summary[k] = v

    run.summary.update()
    updated += 1

print("\nDone.")
print(f"Updated: {updated}")
print(f"Skipped: {skipped}")