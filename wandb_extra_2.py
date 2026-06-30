#!/usr/bin/env python3
"""
make_dissertation_metrics_tables.py

Genereaza doua tabele pentru disertatie din folderele de rezultate ale experimentelor:

1) Tabel principal:
   Method | Acc (%) | Balanced Acc (%) | Macro F1 | Val IoU | Overall IoU | Worst-3 IoU | Best epoch

2) Tabel per clasa:
   Class | Support | Baseline IoU | Best method IoU | Delta IoU | Baseline Recall | Best method Recall | Delta Recall

Input asteptat pentru fiecare run:
   results/<run_name>/
       val_metrics.json
       overall.json
       conf_mat.pkl
       model.pth.tar    optional, pentru best_epoch

Exemple:

python make_dissertation_metrics_tables.py \
  --results-root ./results \
  --baseline target_2025_from_2020_2024_128_scratch_no_mixup \
  --best target_2025_from_2020_2024_128_mixup_labels_W50_scratch \
  --prefix target_2025_from_2020_2024_128_ \
  --out-dir dissertation_tables

Pentru runurile vechi:
python make_dissertation_metrics_tables.py \
  --results-root ./results \
  --baseline 2020_128_scratch \
  --best 2020_128_mixup_pixels_20epochs \
  --include-old \
  --out-dir dissertation_tables_old
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None


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


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_conf_mat(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        cm = pickle.load(f)
    return np.asarray(cm, dtype=np.float64)


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def class_metrics_from_cm(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returneaza metrici per clasa din matricea de confuzie.
    Assumption: randuri = true labels, coloane = predicted labels.
    """
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    fp = predicted - tp
    fn = support - tp

    recall = safe_div(tp, support)
    precision = safe_div(tp, predicted)
    iou = safe_div(tp, tp + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "support": support,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def aggregate_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    m = class_metrics_from_cm(cm)

    support = m["support"]
    total = support.sum()
    correct = np.diag(cm).sum()

    present = support > 0

    acc = float(correct / total) if total > 0 else 0.0
    balanced_acc = float(np.mean(m["recall"][present])) if np.any(present) else 0.0
    macro_f1 = float(np.mean(m["f1"][present])) if np.any(present) else 0.0
    macro_iou = float(np.mean(m["iou"][present])) if np.any(present) else 0.0

    weighted_acc = float(np.sum(m["recall"][present] * support[present]) / total) if total > 0 else 0.0
    weighted_f1 = float(np.sum(m["f1"][present] * support[present]) / total) if total > 0 else 0.0
    weighted_iou = float(np.sum(m["iou"][present] * support[present]) / total) if total > 0 else 0.0

    ious_present = m["iou"][present]
    if len(ious_present) == 0:
        worst3_iou = 0.0
        worst5_iou = 0.0
    else:
        worst3_iou = float(np.mean(np.sort(ious_present)[: min(3, len(ious_present))]))
        worst5_iou = float(np.mean(np.sort(ious_present)[: min(5, len(ious_present))]))

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
        "weighted_accuracy": weighted_acc,
        "weighted_f1": weighted_f1,
        "weighted_iou": weighted_iou,
        "worst3_iou": worst3_iou,
        "worst5_iou": worst5_iou,
    }


def get_best_epoch(run_dir: Path) -> Optional[int]:
    ckpt_path = run_dir / "model.pth.tar"
    if torch is None or not ckpt_path.exists():
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "epoch" in ckpt:
            return int(ckpt["epoch"])
    except Exception:
        return None

    return None


def method_label(run_name: str) -> str:
    """
    Nume mai curat pentru tabel. Poti modifica mapping-ul dupa cum vrei sa apara in disertatie.
    """
    mapping = {
        "scratch_no_mixup": "Scratch, fara mixup",
        "pretrained_finetune_all": "Pretrained, fine-tune all",
        "pretrained_freeze_spatial": "Pretrained + freeze spatial",
        "pretrained_freeze_temporal": "Pretrained + freeze temporal",
        "pretrained_freeze_all_encoder": "Pretrained + freeze all encoder",
        "mixup_labels_W50_scratch": "Mixup labels, W=50, scratch",
        "mixup_labels_W20_scratch": "Mixup labels, W=20, scratch",
        "mixup_pixels_W50_scratch": "Mixup pixels, W=50, scratch",
        "mixup_pixels_W20_scratch": "Mixup pixels, W=20, scratch",
        "mixup_temporal_W50_scratch": "Mixup temporal, W=50, scratch",
        "mixup_bands_W50_scratch": "Mixup bands, W=50, scratch",
        "mixup_bands_W20_scratch": "Mixup bands, W=20, scratch",
        "mixup_labels_W50_pretrained": "Mixup labels, W=50, pretrained",
        "mixup_labels_W20_pretrained": "Mixup labels, W=20, pretrained",
        "mixup_pixels_W20_pretrained": "Mixup pixels, W=20, pretrained",
        "mixup_temporal_W20_pretrained": "Mixup temporal, W=20, pretrained",
        "mixup_bands_W20_pretrained": "Mixup bands, W=20, pretrained",
    }

    # Pentru runurile noi, scoatem prefixul lung.
    prefix = "target_2025_from_2020_2024_128_"
    short = run_name[len(prefix):] if run_name.startswith(prefix) else run_name

    return mapping.get(short, run_name)


def collect_run_dirs(results_root: Path, prefix: str, include_old: bool) -> List[Path]:
    run_dirs = []

    for p in sorted(results_root.iterdir()):
        if not p.is_dir():
            continue

        has_files = (p / "conf_mat.pkl").exists() and ((p / "val_metrics.json").exists() or (p / "overall.json").exists())
        if not has_files:
            continue

        if prefix and p.name.startswith(prefix):
            run_dirs.append(p)
            continue

        if include_old and p.name in OLD_RUN_NAMES:
            run_dirs.append(p)
            continue

    return run_dirs


def build_main_table(run_dirs: List[Path]) -> List[Dict[str, object]]:
    rows = []

    for run_dir in run_dirs:
        cm_path = run_dir / "conf_mat.pkl"
        cm = load_conf_mat(cm_path)

        val_metrics = load_json(run_dir / "val_metrics.json") or {}
        overall = load_json(run_dir / "overall.json") or {}
        cm_aggr = aggregate_metrics_from_cm(cm)

        # Preferam valorile deja salvate in fisierele tale pentru coloanele originale,
        # iar metricile noi le luam din conf_mat.
        val_acc = val_metrics.get("val_accuracy")
        val_iou = val_metrics.get("val_IoU")
        overall_iou = overall.get("MACRO_IoU", cm_aggr["macro_iou"])
        macro_f1 = overall.get("MACRO_F1-score", cm_aggr["macro_f1"])

        if val_acc is None:
            val_acc = cm_aggr["accuracy"] * 100.0

        if val_iou is None:
            val_iou = cm_aggr["macro_iou"]

        row = {
            "run_name": run_dir.name,
            "method": method_label(run_dir.name),
            "Acc (%)": float(val_acc),
            "Balanced Acc (%)": cm_aggr["balanced_accuracy"] * 100.0,
            "Macro F1": float(macro_f1),
            "Val IoU": float(val_iou),
            "Overall IoU": float(overall_iou),
            "Worst-3 IoU": cm_aggr["worst3_iou"],
            "Worst-5 IoU": cm_aggr["worst5_iou"],
            "Weighted IoU": cm_aggr["weighted_iou"],
            "Best epoch": get_best_epoch(run_dir),
        }

        rows.append(row)

    # Sortare implicita: dupa Val IoU descrescator.
    rows.sort(key=lambda r: r["Val IoU"], reverse=True)
    return rows


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        print(f"[WARN] Nu am randuri pentru {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Saved: {path}")


def save_latex_table(rows: List[Dict[str, object]], path: Path, max_rows: Optional[int] = None) -> None:
    """
    Scrie un tabel LaTeX simplu. Il poti copia direct in disertatie si ajusta caption/label.
    """
    if max_rows is not None:
        rows = rows[:max_rows]

    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        ("method", "Metoda"),
        ("Acc (%)", "Acc (\\%)"),
        ("Balanced Acc (%)", "Bal. Acc (\\%)"),
        ("Macro F1", "Macro F1"),
        ("Val IoU", "Val IoU"),
        ("Overall IoU", "Overall IoU"),
        ("Worst-3 IoU", "Worst-3 IoU"),
    ]

    with open(path, "w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\hline\n")
        f.write(" & ".join([c[1] for c in cols]) + " \\\\\n")
        f.write("\\hline\n")

        for r in rows:
            values = []
            for key, _ in cols:
                v = r[key]
                if key == "method":
                    values.append(str(v))
                elif key in {"Acc (%)", "Balanced Acc (%)"}:
                    values.append(f"{float(v):.2f}")
                else:
                    values.append(f"{float(v):.3f}")
            f.write(" & ".join(values) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparatie intre experimente pe setul de validare.}\n")
        f.write("\\label{tab:experiment_metrics}\n")
        f.write("\\end{table}\n")

    print(f"[OK] Saved: {path}")


def per_class_comparison(
    baseline_dir: Path,
    best_dir: Path,
    class_names_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    cm_base = load_conf_mat(baseline_dir / "conf_mat.pkl")
    cm_best = load_conf_mat(best_dir / "conf_mat.pkl")

    if cm_base.shape != cm_best.shape:
        raise ValueError(
            f"Confusion matrices au forme diferite: baseline={cm_base.shape}, best={cm_best.shape}. "
            "Asta inseamna ca axele claselor nu sunt identice. Trebuie mapare explicita."
        )

    base = class_metrics_from_cm(cm_base)
    best = class_metrics_from_cm(cm_best)

    n = cm_base.shape[0]
    class_names = [str(i) for i in range(n)]

    if class_names_path is not None and class_names_path.exists():
        with open(class_names_path, "r") as f:
            loaded = json.load(f)
        # Accepta fie lista, fie dict {"0": "class_name", ...}
        if isinstance(loaded, list):
            class_names = [str(x) for x in loaded]
        elif isinstance(loaded, dict):
            class_names = [str(loaded.get(str(i), i)) for i in range(n)]

    rows = []
    for i in range(n):
        support = int(best["support"][i])
        rows.append({
            "class_index": i,
            "class": class_names[i] if i < len(class_names) else str(i),
            "Support": support,
            "Baseline IoU": float(base["iou"][i]),
            "Best method IoU": float(best["iou"][i]),
            "Delta IoU": float(best["iou"][i] - base["iou"][i]),
            "Baseline Recall": float(base["recall"][i]),
            "Best method Recall": float(best["recall"][i]),
            "Delta Recall": float(best["recall"][i] - base["recall"][i]),
            "Baseline F1": float(base["f1"][i]),
            "Best method F1": float(best["f1"][i]),
            "Delta F1": float(best["f1"][i] - base["f1"][i]),
        })

    rows.sort(key=lambda r: r["Delta IoU"], reverse=True)
    return rows


def save_per_class_latex(rows: List[Dict[str, object]], path: Path, top_k: Optional[int] = None) -> None:
    if top_k is not None:
        rows = rows[:top_k]

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write("Clasa & Support & Baseline IoU & Best IoU & $\\Delta$ IoU \\\\\n")
        f.write("\\hline\n")

        for r in rows:
            f.write(
                f"{r['class']} & {int(r['Support'])} & "
                f"{float(r['Baseline IoU']):.3f} & "
                f"{float(r['Best method IoU']):.3f} & "
                f"{float(r['Delta IoU']):+.3f} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparatie per clasa intre baseline si cea mai buna metoda.}\n")
        f.write("\\label{tab:per_class_gain}\n")
        f.write("\\end{table}\n")

    print(f"[OK] Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--prefix", default="target_2025_from_2020_2024_128_")
    parser.add_argument("--include-old", action="store_true")
    parser.add_argument("--baseline", required=True, help="Numele folderului baseline din results.")
    parser.add_argument("--best", required=True, help="Numele folderului metodei best din results.")
    parser.add_argument("--out-dir", default="dissertation_tables")
    parser.add_argument("--class-names-json", default=None)
    parser.add_argument("--top-k-classes", type=int, default=None, help="Optional: salveaza doar top K clase dupa Delta IoU in LaTeX.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)

    run_dirs = collect_run_dirs(results_root, args.prefix, args.include_old)
    if not run_dirs:
        raise RuntimeError("Nu am gasit niciun run valid. Verifica --results-root, --prefix sau --include-old.")

    main_rows = build_main_table(run_dirs)

    save_csv(main_rows, out_dir / "main_experiment_table.csv")
    save_latex_table(main_rows, out_dir / "main_experiment_table.tex")

    baseline_dir = results_root / args.baseline
    best_dir = results_root / args.best

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Nu exista baseline dir: {baseline_dir}")
    if not best_dir.exists():
        raise FileNotFoundError(f"Nu exista best dir: {best_dir}")

    class_names_path = Path(args.class_names_json) if args.class_names_json else None
    class_rows = per_class_comparison(baseline_dir, best_dir, class_names_path)

    save_csv(class_rows, out_dir / "per_class_gain_table.csv")
    save_per_class_latex(class_rows, out_dir / "per_class_gain_table.tex", top_k=args.top_k_classes)

    print("\nDone.")
    print("Tabele generate:")
    print(f"  {out_dir / 'main_experiment_table.csv'}")
    print(f"  {out_dir / 'main_experiment_table.tex'}")
    print(f"  {out_dir / 'per_class_gain_table.csv'}")
    print(f"  {out_dir / 'per_class_gain_table.tex'}")
    print("\nRecomandare:")
    print("  In tabelul principal foloseste coloanele: Acc (%), Balanced Acc (%), Macro F1, Val IoU, Overall IoU, Worst-3 IoU.")
    print("  In tabelul per clasa foloseste: class, Support, Baseline IoU, Best method IoU, Delta IoU.")


if __name__ == "__main__":
    main()
