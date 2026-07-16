"""
Class Distribution — PASTIS-R Format
====================================
O SINGURĂ figură cu două panouri suprapuse:
    - sus:  PASTIS-R (France)
    - jos:  Slovakia
Fiecare panou = distribuția pe cele 18 clase de cultură (0..17, Meadow..Sorghum),
cu procentul afișat deasupra fiecărei bare.

Toate fișierele se salvează în ACELAȘI folder ca acest script.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  MĂRIMEA TEXTULUI  ← schimbă aici dacă vrei și mai mare / mai mic
# ══════════════════════════════════════════════════════════════════════════════
FS_PCT      = 13   # procentele de deasupra barelor
FS_XTICK    = 13   # numele claselor de pe axa x
FS_YTICK    = 13   # valorile de pe axa y
FS_YLABEL   = 15   # "% of parcels"
FS_TITLE    = 18   # titlul fiecărui panou
FS_SUPTITLE = 22   # titlul general

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS MAPPING
# ══════════════════════════════════════════════════════════════════════════════

# Schema semantică completă (0 = Background ... 19 = Void).
CLASS_NAMES_ORDERED = [
    "Background",                  # 0
    "Meadow",                      # 1
    "Soft winter wheat",           # 2
    "Corn",                        # 3
    "Winter barley",               # 4
    "Winter rapeseed",             # 5
    "Spring barley",               # 6
    "Sunflower",                   # 7
    "Grapevine",                   # 8
    "Beet",                        # 9
    "Winter triticale",            # 10
    "Winter durum wheat",          # 11
    "Fruits, veg, flowers",        # 12
    "Potatoes",                    # 13
    "Leguminous fodder",           # 14
    "Soybeans",                    # 15
    "Orchard",                     # 16
    "Mixed cereal",                # 17
    "Sorghum",                     # 18
    "Void label",                  # 19
]

N_CLASSES = len(CLASS_NAMES_ORDERED)

CROP_IDX   = list(range(1, 19))                        # 1..18 în schema completă
CROP_NAMES = [CLASS_NAMES_ORDERED[i] for i in CROP_IDX]  # Meadow .. Sorghum
N_CROPS    = len(CROP_IDX)

CLASS_COLORS = [
    "#aec6e8", "#f57e20", "#f5c28a", "#6ab86a", "#b8e68a", "#d62728",
    "#f7b6c2", "#7f4fbf", "#c5b0d5", "#7f5234", "#c49c7f", "#e377c2",
    "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf",
    "#9edae5", "#bfbfbf",
]
CROP_COLORS = [CLASS_COLORS[i] for i in CROP_IDX]

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← schimbă path-urile aici
# ══════════════════════════════════════════════════════════════════════════════

# ►►► COMPLETEAZĂ path-ul (sau path-urile pe ani) ale datelor Franței aici ◄◄◄
FRANCE_YEARS = {
    "France": "",

}

SLOVAKIA_YEARS = {
    "Slovakia 2020": ""
}

# Titlurile celor două panouri (sus / jos).
FRANCE_TITLE   = "PASTIS-R (France)"
SLOVAKIA_TITLE = "Slovakia"

# ── Maparea claselor ──────────────────────────────────────────────────────────
# psetae: codurile brute din label_44class se remapează la POZIȚIA lor în lista
# sub_classes → asta dă indexul semantic 0..19, care coincide cu CLASS_NAMES_ORDERED.
SUB_CLASSES_44 = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14,
                  16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
SUB44_TO_IDX = {c: i for i, c in enumerate(SUB_CLASSES_44)}   # cod 44class -> 0..19

# Cheia din labels.json + eventuala remapare, per set:
#   - Franța   : label_44class + remap prin sub_classes (maparea canonică psetae).
#                Codul brut -> POZIȚIA în sub_classes = indexul semantic 0..19
#                (0=Background, 1=Meadow, ... 18=Sorghum, 19=Void).
#                NU citi label_19class direct: e o grupare pe 19 clase cu ALTĂ
#                numerotare, iar citită direct lasă Corn=0, Beet=0 (codurile 3 și 9
#                nu apar în label_19class).
#   - Slovacia : CODE_GROUP, valorile sunt deja indexul semantic (fără remap).
FRANCE_LABEL   = "label_44class"
FRANCE_REMAP   = SUB44_TO_IDX
SLOVAKIA_LABEL = "CODE_GROUP"
SLOVAKIA_REMAP = None

# Salvează imaginile/JSON în ACELAȘI folder ca acest script (lângă fișierul tău).
OUTPUT_DIR = str(Path(__file__).resolve().parent)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_as_semantic(root: str, label_name: str, remap: dict | None) -> np.ndarray:
    """
    Numără parcelele pe clasă (index semantic 0..19).
      - remap=None : valoarea din labels.json e deja indexul semantic.
      - remap=dict : valoarea e un cod brut (ex. label_44class) remapat prin dict.
    """
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    p = Path(root) / "META" / "labels.json"
    if not p.exists():
        print(f"  ✗ labels.json nu există în {root}")
        return counts
    with open(p) as f:
        labels = json.load(f)
    if label_name not in labels:
        print(f"  ✗ cheia '{label_name}' nu există în {p} "
              f"(chei disponibile: {list(labels.keys())})")
        return counts

    skipped = 0
    for v in labels[label_name].values():
        if v is None:
            continue
        code = int(v)
        if remap is not None:
            idx = remap.get(code)
            if idx is None:                # cod care nu e în sub_classes -> ignorat
                skipped += 1
                continue
            counts[idx] += 1
        elif 0 <= code < N_CLASSES:
            counts[code] += 1
        else:
            skipped += 1
    if skipped:
        print(f"      (ignorate {skipped} parcele cu cod în afara mapării)")
    return counts


def load_combined(year_paths: dict, label: str, label_name: str,
                  remap: dict | None) -> np.ndarray:
    total = np.zeros(N_CLASSES, dtype=np.int64)
    for name, path in year_paths.items():
        if Path(path).exists():
            c = load_as_semantic(path, label_name, remap)
            total += c
            print(f"  ✓ {name}: {c.sum():,} parcele | "
                  f"{int((c > 0).sum())}/{N_CLASSES} clase")
        else:
            print(f"  ✗ {name}: path nu există ({path})")
    print(f"  → {label} total: {total.sum():,} parcele")
    return total


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT  —  o singură figură, două panouri (France sus / Slovakia jos)
# ══════════════════════════════════════════════════════════════════════════════

def _panel(ax, counts_full: np.ndarray, title: str):
    """Desenează un panou cu distribuția pe cele 18 clase de cultură."""
    crop_counts = counts_full[CROP_IDX]           # doar culturile (1..18)
    total       = crop_counts.sum()               # EXCLUDE Background(0) și Void(19)
    pcts        = crop_counts / total * 100 if total > 0 else crop_counts

    x = np.arange(N_CROPS)                         # 0..17
    ax.bar(x, pcts, color=CROP_COLORS,
           edgecolor="white", linewidth=0.6)

    # procentul deasupra fiecărei bare
    for xi, v in zip(x, pcts):
        if v > 0:
            ax.text(xi, v, f"{v:.1f}%",
                    ha="center", va="bottom",
                    fontsize=FS_PCT, fontweight="bold", color="#222222")

    ax.set_ylim(0, (pcts.max(initial=0) * 1.22) or 1)   # spațiu pt. etichete mari
    ax.set_xticks(x)
    ax.set_xticklabels([f"{j}\n{CROP_NAMES[j]}" for j in range(N_CROPS)],
                       rotation=45, ha="right", fontsize=FS_XTICK)
    ax.tick_params(axis="y", labelsize=FS_YTICK)
    ax.set_ylabel("% of parcels", fontsize=FS_YLABEL)
    ax.set_title(f"{title}   (n = {total:,} parcels)",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))


def plot_stacked(france: np.ndarray, slovakia: np.ndarray, save: bool = True):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 12), constrained_layout=True
    )

    _panel(ax_top, france,   FRANCE_TITLE)
    _panel(ax_bot, slovakia, SLOVAKIA_TITLE)

    fig.suptitle("Class Distribution — PASTIS-R Format",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    if save:
        p = Path(OUTPUT_DIR) / "class_distribution_france_slovakia.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"\n  Salvat: {p}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n── France ──")
    france_counts = load_combined(FRANCE_YEARS, "France", FRANCE_LABEL, FRANCE_REMAP)

    print("\n── Slovakia ──")
    slovakia_counts = load_combined(SLOVAKIA_YEARS, "Slovakia", SLOVAKIA_LABEL, SLOVAKIA_REMAP)

    if france_counts.sum() == 0 and slovakia_counts.sum() == 0:
        print("Niciun dataset găsit. Verifică path-urile.")
        raise SystemExit(1)

    # ── O singură figură ──────────────────────────────────────────────────────
    plot_stacked(france_counts, slovakia_counts)

    # ── JSON cu statistici (opțional) ─────────────────────────────────────────
    def _summary(counts):
        crop  = counts[CROP_IDX]
        total = int(crop.sum())                    # EXCLUDE Background și Void
        pct   = crop / total * 100 if total > 0 else crop
        return {
            "total_parcels_crops_only": total,
            "excluded": {
                "Background": int(counts[0]),
                "Void label": int(counts[19]),
            },
            "classes": {
                str(j): {
                    "name":  CROP_NAMES[j],
                    "count": int(crop[j]),
                    "pct":   round(float(pct[j]), 2),
                }
                for j in range(N_CROPS)
            }
        }

    stats = {"France": _summary(france_counts),
             "Slovakia": _summary(slovakia_counts)}
    out = Path(OUTPUT_DIR) / "class_distribution_france_slovakia.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"  JSON salvat: {out}")