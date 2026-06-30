#!/usr/bin/env python3
"""
Ruleaza workflow-ul corect pentru Experiment 2:

SOURCE pretraining:
  /home/mnegru/Adelina/Experiment_2/PixelSet-Slovakia-2020-2024
  labels: CODE_GROUP
  subclasses: [0..19]
  mean/std: 2020-2024-Slovakia-meanstd.pkl

TARGET experiments:
  /home/mnegru/Adelina/Final_data/PixelSet-Slovakia-2025
  labels: CODE_GROUP
  subclasses: [0..19]
  mean/std: 2025-Slovakia-meanstd.pkl

Important:
- In codul tau original, argumentele se numesc dataset_pastis si dataset_slovakia.
- Aici dataset_pastis este folosit ca SOURCE 2020-2024.
- dataset_slovakia este folosit ca TARGET 2025.
- Pentru experimentele pretrained, checkpointul folosit este checkpointul rezultat din
  antrenarea scratch pe SOURCE 2020-2024.

Exemplu:
python run_slovakia_transfer_experiments.py \
  --train-script /home/mnegru/Adelina/PASTIS/train_non_adv.py \
  --device cuda \
  --wandb-offline \
  --skip-existing
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CLASSES_0_19 = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]"

# Mapping din tabel -> argumentele acceptate de codul tau.
# "Mixup labels" este mapat la mixup_type="linear", deoarece functia mixup_2_dataset
# combina inputul si calculeaza loss mixt intre labelurile source/target.
EXPERIMENTS: List[Dict[str, object]] = [
    # key,                                   mixup,     W,    pretrained, freeze
    {"key": "mixup_labels_W50_scratch",      "mixup": "linear",   "warmup": 50, "pretrained": False, "freeze": None},
    {"key": "mixup_pixels_W50_scratch",      "mixup": "pixels",   "warmup": 50, "pretrained": False, "freeze": None},
    {"key": "mixup_pixels_W20_scratch",      "mixup": "pixels",   "warmup": 20, "pretrained": False, "freeze": None},
    {"key": "mixup_labels_W50_pretrained",   "mixup": "linear",   "warmup": 50, "pretrained": True,  "freeze": None},
    {"key": "mixup_pixels_W20_pretrained",   "mixup": "pixels",   "warmup": 20, "pretrained": True,  "freeze": None},
    {"key": "scratch_no_mixup",              "mixup": None,       "warmup": None, "pretrained": False, "freeze": None},
    {"key": "mixup_labels_W20_pretrained",   "mixup": "linear",   "warmup": 20, "pretrained": True,  "freeze": None},
    {"key": "mixup_labels_W20_scratch",      "mixup": "linear",   "warmup": 20, "pretrained": False, "freeze": None},
    {"key": "pretrained_finetune_all",       "mixup": None,       "warmup": None, "pretrained": True,  "freeze": None},
    {"key": "mixup_temporal_W50_scratch",    "mixup": "temporal", "warmup": 50, "pretrained": False, "freeze": None},
    {"key": "mixup_temporal_W20_pretrained", "mixup": "temporal", "warmup": 20, "pretrained": True,  "freeze": None},
    {"key": "mixup_bands_W20_pretrained",    "mixup": "bands",    "warmup": 20, "pretrained": True,  "freeze": None},
    {"key": "mixup_bands_W50_scratch",       "mixup": "bands",    "warmup": 50, "pretrained": False, "freeze": None},
    {"key": "mixup_bands_W20_scratch",       "mixup": "bands",    "warmup": 20, "pretrained": False, "freeze": None},
    {"key": "pretrained_freeze_spatial",     "mixup": None,       "warmup": None, "pretrained": True,  "freeze": "spatial"},
    {"key": "pretrained_freeze_temporal",    "mixup": None,       "warmup": None, "pretrained": True,  "freeze": "temporal"},
    {"key": "pretrained_freeze_all_encoder", "mixup": None,       "warmup": None, "pretrained": True,  "freeze": "all"},
]


PATCH_MARKER = "# PATCHED_BY_run_slovakia_transfer_experiments__mixup_none_support"


def patch_training_script(train_script: Path, patched_script: Optional[Path] = None) -> Path:
    """Creeaza o copie a scriptului care accepta --mixup_type none."""
    train_script = train_script.expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(f"Nu exista train-script: {train_script}")

    if patched_script is None:
        patched_script = train_script.with_name(train_script.stem + "_patched_for_slovakia_transfer.py")
    else:
        patched_script = patched_script.expanduser().resolve()

    text = train_script.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        patched_script.write_text(text, encoding="utf-8")
        return patched_script

    old_choices = "choices=['linear', 'temporal', 'bands', 'pixels'],"
    new_choices = "choices=['linear', 'temporal', 'bands', 'pixels', 'none'],"
    if old_choices not in text:
        raise RuntimeError(
            "Nu am gasit linia cu choices pentru --mixup_type. "
            "Verifica manual parser.add_argument('--mixup_type', ...)."
        )
    text = text.replace(old_choices, new_choices, 1)

    old_config_line = "    config = vars(config)\n"
    injected = (
        "    config = vars(config)\n\n"
        f"    {PATCH_MARKER}\n"
        "    if config.get('mixup_type') == 'none':\n"
        "        config['mixup_type'] = None\n"
    )
    if old_config_line not in text:
        raise RuntimeError("Nu am gasit linia `config = vars(config)` pentru patch.")
    text = text.replace(old_config_line, injected, 1)

    patched_script.write_text(text, encoding="utf-8")
    return patched_script


def shell(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def selected_experiments(only: Optional[List[str]]) -> List[Dict[str, object]]:
    if not only:
        return EXPERIMENTS
    wanted = set(only)
    existing = {str(exp["key"]) for exp in EXPERIMENTS}
    missing = wanted - existing
    if missing:
        raise ValueError(
            "Experimente inexistente in --only: " + ", ".join(sorted(missing)) +
            "\nValori disponibile: " + ", ".join(sorted(existing))
        )
    return [exp for exp in EXPERIMENTS if str(exp["key"]) in wanted]


def common_train_args(args: argparse.Namespace, wandb_name: str) -> List[str]:
    return [
        "--wandb_name", wandb_name,
        "--res_dir", str(args.res_dir),
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--rdm_seed", str(args.seed),
        "--num_classes", "20",
        "--mlp4", "[128,64,32,20]",
    ]


def source_pretrain_checkpoint(args: argparse.Namespace) -> Path:
    if args.source_ckpt:
        return Path(args.source_ckpt).expanduser().resolve()
    return Path(args.res_dir).expanduser().resolve() / args.source_wandb_name / "model.pth.tar"


def build_source_pretrain_command(args: argparse.Namespace, script_path: Path) -> List[str]:
    """Scratch training pe SOURCE 2020-2024. Acest checkpoint devine pretraining intern."""
    epochs = args.source_epochs if args.source_epochs is not None else args.epochs
    cmd = [sys.executable, str(script_path)]
    cmd += common_train_args(args, args.source_wandb_name)
    cmd += [
        "--epochs", str(epochs),
        "--mixup_type", "none",
        "--dataset_pastis", str(args.source_root),
        "--pastis_meanstd", args.source_meanstd,
        "--pastis_labels", args.source_labels,
        "--pastis_subclasses", args.source_subclasses,
        "--dataset_slovakia", "",
        "--resume_path", "",
        "--resume_mode", "finetune",
    ]
    if args.extra_args:
        cmd += args.extra_args
    return cmd


def build_target_experiment_command(
    args: argparse.Namespace,
    script_path: Path,
    exp: Dict[str, object],
    ckpt_path: Path,
) -> List[str]:
    key = str(exp["key"])
    mixup = exp["mixup"]
    warmup = exp["warmup"]
    pretrained = bool(exp["pretrained"])
    freeze = exp["freeze"]

    wandb_name = f"target_{args.target_year}_from_2020_2024_128_{key}"

    cmd = [sys.executable, str(script_path)]
    cmd += common_train_args(args, wandb_name)
    cmd += [
        "--epochs", str(args.epochs),
        "--dataset_slovakia", str(args.target_root),
        "--slovakia_meanstd", args.target_meanstd,
        "--slovakia_labels", args.target_labels,
        "--slovakia_subclasses", args.target_subclasses,
    ]

    if mixup is None:
        # Fara mixup: target-only. Nu punem source in train_loader.
        cmd += [
            "--mixup_type", "none",
            "--dataset_pastis", "",
        ]
    else:
        # Cu mixup: source 2020-2024 + target 2025.
        cmd += [
            "--dataset_pastis", str(args.source_root),
            "--pastis_meanstd", args.source_meanstd,
            "--pastis_labels", args.source_labels,
            "--pastis_subclasses", args.source_subclasses,
            "--mixup_type", str(mixup),
            "--mixup_warmup_epochs", str(warmup),
        ]

    if pretrained:
        cmd += ["--resume_path", str(ckpt_path), "--resume_mode", "finetune"]
    else:
        cmd += ["--resume_path", "", "--resume_mode", "finetune"]

    if freeze:
        cmd += ["--freeze_encoder", str(freeze)]

    if args.extra_args:
        cmd += args.extra_args

    return cmd


def run_command(cmd: List[str], env: dict, dry_run: bool) -> int:
    print(shell(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_all(args: argparse.Namespace) -> int:
    train_script = Path(args.train_script)
    script_path = patch_training_script(train_script, Path(args.patched_script) if args.patched_script else None)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.wandb_offline:
        env["WANDB_MODE"] = "offline"

    ckpt_path = source_pretrain_checkpoint(args)
    failures: List[str] = []

    print(f"[Runner] Script folosit: {script_path}")
    print(f"[Runner] SOURCE: {args.source_root}")
    print(f"[Runner] TARGET: {args.target_root}")
    print(f"[Runner] Checkpoint pretraining intern: {ckpt_path}")

    # 1) SOURCE scratch pretraining
    source_done = ckpt_path.exists()
    if args.skip_source_pretrain:
        print("\n[Source pretrain] Sar peste antrenarea source deoarece ai setat --skip-source-pretrain.")
    elif args.skip_existing and source_done:
        print(f"\n[Source pretrain] Skip, checkpoint existent: {ckpt_path}")
    else:
        print("\n[Source pretrain] Scratch pe SOURCE 2020-2024")
        cmd = build_source_pretrain_command(args, script_path)
        code = run_command(cmd, env, args.dry_run)
        if code != 0:
            print(f"[Eroare] Source pretrain a esuat cu codul {code}")
            return code

    if not args.dry_run and not ckpt_path.exists():
        raise FileNotFoundError(
            f"Nu gasesc checkpointul pentru experimentele pretrained: {ckpt_path}\n"
            "Verifica daca source pretrain a salvat model.pth.tar sau da explicit --source-ckpt."
        )

    # 2) TARGET experiments
    exps = selected_experiments(args.only)
    print(f"\n[Target experiments] {len(exps)} experimente pe TARGET {args.target_year}")

    for exp in exps:
        key = str(exp["key"])
        run_name = f"target_{args.target_year}_from_2020_2024_128_{key}"
        run_dir = Path(args.res_dir) / run_name
        done_file = run_dir / "overall.json"

        if args.skip_existing and done_file.exists():
            print(f"\n[Skip] {key} exista deja: {done_file}")
            continue

        print(f"\n[Run target] {key}")
        cmd = build_target_experiment_command(args, script_path, exp, ckpt_path)
        code = run_command(cmd, env, args.dry_run)
        if code != 0:
            msg = f"{key} a esuat cu codul {code}"
            failures.append(msg)
            print(f"[Eroare] {msg}")
            if not args.continue_on_error:
                return code

    if failures:
        print("\nExperimente esuate:")
        for failure in failures:
            print(" - " + failure)
        return 1

    print("\nWorkflow finalizat.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runner pentru transfer Slovakia 2020-2024 -> Slovakia 2025.",
    )
    parser.add_argument("--train-script", required=True, help="Path catre scriptul tau de training .py")
    parser.add_argument("--patched-script", default="", help="Unde salvez copia patch-uita. Gol = langa train-script.")

    # SOURCE = 2020-2024
    parser.add_argument("--source-root", default="/home/mnegru/Adelina/Experiment_2/PixelSet-Slovakia-2020-2024")
    parser.add_argument("--source-meanstd", default="2020-2024-Slovakia-meanstd.pkl")
    parser.add_argument("--source-labels", default="CODE_GROUP")
    parser.add_argument("--source-subclasses", default=CLASSES_0_19)
    parser.add_argument("--source-wandb-name", default="source_2020_2024_scratch_pretrain_128")
    parser.add_argument("--source-epochs", type=int, default=None, help="Epoci pentru pretraining source. Gol = aceleasi ca --epochs.")
    parser.add_argument("--source-ckpt", default="", help="Checkpoint source existent. Gol = <res-dir>/<source-wandb-name>/model.pth.tar")
    parser.add_argument("--skip-source-pretrain", action="store_true", help="Nu mai ruleaza pretrainingul source; foloseste --source-ckpt sau checkpointul implicit.")

    # TARGET = 2025
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--target-root", default="/home/mnegru/Adelina/Experiment_2/PixelSet-Slovakia-2025")
    parser.add_argument("--target-meanstd", default="2025-Slovakia-meanstd.pkl")
    parser.add_argument("--target-labels", default="CODE_GROUP")
    parser.add_argument("--target-subclasses", default=CLASSES_0_19)

    # Training params
    parser.add_argument("--res-dir", default="./results", help="Folder radacina pentru rezultate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)

    # Runner behavior
    parser.add_argument("--wandb-offline", action="store_true", help="Seteaza WANDB_MODE=offline")
    parser.add_argument("--skip-existing", action="store_true", help="Sare peste run-uri cu overall.json existent si peste source ckpt existent")
    parser.add_argument("--continue-on-error", action="store_true", help="Continua chiar daca un run esueaza")
    parser.add_argument("--dry-run", action="store_true", help="Afiseaza comenzile fara sa le ruleze")
    parser.add_argument("--only", nargs="+", default=None, help="Ruleaza doar anumite chei de experiment")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER,
                        help="Argumente extra trimise scriptului de training. Pune-le la final dupa --extra-args")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_all(parse_args()))
