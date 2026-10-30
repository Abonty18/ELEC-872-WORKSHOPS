"""Run ONLY the experiments requested in Workshop.pdf, with explicit baselines.

python workshop.py --list
python workshop.py --part ref
python workshop.py --part A
python workshop.py --summary
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


# name, part, comparison baseline, changed options
# D needs its own comparison baseline because the paper reference already has
# weight decay=0 and augmentation disabled. This adds one necessary baseline.
RUNS = [
    ("ref", "ref", None, []),
    ("A_cnn", "A", "ref", ["--model", "cnn"]),
    ("A_gru", "A", "ref", ["--rnn", "gru"]),
    ("B_conv1", "B", "ref", ["--n-conv", "1"]),
    ("B_conv2", "B", "ref", ["--n-conv", "2"]),
    ("C_w20", "C", "ref", ["--window", "20"]),
    ("C_w60", "C", "ref", ["--window", "60"]),
    ("D_ref", "D", None, ["--weight-decay", "1e-4", "--augment"]),
    ("D_no_dropout", "D", "D_ref", ["--weight-decay", "1e-4", "--augment", "--dropout", "0"]),
    ("D_no_weight_decay", "D", "D_ref", ["--augment"]),
    ("D_no_augmentation", "D", "D_ref", ["--weight-decay", "1e-4"]),
    ("E_warmup5", "E", "ref", ["--warmup-epochs", "5"]),
    ("F_focal", "F", "ref", ["--loss", "focal", "--focal-gamma", "2"]),
]


def training_args(run, seed=0, epochs=100, device="auto"):
    name, _, _, extra = run
    return ["--task", "gestures", "--epochs", str(epochs), "--seed", str(seed),
            "--device", device, "--name", f"{name}_s{seed}", *extra]


def show_plan():
    print("13 runs: paper reference, A/B/C/E/F, and four runs for Part D.")
    print("All use seed=0, 100 epochs maximum, patience=20 unless overridden.")
    print("Part D variants must be compared with D_ref, not the paper reference.\n")
    for name, part, baseline, extra in RUNS:
        print(f"{name:23s} baseline={baseline or '(baseline)':10s} {' '.join(extra) or '(unchanged)'}")


def collect_summary():
    definitions = {name: (part, baseline) for name, part, baseline, _ in RUNS}
    finished = {}
    for path in sorted(Path("runs").glob("*/metrics.json")):
        data = json.loads(path.read_text())
        if "test_f1_weighted" not in data:
            continue
        variant, separator, seed = path.parent.name.rpartition("_s")
        if separator and seed.isdigit() and variant in definitions:
            finished[(variant, int(seed))] = data
    rows = []
    for name, part, baseline, _ in RUNS:
        for (variant, seed), data in sorted(finished.items()):
            if variant != name:
                continue
            cfg = data["config"]
            base = finished.get((baseline, seed)) if baseline else None
            row = {
                "run": f"{name}_s{seed}", "part": part, "seed": seed,
                "compare_with": f"{baseline}_s{seed}" if baseline else "baseline",
                "weighted_F1": data["test_f1_weighted"], "macro_F1": data["test_f1_macro"],
                "accuracy": data["test_accuracy"],
                "delta_weighted_F1": data["test_f1_weighted"] - base["test_f1_weighted"] if base else "",
                "delta_macro_F1": data["test_f1_macro"] - base["test_f1_macro"] if base else "",
                "parameters": data["params"], "best_epoch": data["best_epoch"],
                "training_minutes": data["train_minutes"],
                "window": cfg["window"], "n_conv": cfg["n_conv"],
                "dropout": cfg["dropout"], "weight_decay": cfg["weight_decay"],
                "augmentation": cfg["augment"], "loss": cfg.get("loss", "cross_entropy"),
                "warmup_epochs": cfg.get("warmup_epochs", 0),
            }
            rows.append(row)
    if not rows:
        print("No completed workshop runs found; train the reference first.")
        return
    with open("workshop_comparison.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['run']:27s} F1w={row['weighted_F1']:.4f} F1m={row['macro_F1']:.4f} "
              f"compare={row['compare_with']}")
    print("Saved workshop_comparison.csv (deltas are score differences, not percentage changes).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--part", choices=["ref", "A", "B", "C", "D", "E", "F", "all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--force", action="store_true", help="explicitly retrain completed runs")
    args = parser.parse_args(argv)
    if args.list or (not args.part and not args.summary):
        show_plan()
    if args.part:
        # Imported only when training is requested; --list needs no PyTorch.
        from train import parse_args
        for run in RUNS:
            name, part, baseline, _ = run
            if args.part not in ("all", part):
                continue
            if name == "E_warmup5" and args.epochs < 5:
                parser.error("Part E needs at least 5 epochs to complete its 5-epoch warm-up")
            options = training_args(run, args.seed, args.epochs, args.device)
            metrics = Path("runs") / f"{name}_s{args.seed}" / "metrics.json"
            if metrics.exists() and not args.force:
                completed = json.loads(metrics.read_text())
                old = completed["config"]
                requested = vars(parse_args(options))
                changed = {key: (old.get(key), value) for key, value in requested.items()
                           if old.get(key) != value}
                if "test_f1_weighted" in completed and not changed:
                    print(f"Skipping completed run {metrics.parent.name}", flush=True)
                    continue
                if "test_f1_weighted" in completed:
                    raise RuntimeError(f"{metrics.parent} has different settings {changed}; "
                                       "use a different seed/name or --force to retrain explicitly")
            if baseline and not (Path("runs") / f"{baseline}_s{args.seed}" / "metrics.json").exists():
                print(f"Note: comparison baseline {baseline}_s{args.seed} is not finished yet.", flush=True)
            print(f"\nRunning {name}; comparison baseline: {baseline or 'this is a baseline'}", flush=True)
            subprocess.run([sys.executable, "train.py", *options], check=True)
        collect_summary()
    if args.summary:
        collect_summary()


if __name__ == "__main__":
    main()
