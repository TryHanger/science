import argparse
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate VQA evaluation metrics across seeds.")
    parser.add_argument(
        "--root",
        type=str,
        default="outputs",
        help="Root directory containing seed_* subdirectories (default: outputs)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for results_table.csv (default: <root>/results_table.csv)."
    )
    return parser.parse_args(argv)


def aggregate_seeds(root_dir: Path, out_path: Optional[Path] = None) -> pd.DataFrame:
    if not root_dir.exists():
        print(f"Error: Root directory '{root_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    seed_dirs = sorted([d for d in root_dir.glob("seed_*") if d.is_dir()])
    if not seed_dirs:
        print(f"Error: No seed_* directories found in '{root_dir}'.", file=sys.stderr)
        sys.exit(1)

    metrics_files = []
    for s_dir in seed_dirs:
        csv_file = s_dir / "metrics_table.csv"
        if csv_file.exists():
            metrics_files.append((s_dir, csv_file))

    if not metrics_files:
        print(f"Error: No metrics_table.csv found in any seed_* directory under '{root_dir}'.", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for s_dir, csv_file in metrics_files:
        df = pd.read_csv(csv_file)
        if "Seed" not in df.columns:
            name_part = s_dir.name.replace("seed_", "")
            df["Seed"] = int(name_part) if name_part.isdigit() else name_part
        if "Split" not in df.columns:
            df["Split"] = "val"
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    if out_path is None:
        out_path = root_dir / "results_table.csv"
    per_seed_path = out_path.parent / "results_per_seed.csv"

    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(per_seed_path, index=False, encoding="utf-8")
    print(f"[+] Saved per-seed results to: {per_seed_path}")

    type_priority = {"overall": 0, "yes/no": 1, "number": 2, "other": 3}
    records = []

    for (model, split, a_type), grp in df_all.groupby(["Model", "Split", "Answer Type"], sort=False):
        n_seeds = int(grp["Seed"].nunique())
        acc_off = grp["Accuracy (%)"].astype(float)
        acc_simp = grp["Accuracy simplified (%)"].astype(float) if "Accuracy simplified (%)" in grp else pd.Series(dtype=float)

        std_off = round(float(acc_off.std(ddof=1)), 2) if n_seeds > 1 else np.nan
        std_simp = round(float(acc_simp.std(ddof=1)), 2) if (n_seeds > 1 and not acc_simp.empty) else np.nan

        records.append({
            "Model": model,
            "Split": split,
            "Answer Type": a_type,
            "n_seeds": n_seeds,
            "Accuracy (%) mean": round(float(acc_off.mean()), 2),
            "Accuracy (%) std": std_off,
            "Accuracy (%) min": round(float(acc_off.min()), 2),
            "Accuracy (%) max": round(float(acc_off.max()), 2),
            "Accuracy simplified (%) mean": round(float(acc_simp.mean()), 2) if not acc_simp.empty else np.nan,
            "Accuracy simplified (%) std": std_simp,
            "Accuracy simplified (%) min": round(float(acc_simp.min()), 2) if not acc_simp.empty else np.nan,
            "Accuracy simplified (%) max": round(float(acc_simp.max()), 2) if not acc_simp.empty else np.nan,
        })

    # Sort records to keep Answer Types ordered: overall, yes/no, number, other
    records.sort(key=lambda r: (r["Model"], r["Split"], type_priority.get(r["Answer Type"], 99)))

    columns = [
        "Model", "Split", "Answer Type", "n_seeds",
        "Accuracy (%) mean", "Accuracy (%) std", "Accuracy (%) min", "Accuracy (%) max",
        "Accuracy simplified (%) mean", "Accuracy simplified (%) std",
        "Accuracy simplified (%) min", "Accuracy simplified (%) max"
    ]
    df_results = pd.DataFrame(records, columns=columns)
    df_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[+] Saved aggregated results to: {out_path}")

    # Print markdown table: Model | Answer Type | Official mean ± std | Simplified mean ± std | n_seeds
    lines = [
        "| Model | Answer Type | Official mean ± std | Simplified mean ± std | n_seeds |",
        "|---|---|---|---|---|"
    ]
    for r in records:
        if pd.notna(r["Accuracy (%) std"]):
            off_str = f"{r['Accuracy (%) mean']:.2f} ± {r['Accuracy (%) std']:.2f}"
        else:
            off_str = f"{r['Accuracy (%) mean']:.2f}"

        if pd.notna(r["Accuracy simplified (%) std"]):
            simp_str = f"{r['Accuracy simplified (%) mean']:.2f} ± {r['Accuracy simplified (%) std']:.2f}"
        elif pd.notna(r["Accuracy simplified (%) mean"]):
            simp_str = f"{r['Accuracy simplified (%) mean']:.2f}"
        else:
            simp_str = "N/A"

        lines.append(f"| {r['Model']} | {r['Answer Type']} | {off_str} | {simp_str} | {r['n_seeds']} |")

    md_table = "\n".join(lines)
    print("\n" + md_table + "\n")

    return df_results


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    args = parse_args(argv)
    root_dir = Path(args.root)
    out_path = Path(args.out) if args.out else None
    return aggregate_seeds(root_dir, out_path)


if __name__ == "__main__":
    main()
