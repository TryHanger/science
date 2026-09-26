"""
scripts/coverage_report.py
Verification and reporting of image feature coverage for VQA datasets (ADR-011 item 6).
Computes coverage for train and val splits, verifies missing fraction against threshold,
and writes coverage_report.json and coverage_report.csv to the output directory.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import h5py
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_coverage(
    questions_path: Union[str, Path],
    h5_path: Optional[Union[str, Path]] = None,
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute image feature coverage for a given split questions JSON and HDF5 feature store.

    Args:
        questions_path: Path to VQA questions JSON file.
        h5_path: Optional path to HDF5 feature cache file.
        max_questions: Optional limit on the number of questions evaluated.

    Returns:
        Dictionary with:
          - unique_images: Count of unique image_ids in evaluated questions.
          - missing_images: Count of unique image_ids not found in HDF5 keys.
          - missing_pct: Percentage of unique images missing (4 decimal places).
          - total_questions: Count of evaluated questions.
          - affected_questions: Count of questions referring to missing images.
          - affected_questions_pct: Percentage of affected questions (4 decimal places).
    """
    q_path = Path(questions_path)
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    with open(q_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError(f"Unrecognized questions format in {q_path}")

    if max_questions is not None and max_questions > 0:
        questions = questions[:max_questions]

    total_questions = len(questions)
    unique_image_ids = {q["image_id"] for q in questions if "image_id" in q}
    unique_images = len(unique_image_ids)

    h5_p = Path(h5_path) if h5_path is not None else None
    if h5_p is not None and h5_p.is_file():
        with h5py.File(h5_p, "r") as h5_f:
            h5_keys = set(h5_f.keys())
        missing_image_ids = {
            iid
            for iid in unique_image_ids
            if (str(iid) not in h5_keys) and (iid not in h5_keys)
        }
    else:
        missing_image_ids = set(unique_image_ids)

    missing_images = len(missing_image_ids)
    affected_questions = sum(1 for q in questions if q.get("image_id") in missing_image_ids)

    missing_pct = (
        round((missing_images / unique_images * 100.0), 4) if unique_images > 0 else 0.0
    )
    affected_questions_pct = (
        round((affected_questions / total_questions * 100.0), 4)
        if total_questions > 0
        else 0.0
    )

    return {
        "unique_images": unique_images,
        "missing_images": missing_images,
        "missing_pct": missing_pct,
        "total_questions": total_questions,
        "affected_questions": affected_questions,
        "affected_questions_pct": affected_questions_pct,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify image feature coverage and export coverage report (ADR-011)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-missing-frac",
        type=float,
        default=None,
        help="Maximum allowed fraction of missing images (default: cfg.data.max_missing_feature_frac or 0.01).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save coverage_report.json and coverage_report.csv (default: cfg.output_dir).",
    )
    return parser.parse_args(argv)


def run_coverage_report(
    config_path: str,
    max_missing_frac_override: Optional[float] = None,
    out_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path=config_path)

    if max_missing_frac_override is not None:
        max_missing_frac = float(max_missing_frac_override)
    elif (
        hasattr(cfg, "data")
        and "max_missing_feature_frac" in cfg.data
        and cfg.data.max_missing_feature_frac is not None
    ):
        max_missing_frac = float(cfg.data.max_missing_feature_frac)
    else:
        max_missing_frac = 0.01

    out_dir = Path(out_dir_override if out_dir_override is not None else cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold_pct = round(max_missing_frac * 100.0, 4)

    # Legacy config question limits
    num_train = (
        cfg.data.get("num_train_questions", None)
        if hasattr(cfg.data, "get")
        else getattr(cfg.data, "num_train_questions", None)
    )
    if num_train == 0:
        num_train = None

    num_val = (
        cfg.data.get("num_val_questions", None)
        if hasattr(cfg.data, "get")
        else getattr(cfg.data, "num_val_questions", None)
    )
    if num_val == 0:
        num_val = None

    train_h5 = Path(cfg.paths.train_features_h5)
    if not train_h5.exists() and (out_dir / "train_img_features.h5").exists():
        train_h5 = out_dir / "train_img_features.h5"

    val_h5 = Path(cfg.paths.val_features_h5)
    if not val_h5.exists() and (out_dir / "val_img_features.h5").exists():
        val_h5 = out_dir / "val_img_features.h5"

    splits_definitions = [
        ("train2014", cfg.data.train_questions, train_h5, num_train),
        ("val2014", cfg.data.val_questions, val_h5, num_val),
    ]

    splits_report = {}
    all_passed = True

    for split_name, q_path, h_path, max_q in splits_definitions:
        cov = compute_coverage(q_path, h_path, max_questions=max_q)
        frac = (
            (cov["missing_images"] / cov["unique_images"])
            if cov["unique_images"] > 0
            else 1.0
        )
        split_passed = bool(cov["unique_images"] > 0 and frac <= max_missing_frac)
        if not split_passed:
            all_passed = False

        splits_report[split_name] = {
            "unique_images": cov["unique_images"],
            "missing_images": cov["missing_images"],
            "missing_pct": cov["missing_pct"],
            "total_questions": cov["total_questions"],
            "affected_questions": cov["affected_questions"],
            "affected_questions_pct": cov["affected_questions_pct"],
            "threshold_pct": threshold_pct,
            "passed": split_passed,
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "max_missing_frac": max_missing_frac,
        "splits": splits_report,
        "passed": all_passed,
    }

    # Save JSON report
    json_path = out_dir / "coverage_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved coverage report JSON to: {json_path}")

    # Save CSV report
    csv_path = out_dir / "coverage_report.csv"
    fieldnames = [
        "split",
        "unique_images",
        "missing_images",
        "missing_pct",
        "total_questions",
        "affected_questions",
        "affected_questions_pct",
        "threshold_pct",
        "passed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, s_data in splits_report.items():
            row = {"split": split_name}
            row.update(s_data)
            writer.writerow(row)
    print(f"[+] Saved coverage report CSV to: {csv_path}")

    # Print summary table
    table_rows = []
    for split_name, s_data in splits_report.items():
        row = {"split": split_name}
        row.update(s_data)
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    print("\n=== Feature Coverage Summary ===")
    print(df_table.to_string(index=False))
    print(f"\nOverall Status: {'PASSED' if all_passed else 'FAILED'}\n")

    return report


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    report = run_coverage_report(
        config_path=args.config,
        max_missing_frac_override=args.max_missing_frac,
        out_dir_override=args.out_dir,
    )
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
