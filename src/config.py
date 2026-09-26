# ==============================================================================
# src/config.py
# Loading and managing project configuration, handling paths, seeds, and devices.
# ==============================================================================

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ConfigDict(dict):
    """Dictionary allowing attribute-style access like: cfg.model.dropout_rate."""
    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[name] = val
            return val
        except KeyError:
            raise AttributeError(f"Configuration does not contain parameter: '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def set_seed(seed: int = 42) -> None:
    """Set random seed across all devices for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_setting: str = "auto") -> torch.device:
    """Automatically determine device availability or return specified device."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def resolve_kaggle_paths(cfg: ConfigDict) -> ConfigDict:
    """
    Automatically resolve paths to VQA v2 files and COCO 2014 directories in /kaggle/input.
    Allows running the pipeline when multiple Kaggle datasets are attached.
    """
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        return cfg

    # 1. Search for question and annotation files
    for p in input_dir.rglob("*train2014_questions.json"):
        cfg.data.train_questions = str(p)
        break
    for p in input_dir.rglob("*train2014_annotations.json"):
        cfg.data.train_annotations = str(p)
        break
    for p in input_dir.rglob("*val2014_questions.json"):
        cfg.data.val_questions = str(p)
        break
    for p in input_dir.rglob("*val2014_annotations.json"):
        cfg.data.val_annotations = str(p)
        break

    # 2. Search for image directories
    for d in input_dir.rglob("train2014"):
        if d.is_dir() and any(d.glob("*.jpg")):
            cfg.data.train_img_dir = str(d)
            break
    for d in input_dir.rglob("val2014"):
        if d.is_dir() and any(d.glob("*.jpg")):
            cfg.data.val_img_dir = str(d)
            break

    return cfg


def load_config(
    env: Optional[str] = None,
    config_path: Optional[str] = None,
    seed: Optional[int] = None
) -> ConfigDict:
    if config_path is None:
        if env is None:
            env = "kaggle" if os.path.exists("/kaggle/input") else "local"
        
        project_root = Path(__file__).resolve().parent.parent
        config_path = str(project_root / "configs" / f"{env}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    cfg = ConfigDict(raw_cfg)

    # Auto-resolve paths on Kaggle
    if cfg.get("env") == "kaggle" or os.path.exists("/kaggle/input"):
        cfg = resolve_kaggle_paths(cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        cfg.seed = seed
        run_dir = output_dir / f"seed_{seed}"
    else:
        if "seed" not in cfg:
            cfg.seed = 42
        run_dir = output_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.paths = ConfigDict({
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "train_features_h5": str(output_dir / "train_img_features.h5"),
        "val_features_h5": str(output_dir / "val_img_features.h5"),
        "vocab_json": str(output_dir / "vocab.json"),
        "best_model_pth": str(run_dir / "best_model.pth"),
        "best_question_only_pth": str(run_dir / "best_question_only_model.pth"),
        "predictions_csv": str(run_dir / "predictions.csv"),
        "metrics_table_csv": str(run_dir / "metrics_table.csv"),
        "learning_curves_png": str(run_dir / "learning_curves.png"),
        "metrics_history_json": str(run_dir / "metrics_history.json")
    })

    actual_device = get_device(cfg.get("device", "auto"))
    cfg.resolved_device = actual_device

    set_seed(cfg.seed)

    return cfg


def parse_args_and_get_config() -> ConfigDict:
    parser = argparse.ArgumentParser(description="VQA Baseline Pipeline")
    parser.add_argument("--env", type=str, choices=["local", "kaggle"], default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args, _ = parser.parse_known_args()
    return load_config(env=args.env, config_path=args.config, seed=args.seed)


if __name__ == "__main__":
    config = parse_args_and_get_config()
    print("=== Config Test ===")
    print(f"Env: {config.env}")
    print(f"Device: {config.resolved_device}")
    print(f"Output: {config.output_dir}")
