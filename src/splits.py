"""
src/splits.py
Train/dev/test dataset splitting protocol (ADR-011).
Handles deterministic split generation and manifest persistence.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_dev_image_ids(image_ids: Iterable[int], dev_fraction: float, seed: int) -> Set[int]:
    """
    Select a deterministic dev split of image IDs.
    Sorts unique IDs, shuffles with random.Random(seed),
    and takes the first round(len * dev_fraction) IDs (at least 1 if dev_fraction > 0 and IDs non-empty).

    Args:
        image_ids: Iterable of integer image IDs.
        dev_fraction: Fraction of unique image IDs to assign to dev set.
        seed: Random seed for shuffling.

    Returns:
        Set of integer image IDs assigned to dev set.
    """
    unique_ids = sorted({int(x) for x in image_ids})
    n = len(unique_ids)
    if n == 0 or dev_fraction <= 0.0:
        return set()

    k = round(n * dev_fraction)
    if dev_fraction > 0.0 and k < 1:
        k = 1
    k = min(n, k)

    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    return set(unique_ids[:k])


def save_split_manifest(split: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save split dictionary to JSON manifest with dev_image_ids as a sorted list.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = dict(split)
    manifest_data["dev_image_ids"] = sorted(list(split["dev_image_ids"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def load_split_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load split dictionary from JSON manifest with dev_image_ids converted to a set.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[Data Error] Split manifest file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["dev_image_ids"] = set(data["dev_image_ids"])
    return data


def build_split(cfg: Any) -> Dict[str, Any]:
    """
    Build or load train/dev split manifest according to experimental protocol (ADR-011).
    Reads cfg.data.train_questions, computes dev image IDs using split_seed and dev_image_fraction,
    and manages split_manifest.json in cfg.output_dir.

    If manifest exists and matches config parameters, it is loaded and reused without recomputation.
    If parameters do not match, ValueError is raised.
    """
    if isinstance(cfg, dict):
        data_cfg = cfg.get("data", {})
        output_dir = cfg.get("output_dir", "./outputs")
    else:
        data_cfg = getattr(cfg, "data", {})
        output_dir = getattr(cfg, "output_dir", "./outputs")

    if isinstance(data_cfg, dict):
        split_seed = data_cfg.get("split_seed", 2026)
        dev_image_fraction = data_cfg.get("dev_image_fraction", 0.1)
        train_questions_path = data_cfg.get("train_questions")
    else:
        split_seed = getattr(data_cfg, "split_seed", 2026)
        dev_image_fraction = getattr(data_cfg, "dev_image_fraction", 0.1)
        train_questions_path = getattr(data_cfg, "train_questions")

    manifest_path = Path(output_dir) / "split_manifest.json"

    if manifest_path.exists():
        manifest = load_split_manifest(manifest_path)
        m_seed = manifest.get("split_seed")
        m_frac = manifest.get("dev_image_fraction")
        if m_seed != split_seed or (m_frac is not None and abs(m_frac - dev_image_fraction) > 1e-6):
            raise ValueError(
                f"Existing split manifest at {manifest_path} has conflicting parameters: "
                f"split_seed={m_seed} (config={split_seed}), "
                f"dev_image_fraction={m_frac} (config={dev_image_fraction})."
            )
        return manifest

    if not train_questions_path or not Path(train_questions_path).exists():
        raise FileNotFoundError(f"[Data Error] Train questions file not found: {train_questions_path}")

    with open(train_questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)["questions"]

    all_image_ids = {q["image_id"] for q in questions_data}
    dev_image_ids = make_dev_image_ids(all_image_ids, dev_fraction=dev_image_fraction, seed=split_seed)

    n_dev_questions = sum(1 for q in questions_data if q["image_id"] in dev_image_ids)
    n_train_questions = len(questions_data) - n_dev_questions
    n_dev_images = len(dev_image_ids)
    n_train_images = len(all_image_ids) - n_dev_images

    split = {
        "dev_image_ids": dev_image_ids,
        "split_seed": split_seed,
        "dev_image_fraction": dev_image_fraction,
        "n_train_images": n_train_images,
        "n_dev_images": n_dev_images,
        "n_train_questions": n_train_questions,
        "n_dev_questions": n_dev_questions,
    }

    save_split_manifest(split, manifest_path)
    return split
