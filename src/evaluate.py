import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import parse_args_and_get_config
from src.data import get_dataloaders, normalize_answer
from src.model import build_model
from src.train import get_artifact_paths


def evaluate_model_detailed(
    model: torch.nn.Module,
    val_loader,
    idx2ans: List[str],
    device: torch.device
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()

    records = []
    type_scores = defaultdict(list)
    overall_scores = []

    with torch.no_grad():
        for batch in val_loader:
            img_feat = batch["image_feature"].to(device)
            questions = batch["question"].to(device)
            lengths = batch["length"].to(device)

            logits = model(img_feat, questions, lengths=lengths)
            probs = F.softmax(logits, dim=-1)

            max_probs, pred_indices = probs.max(dim=-1)

            pred_indices = pred_indices.cpu().tolist()
            max_probs = max_probs.cpu().tolist()

            batch_qids = batch["question_id"]
            batch_img_ids = batch["image_id"]
            batch_qtexts = batch["question_text"]
            batch_atypes = batch["answer_type"]
            batch_answers = batch["all_answers"]

            for qid, img_id, q_text, a_type, answers, pred_idx, conf in zip(
                batch_qids, batch_img_ids, batch_qtexts, batch_atypes, batch_answers, pred_indices, max_probs
            ):
                # Calculate normalized answer and accuracy
                pred_ans_str = normalize_answer(idx2ans[pred_idx])
                normalized_answers = [normalize_answer(a) for a in answers]

                matches = sum(1 for na in normalized_answers if na == pred_ans_str)
                score = min(matches / 3.0, 1.0)
                is_correct = score >= 0.5

                most_common_gt = Counter(normalized_answers).most_common(1)[0][0]

                records.append({
                    "question_id": qid,
                    "image_id": img_id,
                    "question": q_text,
                    "predicted_answer": pred_ans_str,
                    "ground_truth": most_common_gt,
                    "all_ground_truths": ";".join(answers),
                    "confidence": round(conf, 4),
                    "vqa_score": round(score, 4),
                    "is_correct": bool(is_correct),
                    "answer_type": a_type
                })

                type_scores[a_type].append(score)
                overall_scores.append(score)

    df_preds = pd.DataFrame(records)

    metrics = {
        "overall": (sum(overall_scores) / len(overall_scores)) * 100.0 if overall_scores else 0.0
    }
    for a_type, scores in type_scores.items():
        metrics[a_type] = (sum(scores) / len(scores)) * 100.0 if scores else 0.0

    return df_preds, metrics


def plot_learning_curves(histories: List[Tuple[str, str, Any]], save_path: str):
    valid_histories = []
    for item in histories:
        label, path, style = item
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_histories.append((label, data, style))

    if not valid_histories:
        print("[!] No training history files found to plot learning curves.")
        return

    plt.figure(figsize=(12, 5))

    # 1. Loss subplot: train (solid) and val (dashed) for each variant
    plt.subplot(1, 2, 1)
    for label, h, style in valid_histories:
        color = style.get("color", style) if isinstance(style, dict) else style
        epochs = h.get("epochs", list(range(1, len(h.get("val_loss", [])) + 1)))
        if "train_loss" in h and len(h["train_loss"]) > 0:
            plt.plot(epochs, h["train_loss"], linestyle="-", marker="o", color=color, label=f"{label} Train")
        if "val_loss" in h and len(h["val_loss"]) > 0:
            plt.plot(epochs, h["val_loss"], linestyle="--", marker="s", color=color, label=f"{label} Val")

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 2. Accuracy subplot: val_acc for each variant
    plt.subplot(1, 2, 2)
    for label, h, style in valid_histories:
        color = style.get("color", style) if isinstance(style, dict) else style
        epochs = h.get("epochs", list(range(1, len(h.get("val_acc", [])) + 1)))
        if "val_acc" in h and len(h["val_acc"]) > 0:
            plt.plot(epochs, h["val_acc"], linestyle="-", marker="o", color=color, label=label)

    plt.title("VQA Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Learning curves saved to: {save_path}")


def display_examples(df: pd.DataFrame, num_examples: int = 10):
    print("\n" + "=" * 80)
    print(f" First {num_examples} Correct Predictions (VQA Accuracy >= 0.5)")
    print("=" * 80)
    correct_samples = df[df["is_correct"] == True].head(num_examples)
    for idx, (_, row) in enumerate(correct_samples.iterrows(), 1):
        print(f"[{idx:02d}] Image ID: {row['image_id']} | Type: {row['answer_type']}")
        print(f"     Question:     {row['question']}")
        print(f"     Prediction:   {row['predicted_answer']} (Confidence: {row['confidence']:.2%})")
        print(f"     Ground Truth: {row['ground_truth']} (All Answers: {row['all_ground_truths']})")
        print(f"     VQA Score:    {row['vqa_score']:.2f}\n")

    print("=" * 80)
    print(f" First {num_examples} Incorrect Predictions (VQA Accuracy < 0.5)")
    print("=" * 80)
    error_samples = df[df["is_correct"] == False].head(num_examples)
    for idx, (_, row) in enumerate(error_samples.iterrows(), 1):
        print(f"[{idx:02d}] Image ID: {row['image_id']} | Type: {row['answer_type']}")
        print(f"     Question:     {row['question']}")
        print(f"     Prediction:   {row['predicted_answer']} (Confidence: {row['confidence']:.2%})")
        print(f"     Ground Truth: {row['ground_truth']} (All Answers: {row['all_ground_truths']})")
        print(f"     VQA Score:    {row['vqa_score']:.2f}\n")


def run_evaluation(cfg):
    print("\n=======================================================")
    print("Running Model Evaluation (Evaluation & Results)")
    print("=======================================================")

    original_fusion = cfg.model.get("fusion_method", "mul")
    output_dir = Path(cfg.paths.output_dir if "output_dir" in cfg.paths else cfg.output_dir)

    variants = [
        ("VQA Baseline (mul)", "vqa", "mul"),
        ("VQA Baseline (concat)", "vqa", "concat"),
        ("Question-Only (Ablation)", "question_only", None)
    ]

    # Check which checkpoints exist
    available_variants = []
    for label, model_type, fusion in variants:
        if fusion is not None:
            cfg.model.fusion_method = fusion
        ckpt_path, _ = get_artifact_paths(cfg, model_type)
        if os.path.exists(ckpt_path):
            available_variants.append((label, model_type, fusion, ckpt_path))
    cfg.model.fusion_method = original_fusion

    if not available_variants:
        print("[-] No model checkpoints found for evaluation. Exiting.")
        return

    _, val_loader, word2idx, _, ans2idx, idx2ans = get_dataloaders(cfg)

    table_rows = []
    db_columns = ["question_id", "image_id", "question", "predicted_answer", "confidence", "is_correct", "answer_type"]

    for idx, (label, model_type, fusion, ckpt_path) in enumerate(available_variants, 1):
        print(f"\n[{idx}/{len(available_variants)}] Evaluating {label}: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=cfg.resolved_device, weights_only=False)

        if fusion is not None:
            cfg.model.fusion_method = fusion
        try:
            model = build_model(cfg, len(word2idx), len(ans2idx), model_type=model_type).to(cfg.resolved_device)
        finally:
            cfg.model.fusion_method = original_fusion

        model.load_state_dict(checkpoint["model_state_dict"])

        df_preds, metrics = evaluate_model_detailed(model, val_loader, idx2ans, cfg.resolved_device)

        if fusion == "mul":
            df_preds[db_columns].to_csv(cfg.paths.predictions_csv, index=False, encoding="utf-8")
            print(f"[+] Predictions saved to: {cfg.paths.predictions_csv}")
            display_examples(df_preds, num_examples=10)
        elif fusion == "concat":
            concat_preds_path = output_dir / "predictions_concat.csv"
            df_preds[db_columns].to_csv(concat_preds_path, index=False, encoding="utf-8")
            print(f"[+] Predictions saved to: {concat_preds_path}")

        best_epoch = checkpoint.get("epoch")
        for cat in ["overall", "yes/no", "number", "other"]:
            table_rows.append({
                "Model": label,
                "Answer Type": cat,
                "Accuracy (%)": round(metrics.get(cat, 0.0), 2),
                "Best Epoch": best_epoch
            })

    df_metrics = pd.DataFrame(table_rows, columns=["Model", "Answer Type", "Accuracy (%)", "Best Epoch"])
    print("\n" + "=" * 60)
    print("Final Model Evaluation Results (IMRAD: Results)")
    print("=" * 60)
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(cfg.paths.metrics_table_csv, index=False, encoding="utf-8")
    print(f"\n[+] Metrics table saved to: {cfg.paths.metrics_table_csv}")

    # Plot learning curves for all variants
    cfg.model.fusion_method = "mul"
    _, h_mul_path = get_artifact_paths(cfg, "vqa")
    cfg.model.fusion_method = "concat"
    _, h_concat_path = get_artifact_paths(cfg, "vqa")
    _, h_q_path = get_artifact_paths(cfg, "question_only")
    cfg.model.fusion_method = original_fusion

    histories = [
        ("VQA Baseline (mul)", h_mul_path, "blue"),
        ("VQA Baseline (concat)", h_concat_path, "green"),
        ("Question-Only (Ablation)", h_q_path, "red"),
    ]
    plot_learning_curves(histories, cfg.paths.learning_curves_png)


def main():
    cfg = parse_args_and_get_config()
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
