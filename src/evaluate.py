import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import parse_args_and_get_config
from src.data import get_dataloaders, load_vocabularies
from src.model import build_model


def evaluate_model_detailed(
    model: torch.nn.Module,
    val_loader,
    idx2ans: List[str],
    device: torch.device
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    ????????? ????????? ?????? ?????? ?? ????????????? ????????:
    - ??????? ???????????? ? ??????????? (???????????) ????? Softmax
    - ????????? ?????? ?? ??????? ???????: min(count / 3.0, 1.0)
    - ????????? ????????? ?? ????? ?????????????? ??? ???????? ? PostgreSQL
    - ??????? ??????? ?? ??????????: yes/no, number, other, overall
    """
    model.eval()

    records = []
    type_scores = defaultdict(list)
    overall_scores = []

    with torch.no_grad():
        for batch in val_loader:
            img_feat = batch["image_feature"].to(device)
            questions = batch["question"].to(device)

            logits = model(img_feat, questions)
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
                pred_ans_str = idx2ans[pred_idx]
                
                # ??????? ????? ?????????? ? 10 ???????? ???????????
                matches = sum(1 for h_ans in answers if h_ans.lower().strip() == pred_ans_str)
                score = min(matches / 3.0, 1.0)
                is_correct = score >= 0.5  # ??? ????????? ????? ?????? ? PostgreSQL

                # ????? ?????? ???????????? ????? ??? ???????????
                most_common_gt = Counter([a.lower().strip() for a in answers]).most_common(1)[0][0]

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


def plot_learning_curves(history_vqa_path: str, history_qonly_path: str, save_path: str):
    """
    ?????? ??????? ????????? Loss ? Accuracy ?? ?????? ??? ?????????? ? ??????? ??????.
    """
    plt.figure(figsize=(12, 5))

    # 1. Loss
    plt.subplot(1, 2, 1)
    if os.path.exists(history_vqa_path):
        with open(history_vqa_path, "r", encoding="utf-8") as f:
            h_vqa = json.load(f)
        plt.plot(h_vqa["epochs"], h_vqa["train_loss"], "b-o", label="VQA (ResNet+LSTM) Train")
        plt.plot(h_vqa["epochs"], h_vqa["val_loss"], "b--s", label="VQA (ResNet+LSTM) Val")

    if os.path.exists(history_qonly_path):
        with open(history_qonly_path, "r", encoding="utf-8") as f:
            h_q = json.load(f)
        plt.plot(h_q["epochs"], h_q["val_loss"], "r--^", label="Question-Only Val")

    plt.title("??????? ?????? (Loss) ?? ??????")
    plt.xlabel("?????")
    plt.ylabel("BCEWithLogitsLoss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 2. VQA Accuracy
    plt.subplot(1, 2, 2)
    if os.path.exists(history_vqa_path):
        plt.plot(h_vqa["epochs"], h_vqa["val_acc"], "b-o", label="VQA (Multimodal)")
    if os.path.exists(history_qonly_path):
        plt.plot(h_q["epochs"], h_q["val_acc"], "r-^", label="Question-Only (Ablation)")

    plt.title("???????? (VQA Accuracy %) ?? ?????????")
    plt.xlabel("?????")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[?] ??????? ???????? ????????? ?: {save_path}")


def display_examples(df: pd.DataFrame, num_examples: int = 10):
    """
    ??????? 10 ???????? ? 10 ????????? ????????????.
    """
    print("\n" + "=" * 80)
    print(f" ???-{num_examples} ???????? ???????? ???????????? (VQA Accuracy >= 0.5)")
    print("=" * 80)
    correct_samples = df[df["is_correct"] == True].head(num_examples)
    for idx, (_, row) in enumerate(correct_samples.iterrows(), 1):
        print(f"[{idx:02d}] Image ID: {row['image_id']} | ???: {row['answer_type']}")
        print(f"     ??????:       {row['question']}")
        print(f"     ????????????: {row['predicted_answer']} (???????????: {row['confidence']:.2%})")
        print(f"     ????????:     {row['ground_truth']} (??? ??????: {row['all_ground_truths']})")
        print(f"     VQA Score:    {row['vqa_score']:.2f}\n")

    print("=" * 80)
    print(f" ???-{num_examples} ???????? ????????? ???????????? (VQA Accuracy < 0.5)")
    print("=" * 80)
    error_samples = df[df["is_correct"] == False].head(num_examples)
    for idx, (_, row) in enumerate(error_samples.iterrows(), 1):
        print(f"[{idx:02d}] Image ID: {row['image_id']} | ???: {row['answer_type']}")
        print(f"     ??????:       {row['question']}")
        print(f"     ????????????: {row['predicted_answer']} (???????????: {row['confidence']:.2%})")
        print(f"     ????????:     {row['ground_truth']} (??? ??????: {row['all_ground_truths']})")
        print(f"     VQA Score:    {row['vqa_score']:.2f}\n")


def run_evaluation(cfg):
    print("\n=======================================================")
    print("?????? ?????? ?????? ???????? (Evaluation & Results)")
    print("=======================================================")

    # 1. ???????? ?????? ? ????????
    _, val_loader, word2idx, _, ans2idx, idx2ans = get_dataloaders(cfg)

    table_rows = []

    # 2. ?????? ??????????????? ?????? VQA
    best_vqa_pth = cfg.paths.best_model_pth
    if os.path.exists(best_vqa_pth):
        print(f"\n[1/3] ???????? ?????? ?????? VQA: {best_vqa_pth}")
        checkpoint = torch.load(best_vqa_pth, map_location=cfg.resolved_device)
        model_vqa = build_model(cfg, len(word2idx), len(ans2idx), model_type="vqa").to(cfg.resolved_device)
        model_vqa.load_state_dict(checkpoint["model_state_dict"])

        df_preds, metrics_vqa = evaluate_model_detailed(model_vqa, val_loader, idx2ans, cfg.resolved_device)

        # ?????????? ????????? ???????????? ??? PostgreSQL
        # ?????? ???????: question_id, image_id, question, predicted_answer, confidence, is_correct, answer_type
        db_columns = ["question_id", "image_id", "question", "predicted_answer", "confidence", "is_correct", "answer_type"]
        df_preds[db_columns].to_csv(cfg.paths.predictions_csv, index=False, encoding="utf-8")
        print(f"[?] ???????????? ??? PostgreSQL ????????? ?: {cfg.paths.predictions_csv}")

        # ??????? ??? ??????
        display_examples(df_preds, num_examples=10)

        for cat, score in metrics_vqa.items():
            table_rows.append({
                "Model": f"VQA Baseline ({cfg.model.get('fusion_method', 'mul')})",
                "Answer Type": cat,
                "Accuracy (%)": round(score, 2)
            })

    # 3. ?????? ??????????? ?????? Question-Only Baseline (??? ??????? ??????)
    best_q_pth = cfg.paths.best_question_only_pth
    if os.path.exists(best_q_pth):
        print(f"\n[2/3] ???????? ??????????? ?????? (Question-Only): {best_q_pth}")
        checkpoint_q = torch.load(best_q_pth, map_location=cfg.resolved_device)
        model_q = build_model(cfg, len(word2idx), len(ans2idx), model_type="question_only").to(cfg.resolved_device)
        model_q.load_state_dict(checkpoint_q["model_state_dict"])

        _, metrics_q = evaluate_model_detailed(model_q, val_loader, idx2ans, cfg.resolved_device)
        for cat, score in metrics_q.items():
            table_rows.append({
                "Model": "Question-Only (Ablation)",
                "Answer Type": cat,
                "Accuracy (%)": round(score, 2)
            })

    # 4. ???????? ??????? ?????? (IMRAD Results)
    df_metrics = pd.DataFrame(table_rows)
    print("\n" + "=" * 60)
    print("???????? ??????? ??????? ?????? (IMRAD: Results)")
    print("=" * 60)
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(cfg.paths.metrics_table_csv, index=False, encoding="utf-8")
    print(f"\n[?] ??????? ?????? ????????? ?: {cfg.paths.metrics_table_csv}")

    # 5. ?????????? ????????
    h_vqa_path = cfg.paths.metrics_history_json.replace(".json", "_vqa.json")
    h_q_path = cfg.paths.metrics_history_json.replace(".json", "_question_only.json")
    plot_learning_curves(h_vqa_path, h_q_path, cfg.paths.learning_curves_png)


def main():
    cfg = parse_args_and_get_config()
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
