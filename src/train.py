import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import parse_args_and_get_config
from src.data import get_dataloaders
from src.model import build_model


def compute_vqa_accuracy(predictions: torch.Tensor, all_answers: List[List[str]], idx2ans: List[str]) -> Tuple[float, List[float]]:
    """
    ?????? ??????????? ??????? VQA Accuracy:
    accuracy = min(????? ?????????? ? 10 ???????? ??????????? / 3.0, 1.0)
    ?????????? ??????? ???????? ?? ????? ? ?????? ?????? ??? ??????? ???????.
    """
    pred_indices = predictions.argmax(dim=-1).cpu().tolist()
    scores = []
    for pred_idx, human_answers in zip(pred_indices, all_answers):
        pred_ans_str = idx2ans[pred_idx]
        matches = sum(1 for h_ans in human_answers if h_ans.lower().strip() == pred_ans_str)
        score = min(matches / 3.0, 1.0)
        scores.append(score)
    mean_acc = sum(scores) / len(scores) if scores else 0.0
    return mean_acc, scores


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """???????? ?????? ?? ????? ?????."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="????????", leave=False):
        img_feat = batch["image_feature"].to(device)
        questions = batch["question"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(img_feat, questions)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    idx2ans: List[str],
    device: torch.device
) -> Tuple[float, float]:
    """????????? ??????: ?????? ??????? ?????? ? VQA Accuracy."""
    model.eval()
    total_loss = 0.0
    all_scores = []
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="?????????", leave=False):
            img_feat = batch["image_feature"].to(device)
            questions = batch["question"].to(device)
            targets = batch["target"].to(device)

            logits = model(img_feat, questions)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            num_batches += 1

            _, batch_scores = compute_vqa_accuracy(logits, batch["all_answers"], idx2ans)
            all_scores.extend(batch_scores)

    val_loss = total_loss / max(num_batches, 1)
    val_acc = (sum(all_scores) / len(all_scores)) * 100.0 if all_scores else 0.0
    return val_loss, val_acc


def run_training(cfg, model_type: str = "vqa"):
    """
    ?????? ???? ???????? ? Early Stopping ? ??????????? ?????? ??????????? ?????.
    """
    print(f"\n=======================================================")
    print(f"????? ???????? ??????: {model_type.upper()}")
    print(f"?????????:             {cfg.env}")
    print(f"??????????:            {cfg.resolved_device}")
    print(f"??????? (fusion):      {cfg.model.get('fusion_method', 'mul')}")
    print(f"????:                  {cfg.training.num_epochs}")
    print(f"?????? ?????:          {cfg.training.batch_size}")
    print(f"Learning rate:         {cfg.training.learning_rate}")
    print(f"=======================================================\n")

    # 1. ???????? ??????
    train_loader, val_loader, word2idx, idx2word, ans2idx, idx2ans = get_dataloaders(cfg)

    # 2. ????????????? ??????
    model = build_model(
        cfg=cfg,
        vocab_size=len(word2idx),
        num_answers=len(ans2idx),
        model_type=model_type
    ).to(cfg.resolved_device)

    # 3. ??????? ?????? ? ???????????
    # ??? VQA ?????????? ???????????? BCEWithLogitsLoss ? ??????? ???????
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # 4. ???? ?????????? ?????????
    save_pth = cfg.paths.best_question_only_pth if model_type == "question_only" else cfg.paths.best_model_pth

    # 5. ???? ????????
    best_val_acc = -1.0
    patience_counter = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, cfg.training.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.resolved_device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, idx2ans, cfg.resolved_device)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"????? [{epoch:02d}/{cfg.training.num_epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val VQA Acc: {val_acc:.2f}%")

        # ???????? Early Stopping ? ?????????? ?????
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "model_type": model_type,
                "cfg": dict(cfg)
            }, save_pth)
            print(f"  [?] ????? ?????? ????????: {val_acc:.2f}%. ???????? ???????? ? {save_pth}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"  [!] ???????? Early Stopping (????????={cfg.training.early_stopping_patience} ?????????). ?????????.")
                break

    # ????????? ??????? ??????
    history_path = cfg.paths.metrics_history_json.replace(".json", f"_{model_type}.json")
    with open(history_path, "w", encoding="utf-8") as hf:
        json.dump(history, hf, indent=2)

    print(f"\n[?] ???????? {model_type} ?????????! ?????? Val VQA Accuracy: {best_val_acc:.2f}%")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="VQA Training Script")
    parser.add_argument("--env", type=str, choices=["local", "kaggle"], default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-type", type=str, choices=["vqa", "question_only"], default="vqa",
                        help="??? ????????? ??????: ?????? VQA ??? Question-Only Baseline")
    args, _ = parser.parse_known_args()

    cfg = parse_args_and_get_config()
    run_training(cfg, model_type=args.model_type)


if __name__ == "__main__":
    main()
