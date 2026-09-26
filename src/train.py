import argparse
import json
import os
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
from src.data import get_dataloaders, normalize_answer
from src.model import build_model


def compute_vqa_accuracy(predictions: torch.Tensor, all_answers: List[List[str]], idx2ans: List[str]) -> Tuple[float, List[float]]:
    """
    Compute official VQA accuracy metric:
    accuracy = min(matching human answers count / 3.0, 1.0)
    """
    pred_indices = predictions.argmax(dim=-1).cpu().tolist()
    scores = []
    for pred_idx, human_answers in zip(pred_indices, all_answers):
        pred_ans_str = normalize_answer(idx2ans[pred_idx])
        normalized_human = [normalize_answer(h_ans) for h_ans in human_answers]
        matches = sum(1 for h_ans in normalized_human if h_ans == pred_ans_str)
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
    """Train the model for one epoch using sequence lengths and pack_padded_sequence."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        img_feat = batch["image_feature"].to(device)
        questions = batch["question"].to(device)
        lengths = batch["length"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(img_feat, questions, lengths=lengths)
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
    """Evaluate the model: average loss and VQA Accuracy."""
    model.eval()
    total_loss = 0.0
    all_scores = []
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            img_feat = batch["image_feature"].to(device)
            questions = batch["question"].to(device)
            lengths = batch["length"].to(device)
            targets = batch["target"].to(device)

            logits = model(img_feat, questions, lengths=lengths)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            num_batches += 1

            _, batch_scores = compute_vqa_accuracy(logits, batch["all_answers"], idx2ans)
            all_scores.extend(batch_scores)

    val_loss = total_loss / max(num_batches, 1)
    val_acc = (sum(all_scores) / len(all_scores)) * 100.0 if all_scores else 0.0
    return val_loss, val_acc


def get_artifact_paths(cfg, model_type: str = "vqa") -> Tuple[str, str]:
    if hasattr(cfg, "paths") and "run_dir" in cfg.paths and cfg.paths.run_dir:
        run_dir = Path(cfg.paths.run_dir)
    elif hasattr(cfg, "paths") and "output_dir" in cfg.paths:
        run_dir = Path(cfg.paths.output_dir)
    else:
        run_dir = Path(cfg.output_dir)

    model_type = model_type.lower()
    if model_type == "question_only":
        ckpt_path = str(cfg.paths.best_question_only_pth) if (hasattr(cfg, "paths") and "best_question_only_pth" in cfg.paths) else str(run_dir / "best_question_only_model.pth")
        history_path = str(run_dir / "metrics_history_question_only.json")
    else:
        fusion = cfg.model.get("fusion_method", "mul").lower()
        if fusion == "mul":
            ckpt_path = str(cfg.paths.best_model_pth) if (hasattr(cfg, "paths") and "best_model_pth" in cfg.paths) else str(run_dir / "best_model.pth")
            history_path = str(run_dir / "metrics_history_vqa.json")
        else:
            ckpt_path = str(run_dir / f"best_model_{fusion}.pth")
            history_path = str(run_dir / f"metrics_history_vqa_{fusion}.json")
    return ckpt_path, history_path


def run_training(cfg, model_type: str = "vqa", resume: bool = False):
    resume = resume or cfg.training.get("resume", False)
    if "lr_scheduler" not in cfg.training:
        cfg.training.lr_scheduler = "none"

    run_dir_str = str(cfg.paths.run_dir) if (hasattr(cfg, "paths") and "run_dir" in cfg.paths and cfg.paths.run_dir) else str(cfg.paths.output_dir if (hasattr(cfg, "paths") and "output_dir" in cfg.paths) else cfg.output_dir)
    seed_str = str(cfg.seed) if hasattr(cfg, "seed") else str(cfg.get("seed", 42))

    print(f"\n=======================================================")
    print(f"Model type:            {model_type.upper()}")
    print(f"Environment:           {cfg.env}")
    print(f"Device:                {cfg.resolved_device}")
    print(f"Seed:                  {seed_str}")
    print(f"Run dir:               {run_dir_str}")
    print(f"Fusion method:         {cfg.model.get('fusion_method', 'mul')}")
    print(f"Epochs:                {cfg.training.num_epochs}")
    print(f"Batch size:            {cfg.training.batch_size}")
    print(f"Learning rate:         {cfg.training.learning_rate}")
    if cfg.training.get("lr_scheduler", "none") != "none":
        print(f"LR scheduler:          {cfg.training.lr_scheduler}")
    print(f"=======================================================\n")

    train_loader, val_loader, word2idx, idx2word, ans2idx, idx2ans = get_dataloaders(cfg)

    model = build_model(
        cfg=cfg,
        vocab_size=len(word2idx),
        num_answers=len(ans2idx),
        model_type=model_type
    ).to(cfg.resolved_device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    lr_scheduler_type = cfg.training.get("lr_scheduler", "none")
    if lr_scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.training.get("scheduler_factor", 0.5),
            patience=cfg.training.get("scheduler_patience", 2)
        )
    else:
        scheduler = None

    ckpt_path, history_path = get_artifact_paths(cfg, model_type)

    start_epoch = 1
    best_val_acc = -1.0
    patience_counter = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    if resume:
        if os.path.exists(ckpt_path):
            print(f"[*] Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=cfg.resolved_device, weights_only=False)

            ckpt_model_type = ckpt.get("model_type")
            if ckpt_model_type != model_type:
                raise ValueError(
                    f"Checkpoint model_type '{ckpt_model_type}' does not match current model_type '{model_type}'."
                )

            ckpt_fusion = None
            if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
                model_cfg = ckpt["cfg"].get("model", {})
                if isinstance(model_cfg, dict) and "fusion_method" in model_cfg:
                    ckpt_fusion = model_cfg["fusion_method"]
            if ckpt_fusion is None and "fusion_method" in ckpt:
                ckpt_fusion = ckpt.get("fusion_method")

            current_fusion = cfg.model.get("fusion_method", "mul")
            if ckpt_fusion is not None and ckpt_fusion != current_fusion:
                raise ValueError(
                    f"Checkpoint fusion method '{ckpt_fusion}' does not match current fusion method '{current_fusion}'."
                )

            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            start_epoch = ckpt["epoch"] + 1
            best_val_acc = ckpt["val_acc"]
            print(f"[*] Resumed checkpoint from epoch {ckpt['epoch']} with best_val_acc: {best_val_acc:.2f}%. Next epoch: {start_epoch}")

            if os.path.exists(history_path):
                with open(history_path, "r", encoding="utf-8") as hf:
                    history = json.load(hf)

                ref_len = len(history.get("epochs", []))
                if "lr" not in history or not isinstance(history["lr"], list):
                    history["lr"] = [None] * ref_len
                elif len(history["lr"]) < ref_len:
                    history["lr"].extend([None] * (ref_len - len(history["lr"])))

                ckpt_epoch = ckpt["epoch"]
                for k, v in list(history.items()):
                    if isinstance(v, list):
                        history[k] = v[:ckpt_epoch]

                for key in ["epochs", "train_loss", "val_loss", "val_acc", "lr"]:
                    if key not in history:
                        history[key] = []
        else:
            print(f"[!] Warning: Checkpoint not found at {ckpt_path}. Training from scratch.")

    if start_epoch > cfg.training.num_epochs:
        print(f"[*] Training already finished: start_epoch ({start_epoch}) > num_epochs ({cfg.training.num_epochs}). Exiting without training.")
        return model, history

    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.resolved_device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, idx2ans, cfg.resolved_device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch:02d}/{cfg.training.num_epochs:02d}] | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val VQA Acc: {val_acc:.2f}%")

        if scheduler is not None:
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "val_acc": val_acc,
                "model_type": model_type,
                "fusion_method": cfg.model.get("fusion_method", "mul"),
                "cfg": dict(cfg)
            }, ckpt_path)
            print(f"  [*] New best accuracy: {val_acc:.2f}%. Checkpoint saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"  [!] Early stopping triggered (patience={cfg.training.early_stopping_patience} epochs). Exiting training.")
                with open(history_path, "w", encoding="utf-8") as hf:
                    json.dump(history, hf, indent=2)
                break

        with open(history_path, "w", encoding="utf-8") as hf:
            json.dump(history, hf, indent=2)

    with open(history_path, "w", encoding="utf-8") as hf:
        json.dump(history, hf, indent=2)

    print(f"\n[*] Training {model_type} completed! Best Val VQA Accuracy: {best_val_acc:.2f}%")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="VQA Training Script")
    parser.add_argument("--env", type=str, choices=["local", "kaggle"], default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--model-type", type=str, choices=["vqa", "question_only"], default="vqa")
    parser.add_argument("--fusion", type=str, choices=["mul", "concat"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau"], default=None)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = parse_args_and_get_config()

    if args.fusion is not None:
        cfg.model.fusion_method = args.fusion
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.patience is not None:
        cfg.training.early_stopping_patience = args.patience
    if args.scheduler is not None:
        cfg.training.lr_scheduler = args.scheduler
    elif "lr_scheduler" not in cfg.training:
        cfg.training.lr_scheduler = "none"

    run_training(cfg, model_type=args.model_type, resume=args.resume)


if __name__ == "__main__":
    main()
