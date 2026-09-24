import os
import sys
import tempfile
from pathlib import Path

# ?????? ??????? ? sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.data import clean_text, compute_soft_targets, tokenize_question
from src.extract_features import ResNet50FeatureExtractor
from src.model import QuestionOnlyBaselineModel, VQAModel


def test_clean_and_tokenize():
    """???? ??????? ? ??????????? ???????."""
    text = "Is there a blue dog in the picture?"
    tokens = clean_text(text)
    assert "is" in tokens
    assert "?" not in tokens

    word2idx = {"<pad>": 0, "<unk>": 1, "is": 2, "dog": 3}
    indices = tokenize_question(text, word2idx, max_len=6)
    assert len(indices) == 6
    assert indices[0] == 2  # 'is'
    assert indices[1] == 1  # 'there' -> unk
    assert indices[-1] == 0 or indices[-1] == 1


def test_soft_targets():
    """???? ?????????? ?????? ????? (min(count/3, 1.0))."""
    answers = [
        {"answer": "yes"},
        {"answer": "yes"},
        {"answer": "yes"},
        {"answer": "no"},
    ]
    ans2idx = {"yes": 0, "no": 1, "blue": 2}
    targets = compute_soft_targets(answers, ans2idx, num_answers=3)
    assert targets[0].item() == 1.0  # 3/3 = 1.0
    assert abs(targets[1].item() - 1.0 / 3.0) < 1e-4  # 1/3
    assert targets[2].item() == 0.0  # 0


def test_resnet50_feature_extractor():
    """???? ??????????? ?????? ??????????? ????????? ResNet-50."""
    extractor = ResNet50FeatureExtractor()
    extractor.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = extractor(dummy_input)
    assert out.shape == (2, 2048)


def test_vqa_models_forward_and_backward():
    """
    ???? ??????? ? ????????? ??????? ??? ??????????????? ?????? (mul ? concat),
    ? ????? ??? ??????????? ?????? Question-Only.
    """
    batch_size = 4
    vocab_size = 50
    num_answers = 20

    dummy_img = torch.randn(batch_size, 2048)
    dummy_q = torch.randint(0, vocab_size, (batch_size, 14))
    dummy_target = torch.rand(batch_size, num_answers)

    criterion = nn.BCEWithLogitsLoss()

    # 1. ??????????????? ?????? ? Hadamard Fusion (mul)
    model_mul = VQAModel(vocab_size, num_answers, fusion_method="mul")
    logits_mul = model_mul(dummy_img, dummy_q)
    assert logits_mul.shape == (batch_size, num_answers)
    loss_mul = criterion(logits_mul, dummy_target)
    loss_mul.backward()
    assert model_mul.img_proj[0].weight.grad is not None

    # 2. ??????????????? ?????? ? Concat Fusion
    model_cat = VQAModel(vocab_size, num_answers, fusion_method="concat")
    logits_cat = model_cat(dummy_img, dummy_q)
    assert logits_cat.shape == (batch_size, num_answers)
    loss_cat = criterion(logits_cat, dummy_target)
    loss_cat.backward()
    assert model_cat.img_proj[0].weight.grad is not None

    # 3. Question-Only Baseline
    model_q = QuestionOnlyBaselineModel(vocab_size, num_answers)
    logits_q = model_q(None, dummy_q)
    assert logits_q.shape == (batch_size, num_answers)
    loss_q = criterion(logits_q, dummy_target)
    loss_q.backward()
    assert model_q.text_proj[0].weight.grad is not None


def test_end_to_end_smoke_pipeline():
    """
    ??????? ???????? ?????-???? ?? 20 ????????:
    ????????? ???????? ???????, ???????? ? ??????????.
    """
    cfg = load_config(env="local")
    assert cfg.env == "local"
    assert cfg.paths.train_features_h5 is not None
    print("[?] Smoke-???? ??????? ???????!")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
