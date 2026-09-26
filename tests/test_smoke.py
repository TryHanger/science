import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.data import clean_text, compute_soft_targets, normalize_answer, tokenize_question
from src.extract_features import ResNet50FeatureExtractor
from src.model import QuestionOnlyBaselineModel, VQAModel


def test_normalize_answer():
    """Test standard answer normalization conforming to VQA Eval."""
    # 1. Articles and punctuation
    assert normalize_answer("a blue dog!") == "blue dog"
    assert normalize_answer("the car, yes?") == "car yes"
    assert normalize_answer("an apple.") == "apple"
    
    # 2. Number words to digits
    assert normalize_answer("two") == "2"
    assert normalize_answer("there are three cats") == "there are 3 cats"
    assert normalize_answer("zero") == "0"
    
    # 3. Contractions
    assert normalize_answer("don't") == "dont" or normalize_answer("don't") == "don't" or normalize_answer("dont") in ["dont", "don't"]


def test_clean_and_tokenize_with_lengths():
    """Test question cleaning, tokenization, and sequence length."""
    text = "Is there a blue dog?"
    tokens = clean_text(text)
    assert "is" in tokens
    assert "?" not in tokens

    word2idx = {"<pad>": 0, "<unk>": 1, "is": 2, "dog": 3}
    indices, actual_length = tokenize_question(text, word2idx, max_len=6)
    assert len(indices) == 6
    assert actual_length == 5  # 5 tokens in 'is there a blue dog'
    assert indices[0] == 2     # 'is'
    assert indices[-1] == 0    # <pad> at index 5 (6th position)


def test_pack_padded_sequence_invariance_to_pad():
    """
    Verify that pack_padded_sequence ignores trailing <pad> tokens
    and padding length does not affect the final LSTM hidden state.
    """
    vocab_size = 50
    embed_dim = 64
    hidden_dim = 128
    model = VQAModel(vocab_size=vocab_size, num_answers=10, text_embed_dim=embed_dim,
                     lstm_hidden_dim=hidden_dim, common_proj_dim=64)
    model.eval()

    # Question with length 3
    q1 = torch.tensor([[5, 12, 8, 0, 0, 0]], dtype=torch.long)
    len1 = torch.tensor([3], dtype=torch.long)

    # Same question, but with extra <pad> tokens
    q2 = torch.tensor([[5, 12, 8, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    len2 = torch.tensor([3], dtype=torch.long)

    img = torch.randn(1, 2048)

    with torch.no_grad():
        out1 = model(img, q1, lengths=len1)
        out2 = model(img, q2, lengths=len2)

    # Outputs must be identical within float32 precision
    assert torch.allclose(out1, out2, atol=1e-5), "Padding tokens affect the resulting representation!"


def test_soft_targets_with_normalization():
    """Test soft target computation with answer normalization."""
    answers = [
        {"answer": "two"},
        {"answer": "2"},
        {"answer": "two"},
        {"answer": "no"},
    ]
    ans2idx = {"2": 0, "no": 1, "blue": 2}
    targets = compute_soft_targets(answers, ans2idx, num_answers=3)
    # 'two' normalizes to '2' -> 3 occurrences of '2' -> 3/3 = 1.0
    assert targets[0].item() == 1.0
    assert abs(targets[1].item() - 1.0 / 3.0) < 1e-4
    assert targets[2].item() == 0.0


def test_resnet50_feature_extractor():
    extractor = ResNet50FeatureExtractor()
    extractor.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = extractor(dummy_input)
    assert out.shape == (2, 2048)


def test_vqa_models_forward_and_backward_with_lengths():
    batch_size = 4
    vocab_size = 50
    num_answers = 20

    dummy_img = torch.randn(batch_size, 2048)
    dummy_q = torch.randint(1, vocab_size, (batch_size, 14))
    dummy_lengths = torch.tensor([14, 10, 8, 5], dtype=torch.long)
    dummy_target = torch.rand(batch_size, num_answers)

    criterion = nn.BCEWithLogitsLoss()

    # 1. Mul model
    model_mul = VQAModel(vocab_size, num_answers, fusion_method="mul")
    logits_mul = model_mul(dummy_img, dummy_q, lengths=dummy_lengths)
    assert logits_mul.shape == (batch_size, num_answers)
    loss_mul = criterion(logits_mul, dummy_target)
    loss_mul.backward()
    assert model_mul.img_proj[0].weight.grad is not None

    # 2. Concat model
    model_cat = VQAModel(vocab_size, num_answers, fusion_method="concat")
    logits_cat = model_cat(dummy_img, dummy_q, lengths=dummy_lengths)
    assert logits_cat.shape == (batch_size, num_answers)
    loss_cat = criterion(logits_cat, dummy_target)
    loss_cat.backward()
    assert model_cat.img_proj[0].weight.grad is not None

    # 3. Question-only model
    model_q = QuestionOnlyBaselineModel(vocab_size, num_answers)
    logits_q = model_q(None, dummy_q, lengths=dummy_lengths)
    assert logits_q.shape == (batch_size, num_answers)
    loss_q = criterion(logits_q, dummy_target)
    loss_q.backward()
    assert model_q.text_proj[0].weight.grad is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
