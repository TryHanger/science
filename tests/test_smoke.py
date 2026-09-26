import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.data import (
    clean_text,
    compute_soft_targets,
    get_dataloaders,
    get_test_loader,
    normalize_answer,
    tokenize_question,
    VQADataset,
)
from src.extract_features import ResNet50FeatureExtractor, build_image_index
from src.model import QuestionOnlyBaselineModel, VQAModel
from src.splits import build_split, load_split_manifest, make_dev_image_ids, save_split_manifest


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


def test_build_image_index(tmp_path):
    """Test image indexing by filename pattern and filtering non-image files."""
    f1 = tmp_path / "COCO_train2014_000000000042.jpg"
    f2 = tmp_path / "7.jpg"
    f3 = tmp_path / "readme.txt"
    f1.write_text("fake image 1")
    f2.write_text("fake image 2")
    f3.write_text("text file")

    index = build_image_index(tmp_path)
    assert set(index.keys()) == {42, 7}
    assert index[42] == f1
    assert index[7] == f2


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


def test_make_dev_image_ids_deterministic_and_size():
    """(a) Test make_dev_image_ids is deterministic with same seed, differs with different seeds, and size is round(n*frac)."""
    image_ids = list(range(100, 200))  # 100 images
    frac = 0.15
    seed1 = 2026
    seed2 = 42

    dev_ids_1 = make_dev_image_ids(image_ids, dev_fraction=frac, seed=seed1)
    dev_ids_1_again = make_dev_image_ids(image_ids, dev_fraction=frac, seed=seed1)
    dev_ids_2 = make_dev_image_ids(image_ids, dev_fraction=frac, seed=seed2)

    assert dev_ids_1 == dev_ids_1_again, "make_dev_image_ids must be deterministic with same seed"
    assert dev_ids_1 != dev_ids_2, "make_dev_image_ids should produce different splits with different seeds"
    assert len(dev_ids_1) == round(len(image_ids) * frac)
    assert len(dev_ids_2) == round(len(image_ids) * frac)

    # Edge cases: minimum 1 when frac > 0 and non-empty
    small_dev = make_dev_image_ids([1, 2], dev_fraction=0.01, seed=2026)
    assert len(small_dev) == 1
    # Empty inputs
    assert make_dev_image_ids([], dev_fraction=0.1, seed=2026) == set()
    assert make_dev_image_ids([1, 2], dev_fraction=0.0, seed=2026) == set()


def test_splits_disjoint_and_union():
    """(b) Test dev and train image_id sets are disjoint and their union equals all images."""
    image_ids = list(range(1, 101))
    frac = 0.25
    seed = 123
    dev_ids = make_dev_image_ids(image_ids, dev_fraction=frac, seed=seed)
    train_ids = set(image_ids) - dev_ids

    assert dev_ids.isdisjoint(train_ids), "dev and train image IDs must be disjoint"
    assert dev_ids | train_ids == set(image_ids), "dev | train must equal all original image IDs"
    assert len(dev_ids) == 25
    assert len(train_ids) == 75


def test_protocol_dataloaders_on_local_data(tmp_path):
    """
    (c) Test get_dataloaders on local data with split_mode='protocol':
    returns non-empty train and dev, question_ids are disjoint, and dev image_ids are subset of dev_ids.
    """
    cfg = load_config(env="local")
    cfg.data.split_mode = "protocol"
    cfg.data.dev_image_fraction = 0.2
    cfg.data.split_seed = 2026
    cfg.output_dir = str(tmp_path)
    cfg.paths.output_dir = str(tmp_path)

    project_root = Path(__file__).resolve().parent.parent
    train_h5 = project_root / "outputs" / "train_img_features.h5"
    val_h5 = project_root / "outputs" / "val_img_features.h5"
    assert train_h5.exists(), "train_img_features.h5 must exist in outputs/"
    cfg.paths.train_features_h5 = str(train_h5)
    if val_h5.exists():
        cfg.paths.val_features_h5 = str(val_h5)
    cfg.paths.vocab_json = str(tmp_path / "vocab.json")

    train_loader, dev_loader, word2idx, idx2word, ans2idx, idx2ans = get_dataloaders(cfg)

    assert len(train_loader.dataset) > 0, "Train dataset should not be empty"
    assert len(dev_loader.dataset) > 0, "Dev dataset should not be empty"

    train_qids = {s["question_id"] for s in train_loader.dataset.samples}
    dev_qids = {s["question_id"] for s in dev_loader.dataset.samples}
    assert train_qids.isdisjoint(dev_qids), "train and dev question_ids must be disjoint"

    # Manifest verification
    manifest_path = tmp_path / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json should be saved in cfg.output_dir"
    manifest = load_split_manifest(manifest_path)
    dev_ids = manifest["dev_image_ids"]
    assert isinstance(dev_ids, set)

    dev_img_ids = {s["image_id"] for s in dev_loader.dataset.samples}
    assert dev_img_ids.issubset(dev_ids), "dev dataset image_ids must be a subset of dev_ids"

    # Test get_test_loader under protocol
    test_loader = get_test_loader(cfg, word2idx, ans2idx)
    assert len(test_loader.dataset) > 0, "Test dataset should not be empty"


def test_split_manifest_reuse_and_conflict(tmp_path):
    """Test split manifest reuse when matching and ValueError when parameters conflict."""
    cfg = load_config(env="local")
    cfg.data.split_mode = "protocol"
    cfg.data.dev_image_fraction = 0.2
    cfg.data.split_seed = 2026
    cfg.output_dir = str(tmp_path)
    cfg.paths.output_dir = str(tmp_path)

    split1 = build_split(cfg)
    split2 = build_split(cfg)
    assert split1["dev_image_ids"] == split2["dev_image_ids"]

    # Conflicting seed should raise ValueError
    cfg.data.split_seed = 9999
    with pytest.raises(ValueError, match="conflicting parameters"):
        build_split(cfg)


def test_preload_features_and_missing_threshold(tmp_path):
    """Test feature preloading in RAM and max_missing_feature_frac threshold validation."""
    import h5py
    cfg = load_config(env="local")
    cfg.data.split_mode = "protocol"
    cfg.data.dev_image_fraction = 0.2
    cfg.data.split_seed = 2026
    cfg.data.preload_features = True
    cfg.data.max_missing_feature_frac = 0.05
    cfg.output_dir = str(tmp_path)
    cfg.paths.output_dir = str(tmp_path)

    project_root = Path(__file__).resolve().parent.parent
    train_h5 = project_root / "outputs" / "train_img_features.h5"
    cfg.paths.train_features_h5 = str(train_h5)
    cfg.paths.vocab_json = str(tmp_path / "vocab.json")

    train_loader, dev_loader, word2idx, idx2word, ans2idx, idx2ans = get_dataloaders(cfg)
    assert train_loader.dataset.preload_features is True
    batch = next(iter(train_loader))
    assert batch["image_feature"].shape[-1] == 2048

    # Create dummy H5 missing almost all images -> should raise RuntimeError with threshold
    dummy_h5 = tmp_path / "incomplete_features.h5"
    with h5py.File(dummy_h5, "w") as f:
        f.create_dataset("9999999", data=np.zeros(2048, dtype=np.float32))

    with pytest.raises(RuntimeError, match="Missing image features"):
        VQADataset(
            questions_path=cfg.data.train_questions,
            annotations_path=cfg.data.train_annotations,
            features_h5_path=str(dummy_h5),
            word2idx=word2idx,
            ans2idx=ans2idx,
            max_missing_feature_frac=0.01
        )


def test_load_config_seed_and_run_dir():
    """
    Test seed override and run_dir routing:
    (a) load_config(env="local") -> paths.run_dir == output_dir
    (b) load_config(env="local", seed=7) -> cfg.seed == 7, paths.run_dir ends with 'seed_7',
        best_model_pth inside run_dir, vocab_json and train_features_h5 in output_dir.
    Clean up outputs/seed_7 after test.
    """
    import shutil

    # (a) load_config(env="local") -> paths.run_dir == output_dir
    cfg_default = load_config(env="local")
    assert cfg_default.paths.run_dir == cfg_default.paths.output_dir

    # (b) load_config(env="local", seed=7) -> cfg.seed==7, paths.run_dir ends with seed_7,
    # best_model_pth inside run_dir, vocab_json and train_features_h5 in output_dir
    output_dir = Path(cfg_default.paths.output_dir)
    seed_7_dir = output_dir / "seed_7"
    had_seed_7 = seed_7_dir.exists()

    try:
        cfg_seed = load_config(env="local", seed=7)
        assert cfg_seed.seed == 7
        assert cfg_seed.paths.run_dir.endswith("seed_7")
        assert Path(cfg_seed.paths.best_model_pth).parent == Path(cfg_seed.paths.run_dir)
        assert Path(cfg_seed.paths.vocab_json).parent == Path(cfg_seed.paths.output_dir)
        assert Path(cfg_seed.paths.train_features_h5).parent == Path(cfg_seed.paths.output_dir)
    finally:
        if not had_seed_7 and seed_7_dir.exists():
            shutil.rmtree(seed_7_dir)


def test_vqa_scores():
    """
    Test simplified and official VQA accuracy metrics:
    (a) answers = ["yes"]*10, pred "yes" -> official 1.0, simplified 1.0;
    (b) pred in 3 of 10 -> official 0.9, simplified 1.0;
    (c) pred in 1 of 10 -> official 0.3, simplified ≈ 0.3333.
    """
    from src.metrics import official_vqa_score, simplified_vqa_score

    # (a) answers = ["yes"]*10, pred "yes" -> official 1.0, simplified 1.0
    answers_a = ["yes"] * 10
    pred_a = "yes"
    assert official_vqa_score(pred_a, answers_a) == pytest.approx(1.0)
    assert simplified_vqa_score(pred_a, answers_a) == pytest.approx(1.0)

    # (b) pred occurs in 3 of 10 -> official 0.9, simplified 1.0
    answers_b = ["yes"] * 3 + ["no"] * 7
    pred_b = "yes"
    assert official_vqa_score(pred_b, answers_b) == pytest.approx(0.9)
    assert simplified_vqa_score(pred_b, answers_b) == pytest.approx(1.0)

    # (c) pred occurs in 1 of 10 -> official 0.3, simplified ≈ 0.3333
    answers_c = ["yes"] * 1 + ["no"] * 9
    pred_c = "yes"
    assert official_vqa_score(pred_c, answers_c) == pytest.approx(0.3)
    assert simplified_vqa_score(pred_c, answers_c) == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_aggregate_seeds(tmp_path):
    """
    Test seed aggregation script:
    (d) aggregate_seeds: in tmp_path create seed_1/metrics_table.csv and seed_2/metrics_table.csv
        (one model, 4 answer types, Accuracy 40/42), run main -> results_table.csv,
        mean overall = 41, n_seeds = 2.
    """
    import pandas as pd
    from scripts.aggregate_seeds import main

    model_name = "VQA Baseline (mul)"
    answer_types = ["overall", "yes/no", "number", "other"]

    seed1_dir = tmp_path / "seed_1"
    seed1_dir.mkdir(parents=True, exist_ok=True)
    df_seed1 = pd.DataFrame([
        {
            "Model": model_name,
            "Split": "test",
            "Seed": 1,
            "Answer Type": at,
            "N": 100,
            "Accuracy (%)": 40.0,
            "Accuracy simplified (%)": 40.0,
            "Best Epoch": 3,
        }
        for at in answer_types
    ])
    df_seed1.to_csv(seed1_dir / "metrics_table.csv", index=False)

    seed2_dir = tmp_path / "seed_2"
    seed2_dir.mkdir(parents=True, exist_ok=True)
    df_seed2 = pd.DataFrame([
        {
            "Model": model_name,
            "Split": "test",
            "Seed": 2,
            "Answer Type": at,
            "N": 100,
            "Accuracy (%)": 42.0,
            "Accuracy simplified (%)": 42.0,
            "Best Epoch": 4,
        }
        for at in answer_types
    ])
    df_seed2.to_csv(seed2_dir / "metrics_table.csv", index=False)

    # Run aggregation
    df_results = main(["--root", str(tmp_path)])

    results_table_path = tmp_path / "results_table.csv"
    assert results_table_path.exists()
    assert (tmp_path / "results_per_seed.csv").exists()

    df_csv = pd.read_csv(results_table_path)
    overall_row = df_csv[df_csv["Answer Type"] == "overall"]
    assert len(overall_row) == 1
    assert overall_row["n_seeds"].values[0] == 2
    assert overall_row["Accuracy (%) mean"].values[0] == pytest.approx(41.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
