import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


class VQAModel(nn.Module):
    """
    Classic baseline architecture for Visual Question Answering:
    - Visual branch: ResNet-50 features [2048] -> Linear -> [common_proj_dim] -> Tanh
    - Text branch:   Word Tokens -> Embedding -> LSTM (pack_padded_sequence) -> Linear -> [common_proj_dim] -> Tanh
    - Fusion:
        * 'mul'    - Element-wise multiplication
        * 'concat' - Concatenation
    - Classifier:    Dropout(0.5) -> Linear -> ReLU -> Dropout(0.5) -> Linear -> Logits [num_answers]
    """
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        img_feature_dim: int = 2048,
        text_embed_dim: int = 300,
        lstm_hidden_dim: int = 512,
        common_proj_dim: int = 1024,
        dropout_rate: float = 0.5,
        fusion_method: str = "mul"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_answers = num_answers
        self.fusion_method = fusion_method.lower()

        if self.fusion_method not in ["mul", "concat"]:
            raise ValueError(f"Unknown fusion method: {fusion_method}. Expected 'mul' or 'concat'.")

        # 1. Visual projection
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, common_proj_dim),
            nn.Tanh()
        )

        # 2. Text processing (Embedding + LSTM + Projection)
        self.embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )
        self.text_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, common_proj_dim),
            nn.Tanh()
        )

        # 3. Dimension after fusion
        fused_dim = common_proj_dim if self.fusion_method == "mul" else common_proj_dim * 2

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, common_proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(common_proj_dim, num_answers)
        )

    def forward(
        self,
        img_feat: torch.Tensor,
        question_tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # img_feat: [batch_size, 2048]
        # question_tokens: [batch_size, max_len]

        # 1. Project image features
        v = self.img_proj(img_feat)  # [batch_size, common_proj_dim]

        # 2. Process question with pack_padded_sequence (skipping <pad> tokens)
        embedded = self.embedding(question_tokens)  # [batch_size, max_len, text_embed_dim]

        if lengths is None:
            lengths = (question_tokens != 0).sum(dim=-1).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]                        # [batch_size, lstm_hidden_dim]
        q = self.text_proj(last_hidden)              # [batch_size, common_proj_dim]

        # 3. Fuse multimodal features
        if self.fusion_method == "mul":
            fused = v * q                            # [batch_size, common_proj_dim]
        else:
            fused = torch.cat([v, q], dim=-1)        # [batch_size, common_proj_dim * 2]

        # 4. Classifier -> answer logits
        logits = self.classifier(fused)              # [batch_size, num_answers]
        return logits


class QuestionOnlyBaselineModel(nn.Module):
    """
    Ablation model (Question-Only Baseline):
    Uses only the question and ignores the image to measure unimodal bias.
    """
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        text_embed_dim: int = 300,
        lstm_hidden_dim: int = 512,
        common_proj_dim: int = 1024,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_answers = num_answers

        self.embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )
        self.text_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, common_proj_dim),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(common_proj_dim, common_proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(common_proj_dim, num_answers)
        )

    def forward(
        self,
        img_feat: Optional[torch.Tensor],
        question_tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embedded = self.embedding(question_tokens)

        if lengths is None:
            lengths = (question_tokens != 0).sum(dim=-1).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        q = self.text_proj(last_hidden)
        logits = self.classifier(q)
        return logits


def build_model(cfg, vocab_size: int, num_answers: int, model_type: str = "vqa") -> nn.Module:
    if model_type == "question_only":
        return QuestionOnlyBaselineModel(
            vocab_size=vocab_size,
            num_answers=num_answers,
            text_embed_dim=cfg.model.text_embed_dim,
            lstm_hidden_dim=cfg.model.lstm_hidden_dim,
            common_proj_dim=cfg.model.common_proj_dim,
            dropout_rate=cfg.model.dropout_rate
        )

    return VQAModel(
        vocab_size=vocab_size,
        num_answers=num_answers,
        img_feature_dim=cfg.model.img_feature_dim,
        text_embed_dim=cfg.model.text_embed_dim,
        lstm_hidden_dim=cfg.model.lstm_hidden_dim,
        common_proj_dim=cfg.model.common_proj_dim,
        dropout_rate=cfg.model.dropout_rate,
        fusion_method=cfg.model.get("fusion_method", "mul")
    )
