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
    ???????????? ??????? ??????????? (Baseline) ??? Visual Question Answering:
    - ?????????? ?????: ResNet-50 features [2048] -> Linear -> [common_proj_dim] -> Tanh
    - ????????? ?????:  Word Tokens -> Embedding -> LSTM -> Linear -> [common_proj_dim] -> Tanh
    - ??????? (Fusion):
        * 'mul'    ? ???????????? ???????????? ??????? (Element-wise multiplication)
        * 'concat' ? ???????????? ????????
    - ?????????????:   Dropout -> Linear -> ReLU -> Linear -> Logits [num_answers]
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
            raise ValueError(f"??????????? ????? ???????: {fusion_method}. ????????? 'mul' ??? 'concat'.")

        # 1. ?????????? ????????
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, common_proj_dim),
            nn.Tanh()
        )

        # 2. ????????? ????????? (Embedding + LSTM + Projection)
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

        # 3. ??????????? ????? ???????
        fused_dim = common_proj_dim if self.fusion_method == "mul" else common_proj_dim * 2

        # 4. ?????????????
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, common_proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(common_proj_dim, num_answers)
        )

    def forward(self, img_feat: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        # img_feat: [batch_size, 2048]
        # question_tokens: [batch_size, max_len]

        # 1. ????????? ???????????
        v = self.img_proj(img_feat)  # [batch_size, common_proj_dim]

        # 2. ????????? ???????
        embedded = self.embedding(question_tokens)  # [batch_size, max_len, text_embed_dim]
        _, (h_n, _) = self.lstm(embedded)           # h_n: [1, batch_size, lstm_hidden_dim]
        last_hidden = h_n[-1]                        # [batch_size, lstm_hidden_dim]
        q = self.text_proj(last_hidden)              # [batch_size, common_proj_dim]

        # 3. ??????? ??????????????? ?????????
        if self.fusion_method == "mul":
            fused = v * q                            # [batch_size, common_proj_dim]
        else:
            fused = torch.cat([v, q], dim=-1)        # [batch_size, common_proj_dim * 2]

        # 4. ????????????? -> ?????? ???????
        logits = self.classifier(fused)              # [batch_size, num_answers]
        return logits


class QuestionOnlyBaselineModel(nn.Module):
    """
    ??????????? ?????? (Question-Only Baseline):
    ????????? ?????? ?????? ? ?????????????? ????? ??? ????????????? ????????.
    ? ??????? ???????????? ?? IMRAD ??? ?????? ?????????? ??? ????????
    ????? ???????? (language bias) ? ???????????? ????????? ?????? ?????????? ?????????.
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

    def forward(self, img_feat: Optional[torch.Tensor], question_tokens: torch.Tensor) -> torch.Tensor:
        # img_feat ????????????
        embedded = self.embedding(question_tokens)
        _, (h_n, _) = self.lstm(embedded)
        last_hidden = h_n[-1]
        q = self.text_proj(last_hidden)
        logits = self.classifier(q)
        return logits


def build_model(cfg, vocab_size: int, num_answers: int, model_type: str = "vqa") -> nn.Module:
    """
    ??????? ??????? ?? ?????? ????????????????? ?????.
    - model_type: 'vqa' ??? 'question_only'
    """
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


if __name__ == "__main__":
    # ???? ??????? ??????? (forward pass)
    batch_sz = 4
    vocab_sz = 100
    num_ans = 50
    dummy_img = torch.randn(batch_sz, 2048)
    dummy_q = torch.randint(0, vocab_sz, (batch_sz, 14))

    # 1. Mul model
    model_mul = VQAModel(vocab_sz, num_ans, fusion_method="mul")
    out_mul = model_mul(dummy_img, dummy_q)
    print("Mul output shape:", out_mul.shape)
    assert out_mul.shape == (batch_sz, num_ans)

    # 2. Concat model
    model_cat = VQAModel(vocab_sz, num_ans, fusion_method="concat")
    out_cat = model_cat(dummy_img, dummy_q)
    print("Concat output shape:", out_cat.shape)
    assert out_cat.shape == (batch_sz, num_ans)

    # 3. Question-only model
    model_q = QuestionOnlyBaselineModel(vocab_sz, num_ans)
    out_q = model_q(None, dummy_q)
    print("Question-only output shape:", out_q.shape)
    assert out_q.shape == (batch_sz, num_ans)

    print("??? ????? ??????? ??????? ????????!")
