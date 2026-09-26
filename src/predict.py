import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import parse_args_and_get_config
from src.data import load_vocabularies, normalize_answer, tokenize_question
from src.extract_features import ResNet50FeatureExtractor
from src.model import build_model


class VQAPredictor:
    def __init__(self, cfg, model_path: Optional[str] = None, vocab_path: Optional[str] = None):
        self.cfg = cfg
        self.device = cfg.resolved_device

        model_path = model_path or cfg.paths.best_model_pth
        vocab_path = vocab_path or cfg.paths.vocab_json

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please run training or data.py first.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Please train the model first.")

        self.word2idx, self.idx2word, self.ans2idx, self.idx2ans = load_vocabularies(vocab_path)

        print("[Loading] Loading ResNet-50...")
        self.resnet = ResNet50FeatureExtractor().to(self.device)
        self.resnet.eval()

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"[Loading] Loading VQA model checkpoint from {model_path}...")
        self.vqa_model = build_model(
            cfg=cfg,
            vocab_size=len(self.word2idx),
            num_answers=len(self.ans2idx),
            model_type="vqa"
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.vqa_model.load_state_dict(checkpoint["model_state_dict"])
        self.vqa_model.eval()

    def predict(self, image_path: str, question: str) -> Tuple[str, float]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_image = Image.open(image_path).convert("RGB")
        img_tensor = self.img_transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feature = self.resnet(img_tensor)

        token_indices, actual_len = tokenize_question(question, self.word2idx, self.cfg.data.max_question_len)
        question_tensor = torch.tensor([token_indices], dtype=torch.long, device=self.device)
        lengths = torch.tensor([actual_len], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.vqa_model(img_feature, question_tensor, lengths=lengths)
            probs = F.softmax(logits, dim=-1)
            conf, pred_idx = probs.max(dim=-1)

        predicted_answer = normalize_answer(self.idx2ans[pred_idx.item()])
        confidence = float(conf.item())

        return predicted_answer, confidence


def predict(image_path: str, question: str, env: str = "local") -> Tuple[str, float]:
    from src.config import load_config
    cfg = load_config(env=env)
    predictor = VQAPredictor(cfg)
    return predictor.predict(image_path, question)


def main():
    parser = argparse.ArgumentParser(description="VQA Inference: Answer question about an image")
    parser.add_argument("--image", type=str, default=None, help="Path to image (jpg/png)")
    parser.add_argument("--question", type=str, default="is the object red?", help="Question text in English")
    parser.add_argument("--env", type=str, choices=["local", "kaggle"], default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (overrides --env)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    cfg = parse_args_and_get_config()

    image_path = args.image
    if image_path is None:
        val_imgs = list(Path(cfg.data.val_img_dir).glob("*.jpg"))
        if val_imgs:
            image_path = str(val_imgs[0])
            print(f"[Info] Argument --image not specified. Using sample image: {image_path}")
        else:
            print("[Warning] No images found in validation directory.")
            return

    predictor = VQAPredictor(cfg)
    answer, confidence = predictor.predict(image_path, args.question)

    print("\n" + "=" * 60)
    print("Prediction Results (VQA Predict)")
    print("=" * 60)
    print(f"Image:       {image_path}")
    print(f"Question:    {args.question}")
    print(f"Answer:      {answer}")
    print(f"Confidence:  {confidence:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
