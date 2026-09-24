import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from src.config import parse_args_and_get_config


def find_image_path(img_dir: Path, split_name: str, image_id: int) -> Path:
    """
    ????? ???? ? ??????????? COCO ?? image_id.
    ???? ???? ?? ??????, ?????? ? ???????? ???????.
    """
    candidates = [
        img_dir / f"COCO_{split_name}_{image_id:012d}.jpg",
        img_dir / f"COCO_{split_name}_{image_id:012d}.jpeg",
        img_dir / f"COCO_{split_name}_{image_id}.jpg",
        img_dir / f"{image_id:012d}.jpg",
        img_dir / f"{image_id}.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    id_str = str(image_id)
    for p in img_dir.glob(f"*{id_str}*.jpg"):
        return p

    raise FileNotFoundError(
        f"\n[?????? ??????] ??????????? ??? image_id={image_id} ?? ??????? ? ?????: {img_dir}\n"
        f"?????????, ??? ??????? ???????? MS COCO 2014 ????????? ? ???? ?????? ?????."
    )


class ImageExtractDataset(Dataset):
    def __init__(self, image_ids: List[int], img_dir: Path, split_name: str, transform=None):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.split_name = split_name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = find_image_path(self.img_dir, self.split_name, img_id)
        # ??????????????? ?????????? ? RGB
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return img_id, image


class ResNet50FeatureExtractor(nn.Module):
    """
    ?????????? ????????? ResNet-50.
    ??????? ????????? fc-????, ???????? avgpool (?????? ??????????? 2048).
    """
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        for param in resnet.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)  # [B, 2048, 1, 1]
        feats = feats.flatten(start_dim=1)  # [B, 2048]
        return feats


def extract_features_for_split(
    questions_json_path: str,
    img_dir_path: str,
    output_h5_path: str,
    split_name: str,
    max_questions: int,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0
):
    print(f"\n=======================================================")
    print(f"?????????? ????????? ??? ??????: {split_name.upper()}")
    print(f"???? ????????: {questions_json_path}")
    print(f"????? ????????: {img_dir_path}")
    print(f"???????? H5:    {output_h5_path}")
    print(f"??????????:     {device}")
    print(f"=======================================================")

    # ???????? ??????? ?????? (?? Kaggle ??? ????????)
    if not os.path.exists(questions_json_path):
        raise FileNotFoundError(
            f"\n[?????? ??????] ???? ???????? ?? ??????: {questions_json_path}.\n"
            f"??? Kaggle: ?????????? ??????? VQA v2 ? ???? 'Add Input' ? ????????? ???? ? configs/kaggle.yaml.\n"
            f"??? ?????????? ?????: ????????? `python scripts/download_sample.py`."
        )

    if not os.path.exists(img_dir_path):
        raise FileNotFoundError(
            f"\n[?????? ??????] ????? ? ?????????? ?? ???????: {img_dir_path}.\n"
            f"??? Kaggle: ?????????? ??????? MS COCO 2014 ? ???? 'Add Input' ? ????????? ???? ? configs/kaggle.yaml."
        )

    with open(questions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    if max_questions is not None and max_questions > 0:
        questions = questions[:max_questions]

    unique_image_ids = sorted(list(set(q["image_id"] for q in questions)))
    print(f"[1/3] ???????? {len(questions):,} ????????, ?????????? ???????????: {len(unique_image_ids):,}")

    Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if os.path.exists(output_h5_path):
        with h5py.File(output_h5_path, "r") as h5_check:
            existing_ids = set(int(k) for k in h5_check.keys())
        print(f"[2/3] ????????? ???????????? ???? H5. ??? ????????? ????????: {len(existing_ids):,}")

    pending_ids = [img_id for img_id in unique_image_ids if img_id not in existing_ids]
    if not pending_ids:
        print(f"[?] ??? {len(unique_image_ids):,} ??????????? ??? ??????????. ???????.")
        return

    print(f"[3/3] ???????? ??????? ???????? ???: {len(pending_ids):,} ???????????.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageExtractDataset(
        image_ids=pending_ids,
        img_dir=Path(img_dir_path),
        split_name=split_name,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    extractor = ResNet50FeatureExtractor().to(device)
    extractor.eval()  # model.eval()

    with h5py.File(output_h5_path, "a") as h5_out:
        with torch.no_grad():  # torch.no_grad()
            for batch_ids, batch_images in tqdm(dataloader, desc=f"?????????? {split_name}"):
                batch_images = batch_images.to(device)
                features = extractor(batch_images)  # [B, 2048]
                features_cpu = features.cpu().numpy()

                for img_id_tensor, feat_vec in zip(batch_ids, features_cpu):
                    img_id_str = str(int(img_id_tensor))
                    if img_id_str in h5_out:
                        del h5_out[img_id_str]
                    h5_out.create_dataset(img_id_str, data=feat_vec, compression="gzip")

    print(f"[?] ??????? ????????? ?????????? {split_name}. ????? ? H5: {len(existing_ids) + len(pending_ids):,}")


def main():
    cfg = parse_args_and_get_config()

    extract_features_for_split(
        questions_json_path=cfg.data.train_questions,
        img_dir_path=cfg.data.train_img_dir,
        output_h5_path=cfg.paths.train_features_h5,
        split_name="train2014",
        max_questions=cfg.data.num_train_questions,
        device=cfg.resolved_device,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    extract_features_for_split(
        questions_json_path=cfg.data.val_questions,
        img_dir_path=cfg.data.val_img_dir,
        output_h5_path=cfg.paths.val_features_h5,
        split_name="val2014",
        max_questions=cfg.data.num_val_questions,
        device=cfg.resolved_device,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )


if __name__ == "__main__":
    main()
