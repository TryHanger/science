import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
UNK_INDEX = 1

# Manual mapping conforming to the standard VQA Evaluation Protocol
MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90"
}

ARTICLES = {"a", "an", "the"}

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "i'dve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebodys": "somebody's", "someone's": "someone's",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "youll": "you'll", "youre": "you're", "youve": "you've"
}

PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
PUNCT = [
    ';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
    '_', '-', '>', '<', '@', '`', ',', '?', '!'
]


def normalize_answer(ans_str: str) -> str:
    """
    Standard answer normalization following the VQA Evaluation Protocol:
    1. Convert to lowercase.
    2. Remove or replace punctuation with spaces (preserving decimal numbers).
    3. Expand contractions.
    4. Remove articles (a, an, the).
    5. Convert number words (zero..twenty) to digits ('0'..'20').
    6. Strip extra whitespace.
    """
    if not isinstance(ans_str, str):
        ans_str = str(ans_str)
    s = ans_str.lower().strip()

    # Punctuation
    for p in PUNCT:
        if (p + " " in s or " " + p in s) or (re.search(r"[a-z]" + re.escape(p) + r"[a-z]", s) is not None):
            s = s.replace(p, "")
        else:
            s = s.replace(p, " ")

    # Periods (except decimal numbers) and apostrophes
    s = PERIOD_STRIP.sub("", s)
    s = s.replace("'", "")

    # Articles, numbers, and contractions
    tokens = s.split()
    cleaned_tokens = []
    for token in tokens:
        if token in ARTICLES:
            continue
        token = MANUAL_MAP.get(token, token)
        token = CONTRACTIONS.get(token, token).replace("'", "")
        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens).strip()


def clean_text(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[?!.,:;\"'()\[\]{}]", "", text)
    tokens = text.split()
    return tokens


def tokenize_question(question_text: str, word2idx: Dict[str, int], max_len: int = 14) -> Tuple[List[int], int]:
    tokens = clean_text(question_text)
    actual_length = max(1, min(len(tokens), max_len))
    indices = [word2idx.get(token, UNK_INDEX) for token in tokens]

    if len(indices) < max_len:
        indices = indices + [PAD_INDEX] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices, actual_length


def build_vocabularies(
    train_questions_path: str,
    train_annotations_path: str,
    num_train_questions: Optional[int] = None,
    num_answers: int = 1000,
    min_word_freq: int = 2,
    vocab_save_path: Optional[str] = None,
    exclude_image_ids: Optional[Set[int]] = None
) -> Tuple[Dict[str, int], List[str], Dict[str, int], List[str]]:
    if not Path(train_questions_path).exists():
        raise FileNotFoundError(f"[Data Error] Train questions file not found: {train_questions_path}.")
    if not Path(train_annotations_path).exists():
        raise FileNotFoundError(f"[Data Error] Train annotations file not found: {train_annotations_path}.")

    with open(train_questions_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)["questions"]
    with open(train_annotations_path, "r", encoding="utf-8") as f:
        a_data = json.load(f)["annotations"]

    if exclude_image_ids is not None:
        exclude_set = set(exclude_image_ids)
        q_data = [q for q in q_data if q["image_id"] not in exclude_set]
        if num_train_questions is not None and num_train_questions > 0:
            q_data = q_data[:num_train_questions]
        ann_by_qid = {ann["question_id"]: ann for ann in a_data}
        a_data = [ann_by_qid[q["question_id"]] for q in q_data if q["question_id"] in ann_by_qid]
    else:
        if num_train_questions is not None and num_train_questions > 0:
            q_data = q_data[:num_train_questions]
            a_data = a_data[:num_train_questions]

    # 1. Build question vocabulary
    word_counter = Counter()
    for item in q_data:
        tokens = clean_text(item["question"])
        word_counter.update(tokens)

    word2idx = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    idx2word = [PAD_TOKEN, UNK_TOKEN]

    for word, count in word_counter.most_common():
        if count >= min_word_freq:
            word2idx[word] = len(idx2word)
            idx2word.append(word)

    # 2. Build answer vocabulary with normalization
    answer_counter = Counter()
    for ann in a_data:
        for ans_obj in ann.get("answers", []):
            norm_ans = normalize_answer(ans_obj["answer"])
            if norm_ans:
                answer_counter[norm_ans] += 1

    top_answers = [ans for ans, _ in answer_counter.most_common(num_answers)]
    ans2idx = {ans: idx for idx, ans in enumerate(top_answers)}
    idx2ans = top_answers

    if vocab_save_path:
        Path(vocab_save_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word2idx": word2idx,
            "idx2word": idx2word,
            "ans2idx": ans2idx,
            "idx2ans": idx2ans
        }
        with open(vocab_save_path, "w", encoding="utf-8") as vf:
            json.dump(payload, vf, ensure_ascii=False, indent=2)

    return word2idx, idx2word, ans2idx, idx2ans


def load_vocabularies(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["word2idx"], data["idx2word"], data["ans2idx"], data["idx2ans"]


def compute_soft_targets(answers: List[Dict], ans2idx: Dict[str, int], num_answers: int) -> torch.Tensor:
    target = torch.zeros(num_answers, dtype=torch.float32)
    counts = Counter()
    for ans_dict in answers:
        norm_ans = normalize_answer(ans_dict["answer"])
        counts[norm_ans] += 1

    for ans, count in counts.items():
        if ans in ans2idx:
            idx = ans2idx[ans]
            target[idx] = min(count / 3.0, 1.0)
    return target


class VQADataset(Dataset):
    def __init__(
        self,
        questions_path: str,
        annotations_path: str,
        features_h5_path: str,
        word2idx: Dict[str, int],
        ans2idx: Dict[str, int],
        max_question_len: int = 14,
        max_samples: Optional[int] = None,
        is_train: bool = True,
        include_image_ids: Optional[Set[int]] = None,
        exclude_image_ids: Optional[Set[int]] = None,
        preload_features: bool = False,
        max_missing_feature_frac: Optional[float] = None
    ):
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.max_question_len = max_question_len
        self.is_train = is_train
        self.features_h5_path = features_h5_path
        self.include_image_ids = include_image_ids
        self.exclude_image_ids = exclude_image_ids
        self.preload_features = preload_features
        self.max_missing_feature_frac = max_missing_feature_frac
        self.h5_file = None
        self.preloaded_features: Dict[int, np.ndarray] = {}
        self.missing_feature_count = 0

        if not Path(questions_path).exists():
            raise FileNotFoundError(f"[Data Error] Questions file not found: {questions_path}")
        if not Path(annotations_path).exists():
            raise FileNotFoundError(f"[Data Error] Annotations file not found: {annotations_path}")

        with open(questions_path, "r", encoding="utf-8") as f:
            raw_questions = json.load(f)["questions"]
        with open(annotations_path, "r", encoding="utf-8") as f:
            raw_annotations = json.load(f)["annotations"]

        if include_image_ids is not None or exclude_image_ids is not None:
            inc_set = set(include_image_ids) if include_image_ids is not None else None
            exc_set = set(exclude_image_ids) if exclude_image_ids is not None else None

            filtered_questions = []
            for q in raw_questions:
                img_id = q["image_id"]
                if inc_set is not None and img_id not in inc_set:
                    continue
                if exc_set is not None and img_id in exc_set:
                    continue
                filtered_questions.append(q)
            raw_questions = filtered_questions

            if max_samples is not None and max_samples > 0:
                raw_questions = raw_questions[:max_samples]

            ann_by_qid = {ann["question_id"]: ann for ann in raw_annotations}
        else:
            if max_samples is not None and max_samples > 0:
                raw_questions = raw_questions[:max_samples]
                raw_annotations = raw_annotations[:max_samples]

            ann_by_qid = {ann["question_id"]: ann for ann in raw_annotations}

        candidate_samples = []
        self.discarded_count = 0
        self.val_oov_count = 0

        for q_item in raw_questions:
            qid = q_item["question_id"]
            if qid not in ann_by_qid:
                continue
            ann_item = ann_by_qid[qid]
            raw_answers = ann_item.get("answers", [])
            
            normalized_answers = [normalize_answer(a["answer"]) for a in raw_answers]
            has_valid_answer = any(na in self.ans2idx for na in normalized_answers)

            if is_train and not has_valid_answer:
                self.discarded_count += 1
                continue

            if not is_train and not has_valid_answer:
                self.val_oov_count += 1

            candidate_samples.append({
                "question_id": qid,
                "image_id": q_item["image_id"],
                "question_text": q_item["question"],
                "answer_type": ann_item.get("answer_type", "other"),
                "answers": raw_answers,
                "normalized_answers": normalized_answers
            })

        h5_path_obj = Path(features_h5_path) if features_h5_path else None
        h5_exists = h5_path_obj.exists() if h5_path_obj else False

        if h5_exists:
            with h5py.File(features_h5_path, "r") as h5_f:
                available_keys = set(h5_f.keys())
            available_ids = {int(k) for k in available_keys if k.isdigit()}

            valid_samples = []
            missing_count = 0
            for item in candidate_samples:
                iid = item["image_id"]
                if (iid in available_ids) or (str(iid) in available_keys):
                    valid_samples.append(item)
                else:
                    missing_count += 1

            self.samples = valid_samples
            self.missing_feature_count = missing_count

            total_candidates = len(candidate_samples)
            missing_frac = (missing_count / total_candidates) if total_candidates > 0 else 0.0

            if max_missing_feature_frac is not None and missing_frac > max_missing_feature_frac:
                raise RuntimeError(
                    f"[Data Error] Missing image features for {missing_count}/{total_candidates} samples "
                    f"({missing_frac * 100:.2f}%), exceeding allowed maximum fraction "
                    f"{max_missing_feature_frac * 100:.2f}% in {features_h5_path}!"
                )
            elif max_missing_feature_frac is None and missing_count > 0:
                print(
                    f"[Data Warning] Skipped {missing_count} samples due to missing features in {features_h5_path}."
                )
        else:
            self.samples = candidate_samples
            self.missing_feature_count = 0

        if self.preload_features:
            if not h5_exists:
                raise FileNotFoundError(
                    f"[Data Error] Features file not found for preloading: {features_h5_path}. "
                    f"Please run extract_features.py first!"
                )
            needed_ids = {s["image_id"] for s in self.samples}
            with h5py.File(features_h5_path, "r") as h5_f:
                for img_id in needed_ids:
                    k_str = str(img_id)
                    if k_str in h5_f:
                        self.preloaded_features[img_id] = np.asarray(h5_f[k_str][:], dtype=np.float32)
                    elif img_id in h5_f:
                        self.preloaded_features[img_id] = np.asarray(h5_f[img_id][:], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_h5(self):
        if self.h5_file is None:
            if not Path(self.features_h5_path).exists():
                raise FileNotFoundError(
                    f"[Data Error] Features file not found: {self.features_h5_path}. "
                    f"Please run extract_features.py first!"
                )
            self.h5_file = h5py.File(self.features_h5_path, "r")
        return self.h5_file

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image_id = item["image_id"]
        qid = item["question_id"]
        q_text = item["question_text"]
        ans_type = item["answer_type"]
        answers = item["answers"]

        if self.preload_features:
            if image_id in self.preloaded_features:
                img_feature = torch.from_numpy(self.preloaded_features[image_id])
            elif str(image_id) in self.preloaded_features:
                img_feature = torch.from_numpy(self.preloaded_features[str(image_id)])
            else:
                raise KeyError(
                    f"[Data Error] Image features for image_id={image_id} missing in preloaded features!"
                )
        else:
            h5 = self._get_h5()
            img_key = str(image_id)
            if img_key not in h5:
                raise KeyError(
                    f"[Data Error] Image features for image_id={image_id} missing in {self.features_h5_path}! "
                    f"Verify that all images were extracted via extract_features.py."
                )
            img_feature = torch.from_numpy(h5[img_key][:]).float()

        token_indices, actual_len = tokenize_question(q_text, self.word2idx, self.max_question_len)
        question_tensor = torch.tensor(token_indices, dtype=torch.long)
        target_tensor = compute_soft_targets(answers, self.ans2idx, len(self.ans2idx))

        return {
            "image_feature": img_feature,
            "question": question_tensor,
            "length": actual_len,
            "target": target_tensor,
            "question_id": qid,
            "image_id": image_id,
            "question_text": q_text,
            "answer_type": ans_type,
            "answers": [a["answer"] for a in answers]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    image_features = torch.stack([item["image_feature"] for item in batch], dim=0)
    questions = torch.stack([item["question"] for item in batch], dim=0)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    targets = torch.stack([item["target"] for item in batch], dim=0)

    question_ids = [item["question_id"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    question_texts = [item["question_text"] for item in batch]
    answer_types = [item["answer_type"] for item in batch]
    all_answers = [item["answers"] for item in batch]

    return {
        "image_feature": image_features,
        "question": questions,
        "length": lengths,
        "target": targets,
        "question_id": question_ids,
        "image_id": image_ids,
        "question_text": question_texts,
        "answer_type": answer_types,
        "all_answers": all_answers
    }


def get_dataloaders(cfg):
    split_mode = cfg.data.get("split_mode", "legacy") if hasattr(cfg.data, "get") else getattr(cfg.data, "split_mode", "legacy")

    if split_mode == "protocol":
        from src.splits import build_split
        split = build_split(cfg)
        dev_ids = split["dev_image_ids"]

        preload_features = cfg.data.get("preload_features", False) if hasattr(cfg.data, "get") else getattr(cfg.data, "preload_features", False)
        max_missing_feature_frac = cfg.data.get("max_missing_feature_frac", None) if hasattr(cfg.data, "get") else getattr(cfg.data, "max_missing_feature_frac", None)

        num_train = cfg.data.get("num_train_questions", None) if hasattr(cfg.data, "get") else getattr(cfg.data, "num_train_questions", None)
        if num_train == 0:
            num_train = None

        num_dev = cfg.data.get("num_dev_questions", None) if hasattr(cfg.data, "get") else getattr(cfg.data, "num_dev_questions", None)
        if num_dev == 0:
            num_dev = None

        word2idx, idx2word, ans2idx, idx2ans = build_vocabularies(
            train_questions_path=cfg.data.train_questions,
            train_annotations_path=cfg.data.train_annotations,
            num_train_questions=num_train,
            num_answers=cfg.data.num_answers,
            min_word_freq=cfg.data.min_word_freq,
            vocab_save_path=cfg.paths.vocab_json,
            exclude_image_ids=dev_ids
        )

        train_dataset = VQADataset(
            questions_path=cfg.data.train_questions,
            annotations_path=cfg.data.train_annotations,
            features_h5_path=cfg.paths.train_features_h5,
            word2idx=word2idx,
            ans2idx=ans2idx,
            max_question_len=cfg.data.max_question_len,
            max_samples=num_train,
            is_train=True,
            exclude_image_ids=dev_ids,
            preload_features=preload_features,
            max_missing_feature_frac=max_missing_feature_frac
        )

        val_dataset = VQADataset(
            questions_path=cfg.data.train_questions,
            annotations_path=cfg.data.train_annotations,
            features_h5_path=cfg.paths.train_features_h5,
            word2idx=word2idx,
            ans2idx=ans2idx,
            max_question_len=cfg.data.max_question_len,
            max_samples=num_dev,
            is_train=False,
            include_image_ids=dev_ids,
            preload_features=preload_features,
            max_missing_feature_frac=max_missing_feature_frac
        )

        val_oov_pct = (val_dataset.val_oov_count / max(len(val_dataset), 1)) * 100.0

        print("\n" + "=" * 70)
        print(" [Data Statistics - Protocol Split (ADR-011)]")
        print("=" * 70)
        print(f" - Split seed:                        {split.get('split_seed')}")
        print(f" - Dev image fraction:                {split.get('dev_image_fraction')}")
        print(f" - Total train images:                {split.get('n_train_images'):,}")
        print(f" - Total dev images:                  {split.get('n_dev_images'):,}")
        print(f" - Total train questions (raw):       {split.get('n_train_questions'):,}")
        print(f" - Total dev questions (raw):         {split.get('n_dev_questions'):,}")
        print(f" - Question vocabulary size:          {len(word2idx):,} words (min_word_freq={cfg.data.min_word_freq})")
        print(f" - Answer vocabulary size:            {len(ans2idx):,} (Top-{cfg.data.num_answers})")
        print(f" - Total Train samples after filter:  {len(train_dataset):,} (discarded without Top-K answer: {train_dataset.discarded_count:,})")
        print(f" - Total Dev (selection) samples:     {len(val_dataset):,}")
        print(f" - Dev samples with answer OOV:       {val_oov_pct:.2f}% ({val_dataset.val_oov_count:,} of {len(val_dataset):,})")
        print(f" - Features preloaded in RAM:         {preload_features}")
        print("=" * 70 + "\n")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, word2idx, idx2word, ans2idx, idx2ans

    # Legacy mode: unchanged behavior
    word2idx, idx2word, ans2idx, idx2ans = build_vocabularies(
        train_questions_path=cfg.data.train_questions,
        train_annotations_path=cfg.data.train_annotations,
        num_train_questions=cfg.data.num_train_questions,
        num_answers=cfg.data.num_answers,
        min_word_freq=cfg.data.min_word_freq,
        vocab_save_path=cfg.paths.vocab_json
    )

    train_dataset = VQADataset(
        questions_path=cfg.data.train_questions,
        annotations_path=cfg.data.train_annotations,
        features_h5_path=cfg.paths.train_features_h5,
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_question_len=cfg.data.max_question_len,
        max_samples=cfg.data.num_train_questions,
        is_train=True
    )

    val_dataset = VQADataset(
        questions_path=cfg.data.val_questions,
        annotations_path=cfg.data.val_annotations,
        features_h5_path=cfg.paths.val_features_h5,
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_question_len=cfg.data.max_question_len,
        max_samples=cfg.data.num_val_questions,
        is_train=False
    )

    val_oov_pct = (val_dataset.val_oov_count / max(len(val_dataset), 1)) * 100.0

    print("\n" + "=" * 70)
    print(" [Data Statistics and VQA Vocabularies]")
    print("=" * 70)
    print(f" - Question vocabulary size:          {len(word2idx):,} words (min_word_freq={cfg.data.min_word_freq})")
    print(f" - Answer vocabulary size:            {len(ans2idx):,} (Top-{cfg.data.num_answers})")
    print(f" - Total Train samples after filter:  {len(train_dataset):,} (discarded without Top-K answer: {train_dataset.discarded_count:,})")
    print(f" - Total Val samples (unfiltered):    {len(val_dataset):,}")
    print(f" - Val samples with answer OOV:       {val_oov_pct:.2f}% ({val_dataset.val_oov_count:,} of {len(val_dataset):,})")
    print("=" * 70 + "\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, word2idx, idx2word, ans2idx, idx2ans


def get_test_loader(cfg, word2idx: Dict[str, int], ans2idx: Dict[str, int]) -> DataLoader:
    """
    Create DataLoader for test evaluation.
    Under protocol mode: uses val2014 questions, annotations and features.
    Under legacy mode: uses val2014 (same set as legacy val_loader).
    """
    data_cfg = cfg.data if hasattr(cfg, "data") else cfg.get("data", {})
    paths_cfg = cfg.paths if hasattr(cfg, "paths") else cfg.get("paths", {})
    training_cfg = cfg.training if hasattr(cfg, "training") else cfg.get("training", {})

    preload_features = data_cfg.get("preload_features", False) if hasattr(data_cfg, "get") else getattr(data_cfg, "preload_features", False)
    max_missing_feature_frac = data_cfg.get("max_missing_feature_frac", None) if hasattr(data_cfg, "get") else getattr(data_cfg, "max_missing_feature_frac", None)

    val_questions = data_cfg.get("val_questions") if hasattr(data_cfg, "get") else getattr(data_cfg, "val_questions")
    val_annotations = data_cfg.get("val_annotations") if hasattr(data_cfg, "get") else getattr(data_cfg, "val_annotations")
    val_features_h5 = paths_cfg.get("val_features_h5") if hasattr(paths_cfg, "get") else getattr(paths_cfg, "val_features_h5")
    max_question_len = data_cfg.get("max_question_len", 14) if hasattr(data_cfg, "get") else getattr(data_cfg, "max_question_len", 14)
    num_val_questions = data_cfg.get("num_val_questions", None) if hasattr(data_cfg, "get") else getattr(data_cfg, "num_val_questions", None)
    if num_val_questions == 0:
        num_val_questions = None

    batch_size = training_cfg.get("batch_size", 32) if hasattr(training_cfg, "get") else getattr(training_cfg, "batch_size", 32)
    num_workers = training_cfg.get("num_workers", 0) if hasattr(training_cfg, "get") else getattr(training_cfg, "num_workers", 0)

    test_dataset = VQADataset(
        questions_path=val_questions,
        annotations_path=val_annotations,
        features_h5_path=val_features_h5,
        word2idx=word2idx,
        ans2idx=ans2idx,
        max_question_len=max_question_len,
        max_samples=num_val_questions,
        is_train=False,
        preload_features=preload_features,
        max_missing_feature_frac=max_missing_feature_frac
    )

    test_oov_pct = (test_dataset.val_oov_count / max(len(test_dataset), 1)) * 100.0
    print(f" [Test Loader] Samples: {len(test_dataset):,}, OOV answers: {test_oov_pct:.2f}% ({test_dataset.val_oov_count:,})")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return test_loader
