import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
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
    num_train_questions: int,
    num_answers: int = 1000,
    min_word_freq: int = 2,
    vocab_save_path: Optional[str] = None
) -> Tuple[Dict[str, int], List[str], Dict[str, int], List[str]]:
    if not Path(train_questions_path).exists():
        raise FileNotFoundError(f"[Data Error] Train questions file not found: {train_questions_path}.")
    if not Path(train_annotations_path).exists():
        raise FileNotFoundError(f"[Data Error] Train annotations file not found: {train_annotations_path}.")

    with open(train_questions_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)["questions"]
    with open(train_annotations_path, "r", encoding="utf-8") as f:
        a_data = json.load(f)["annotations"]

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
        is_train: bool = True
    ):
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.max_question_len = max_question_len
        self.is_train = is_train
        self.features_h5_path = features_h5_path
        self.h5_file = None

        if not Path(questions_path).exists():
            raise FileNotFoundError(f"[Data Error] Questions file not found: {questions_path}")
        if not Path(annotations_path).exists():
            raise FileNotFoundError(f"[Data Error] Annotations file not found: {annotations_path}")

        with open(questions_path, "r", encoding="utf-8") as f:
            raw_questions = json.load(f)["questions"]
        with open(annotations_path, "r", encoding="utf-8") as f:
            raw_annotations = json.load(f)["annotations"]

        if max_samples is not None and max_samples > 0:
            raw_questions = raw_questions[:max_samples]
            raw_annotations = raw_annotations[:max_samples]

        ann_by_qid = {ann["question_id"]: ann for ann in raw_annotations}

        self.samples = []
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

            self.samples.append({
                "question_id": qid,
                "image_id": q_item["image_id"],
                "question_text": q_item["question"],
                "answer_type": ann_item.get("answer_type", "other"),
                "answers": raw_answers,
                "normalized_answers": normalized_answers
            })

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
