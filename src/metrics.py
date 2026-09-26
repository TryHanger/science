from typing import List


def simplified_vqa_score(pred_norm: str, answers_norm: List[str]) -> float:
    """
    Simplified VQA score:
    min(count(pred in answers) / 3.0, 1.0).
    Empty list -> 0.0.
    Normalization is expected to be performed by the caller.
    """
    if not answers_norm:
        return 0.0
    matches = sum(1 for a in answers_norm if a == pred_norm)
    return min(matches / 3.0, 1.0)


def official_vqa_score(pred_norm: str, answers_norm: List[str]) -> float:
    """
    Official VQA accuracy score following the standard evaluation protocol:
    Average over all i in [0..len-1] of min(count(pred in answers without i-th) / 3.0, 1.0).
    Empty list -> 0.0.
    Normalization is expected to be performed by the caller.
    """
    n = len(answers_norm)
    if n == 0:
        return 0.0

    total_matches = sum(1 for a in answers_norm if a == pred_norm)

    total_score = 0.0
    for a in answers_norm:
        matches_without_i = total_matches - 1 if a == pred_norm else total_matches
        total_score += min(matches_without_i / 3.0, 1.0)

    return total_score / n
