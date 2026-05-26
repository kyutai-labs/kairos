import re
import string
from collections import Counter

import Levenshtein


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, gold: list[str]) -> float:
    prediction = normalize_answer(prediction)
    for e in gold:
        if prediction == normalize_answer(e):
            return 1.0
    return 0.0


def fuzzy_exact_match(prediction: str, gold: list[str]) -> float:
    # Levenshtein distance
    normed_pred = normalize_answer(str(prediction))
    res = 0.0
    for g in gold:
        check_sim = Levenshtein.ratio(normed_pred, normalize_answer(str(g)))
        res = max(res, check_sim)
    return res


def f1_score(prediction: str, gold: list[str]) -> float:
    res = 0.0
    for g in gold:
        res = max(res, _compute_f1(prediction, g))
    return res


def _compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
