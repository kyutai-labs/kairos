import json
import random
from pathlib import Path

from kairos.evaluation.olmes.fewshot_sources import FEWSHOT_SOURCES, MMLU_FEWSHOT_SOURCE


def format_allenai(data):
    r = {
        "question": data.get("question_stem") or data.get("question"),
        "choices": data["choices"]["text"],
    }
    answer_key = data["answerKey"]
    for i, label in enumerate(data["choices"]["label"]):
        if label == answer_key:
            r["gold"] = i
            break
    return r


def load_allenai(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            r = format_allenai(data)
            if "gold" not in r:
                continue
            yield r


def load_hellaswag(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            r = {
                "question": data["activity_label"] + ": " + data["ctx"]
                if "activity_label" in data
                else data["ctx"],
                "choices": data["endings"],
                "gold": int(data["label"]),
            }
            yield r


def load_winogrande(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            yield {
                "question": data["sentence"],
                "choices": [data["option1"], data["option2"]],
                "gold": int(data["answer"]) - 1,
            }


def load_mmlu(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            yield {
                "question": data["question"],
                "choices": data["choices"],
                "gold": data["answer"],
                "subject": data["subject"],
            }


def load_piqa(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            yield {
                "question": data["goal"],
                "choices": [data["sol1"], data["sol2"]],
                "gold": data["label"],
            }


def load_siqa(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            yield {
                "question": data["context"] + " " + data["question"],
                "choices": [data["answerA"], data["answerB"], data["answerC"]],
                "gold": int(data["label"]) - 1,
            }


def load_boolq(path: Path):
    with path.open() as fin:
        for line in fin:
            data = json.loads(line)
            yield {
                "question": data["passage"] + "\n" + data["question"],
                "choices": ["no", "yes"],
                "gold": 1 if data["answer"] else 0,
            }


def load_mmlu_fewshots(path: Path):
    files = path.glob("*.jsonl")
    subject_data = {}
    for file in files:
        # Skip the cross-subject aggregate file: keying by examples[0]["subject"]
        # would let it overwrite a real per-subject entry (filesystem-order dependent).
        if file.stem == "all_subjects":
            continue
        examples = []
        with file.open() as f:
            for line in f:
                examples.append(json.loads(line))
        subject = examples[0]["subject"]
        subject_data[subject] = examples
    return subject_data


def format_mcf_prompt(
    example: dict, task_name: str, include_answer: bool = False
) -> str:
    """
    Format prompt for MCF according to OLMES standard.

    Standard format: "Question: <question>\n A. <choice1>\n B. <choice2>...\nAnswer:"

    Special formats:
    - PIQA: "Goal: <goal>\n A. <choice1>...\nAnswer:"
    - HellaSwag: "<context>\nChoose the best continuation:\n A. <choice1>...\nAnswer:"
    - WinoGrande: "Fill in the blank: <question>\n A. <choice1>...\nAnswer:"
    """
    question = example["question"]
    choices = example["choices"]
    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ][: len(choices)]

    if task_name == "piqa":
        prompt = f"Goal: {question}"
    elif task_name == "hellaswag":
        prompt = f"{question}\nChoose the best continuation:"
    elif task_name == "winogrande":
        prompt = f"Fill in the blank: {question}"
    else:
        prompt = f"Question: {question}"

    for letter, choice in zip(letters, choices):
        prompt += f"\n {letter}. {choice}"

    if include_answer:
        gold_idx = example["gold"] if "gold" in example else example["answer"]
        gold_letter = letters[gold_idx]
        prompt += f"\nAnswer: {gold_letter}"
    else:
        prompt += "\nAnswer:"

    return prompt


def format_cf_prompt(example: dict, task_name: str, choice_idx: int) -> str:
    """
    Format prompt for Cloze Formulation (CF).
    Returns the prompt with a specific choice completion for scoring.

    For HellaSwag and WinoGrande, prefixes/suffixes are removed for pure language modeling.
    Standard format: "Question: <question>\nAnswer: <choice>"
    """
    question = example["question"]
    choice = example["choices"][choice_idx]

    if task_name == "piqa":
        return f"Goal: {question}\nAnswer: {choice}"
    elif task_name == "hellaswag":
        return f"{question} {choice}"
    elif task_name == "winogrande":
        return question.replace("_", choice)
    else:
        return f"Question: {question}\nAnswer: {choice}"


def sample_data(data: list, max_samples: int, seed: int = 1234) -> list:
    """
    Sample data according to OLMES standard:
    If dataset has > 1500 instances, sample 1000 instances using random(1234).
    """
    if len(data) > 1500:
        random.seed(seed)
        return random.sample(data, min(max_samples, len(data)))
    return data


def batching(iterator, batch_size=16):
    batch = []
    for example in iterator:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def load_fewshot_examples(
    task_name: str, k_shot: int
) -> list[dict] | dict[str, list[dict]] | None:
    if k_shot == 0:
        return None

    raw_examples = FEWSHOT_SOURCES.get(task_name, [])[:k_shot]

    if task_name in ["arc_easy", "arc_challenge", "csqa", "obqa"]:
        return [format_allenai(ex) for ex in raw_examples]
    elif task_name == "boolq":
        return [
            {
                "question": ex["passage"] + "\n" + ex["question"],
                "choices": ["no", "yes"],
                "gold": 1 if ex["label"] else 0,
            }
            for ex in raw_examples
        ]
    elif task_name == "hellaswag":
        return [
            {
                "question": ex["activity_label"] + ": " + ex["ctx"]
                if "activity_label" in ex
                else ex["ctx"],
                "choices": ex["endings"],
                "gold": int(ex["label"]),
            }
            for ex in raw_examples
        ]
    elif task_name == "piqa":
        return [
            {
                "question": ex["goal"],
                "choices": [ex["sol1"], ex["sol2"]],
                "gold": ex["label"],
            }
            for ex in raw_examples
        ]
    elif task_name == "siqa":
        return [
            {
                "question": ex["context"] + " " + ex["question"],
                "choices": [ex["answerA"], ex["answerB"], ex["answerC"]],
                "gold": int(ex["label"]) - 1,
            }
            for ex in raw_examples
        ]
    elif task_name == "winogrande":
        return [
            {
                "question": ex["sentence"],
                "choices": [ex["option1"], ex["option2"]],
                "gold": int(ex["answer"]) - 1,
            }
            for ex in raw_examples
        ]
    elif task_name == "mmlu":
        fewshots = load_mmlu_fewshots(Path(MMLU_FEWSHOT_SOURCE))
        return fewshots

    else:
        raise ValueError(f"Unknown task name: {task_name}")
