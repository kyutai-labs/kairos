import csv
import json
import random
from collections.abc import Callable
from collections.abc import Generator as ABCGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairos.evaluation.metrics import normalize_answer


def is_similar(ans1: str, ans2: str | list[str]) -> bool:
    normed_ans1 = normalize_answer(ans1)
    set1 = set(normed_ans1.split())
    if isinstance(ans2, str):
        ans2 = [ans2]
    for a in ans2:
        normed_a2 = normalize_answer(a)
        set2 = set(normed_a2.split())
        if set1 == set2:
            return True
    return False


@dataclass
class Task:
    prompt: str
    answer_format: str
    dataloader: Callable[..., Any]
    prefix: str = "The following is a list of elementary questions and answers."
    name: str = ""

    def format_example(
        self, example: dict[str, Any], answer: str, prefix: str | None = None
    ) -> str:
        s = self.prompt.format(answer=answer, **example).strip()
        if prefix:
            s = prefix + s
        return s

    def format_choices(
        self, example: dict[str, str | list[str]], prefix: str | None = None
    ) -> list[str]:
        return [self.format_example(example, c, prefix) for c in example["choices"]]

    def format_answers(
        self, example: dict[str, str | list[str]], prefix: str | None = None
    ) -> list[str]:
        return [
            self.answer_format.format(answer=c, prefix=prefix)
            for c in example["choices"]
        ]


def load_std(path: Path, split: str) -> ABCGenerator[dict[str, str | int], None, None]:
    with (path / f"{split}.jsonl").open() as fin:
        for line in fin:
            data = json.loads(line)
            data["gold"] = data["answer"]
            del data["answer"]
            yield data


def format_allenai(data: dict[str, Any]) -> dict[str, Any]:
    r: dict[str, Any] = {"question": data["question"]["stem"], "choices": []}
    for i, c in enumerate(data["question"]["choices"]):
        r["choices"].append(c["text"])
        if c["label"] == data["answerKey"]:
            r["gold"] = i
    if "context" in data["question"]:
        r["context"] = data["question"]["context"]
    return r


def load_allenai(path: Path, split: str) -> ABCGenerator[dict, None, None]:
    with (path / f"{split}.jsonl").open() as fin:
        for line in fin:
            data = json.loads(line)
            r = format_allenai(data)
            if not "gold" in r:
                continue
            yield r


def load_hellaswag(path: Path, split: str) -> ABCGenerator[dict, None, None]:
    with (path / f"{split}.jsonl").open() as fin:
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


def load_winogrande(path: Path, split: str) -> ABCGenerator[dict, None, None]:
    with (path / f"{split}.jsonl").open() as fin:
        for line in fin:
            data = json.loads(line)
            yield {
                "gold": data["label"],
                "question": data["end"],
                "choices": data["ctx"],
            }


def load_mmlu(path: Path, split: str, topic: str) -> ABCGenerator[dict, None, None]:
    split = {"train": "dev", "train_olmes": "dev", "valid": "val", "test": "test"}[
        split
    ]
    filename = path / split / f"{topic}_{split}.csv"
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    with filename.open(newline="") as fin:
        csvreader = csv.reader(fin)
        for row in csvreader:
            yield {
                "question": row[0],
                "choices": row[1:5],
                "gold": letter_to_idx[row[5]],
            }


def load_temporal(
    path: Path,
    target_year: int,
    n_choices: int,
    n_distractor: int,
    k_shot: int,
) -> ABCGenerator[dict, None, None]:
    assert n_choices > n_distractor, (
        "n_choices should be greater than n_distractor + 1, the correct answer"
    )
    with path.open() as fin:
        ind = 0
        for i, line in enumerate(fin):
            data = json.loads(line)

            all_answers = [ans[0] for ans in data["multi_choice_answers"]]
            flatten_all_answers = [item for sublist in all_answers for item in sublist]
            all_years = [int(ans[1]) for ans in data["multi_choice_answers"]]
            nested_years = [[y] * len(ans) for ans, y in zip(all_answers, all_years)]
            flatten_all_years = [item for sublist in nested_years for item in sublist]

            if target_year not in all_years:
                continue

            if None in flatten_all_answers:
                print(f"Skipping question due to None answer: {data['question']}")
                continue

            correct_answers = all_answers[all_years.index(target_year)]
            correct_answers = [
                ans
                for ans in correct_answers
                if ans is not None and len(ans.strip()) > 0
            ]

            if len(correct_answers) == 0:
                print(
                    f"Skipping question due to empty correct answer after filtering: {data['question']}"
                )
                continue

            gt_answer = correct_answers
            choices = [random.Random(target_year + i).choice(correct_answers)]
            choice_years = [target_year]

            unique_answers = list(
                {
                    x: (x, y)
                    for x, y in zip(flatten_all_answers, flatten_all_years)
                    if y != -1
                    and not is_similar(x, correct_answers)
                    and len(x.strip()) > 0
                }.values()
            )
            distractors = set(
                [
                    flatten_all_answers[j]
                    for j, val in enumerate(flatten_all_years)
                    if val == -1
                ]
            )
            unique_answers = sorted(
                unique_answers, key=lambda x: abs(x[1] - target_year)
            )

            while len(choices) < n_choices - n_distractor:
                if unique_answers:
                    if (
                        is_similar(unique_answers[0][0], choices + correct_answers)
                        or len(unique_answers[0][0].strip()) == 0
                    ):
                        del unique_answers[0]
                        continue
                    choices.append(unique_answers[0][0])
                    choice_years.append(unique_answers[0][1])
                    del unique_answers[0]
                else:
                    break

            for di in distractors:
                if len(choices) < n_choices:
                    if (
                        is_similar(di, choices + correct_answers)
                        or len(di.strip()) == 0
                    ):
                        continue
                    choices.append(di)
                    choice_years.append(-1)
                else:
                    break

            if None in choices or len(choices) < 1:
                continue

            perm = list(range(len(choices)))
            random.Random(target_year + i).shuffle(perm)
            choices = [choices[i].strip() for i in perm]
            choice_years = [choice_years[i] for i in perm]
            correct_index = perm.index(0)

            if k_shot < 0 and ind < -k_shot:
                ind += 1
                continue
            elif k_shot > 0 and ind >= k_shot:
                break

            result = {
                "choices": choices,
                "choice_years": choice_years,
                "gold": correct_index,
                "gt": gt_answer,
                "relation": data.get("relation", None),
                "subject_rank": data.get("subject_rank", None),
            }
            result["question"] = data["question"].format(year=target_year)
            yield result
            ind += 1


tasks = {
    "arc_easy": Task(
        name="arc_easy",
        prompt="Question: {question}\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_allenai,
    ),
    "arc_challenge": Task(
        name="arc_challenge",
        prompt="Question: {question}\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_allenai,
    ),
    "obqa": Task(
        name="obqa",
        prompt="Question: {question}\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_allenai,
    ),
    "csqa": Task(
        name="csqa",
        prompt="Question: {question}\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_allenai,
    ),
    "hellaswag": Task(
        name="hellaswag",
        prompt="{question} {answer}",
        answer_format="{answer}",
        dataloader=load_hellaswag,
        prefix="The following is a list of commonsense facts.",
    ),
    "piqa": Task(
        name="piqa",
        prompt="{question} {answer}",
        answer_format="{answer}",
        dataloader=load_hellaswag,
        prefix="The following is a list of elementary facts.",
    ),
    "siqa": Task(
        name="siqa",
        prompt="Question: {question}\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_hellaswag,
    ),
    "winogrande": Task(
        name="winogrande",
        prompt="{answer}{question}",
        answer_format="{prefix}{answer}",
        dataloader=load_winogrande,
        prefix="The following is a list of elementary facts.",
    ),
    "boolq": Task(
        name="boolq",
        prompt="{context}\nQuestion: {question}?\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_std,
    ),
    "temporal": Task(
        name="temporal",
        prompt="Question: {question}\nAnswer: {answer}",
        answer_format="Answer: {answer}",
        dataloader=load_temporal,
        prefix="The following is a list of questions.",
    ),
}
