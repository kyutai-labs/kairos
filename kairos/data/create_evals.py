import json
import os
import queue
import random
import threading
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from kairos.data.templates import (
    PREDEFINED_TEMPLATE,
    TEMPLATE_AR_QUESTION,
    TEMPLATE_DISTRACTOR,
    TEMPLATE_PH_QUESTION,
    TEMPLATE_QUESTION,
)
from kairos.evaluation.metrics import fuzzy_exact_match

INVERTED_PROPERTIES = ["P39", "P166"]

load_dotenv()  # Loads variables from .env
DATA_PATH = Path(os.getenv("DATA_DIR", "./data"))

""" Select WikiData quadruplets and create subject dictionary """


def create_subject_dict(wiki_quad_data, limit_past_time: int = 2000):
    """From WikiData dumps, filter and create a nested dict
    Args:
        wiki_quad_data (list): list of quadruplets from WikiData dumps
        limit_past_time (int, optional): Filter every objects from before this year. Defaults to 2000.
    Returns:
        dict: Nested dictionary with structure "subject"->"relation"->"time"->"object list"
    """
    subject_dict: dict[str, dict] = {}
    count_error_parsing_time = 0
    for item in tqdm(wiki_quad_data):
        if not isinstance(item["object"], list):
            item["object"] = [item["object"]]
        if not isinstance(item["property"], list):
            item["property"] = ["P000", item["property"]]
        if (
            item["property"][0] in INVERTED_PROPERTIES
        ):  # Invert subject and object for instance of and subclass of relations
            item["subject"], item["object"] = item["object"][-1], [item["subject"]]

        subject = item["subject"]
        if subject not in subject_dict:
            subject_dict[subject] = {}
            subject_dict[subject]["subject_rank"] = []

        if item["property"][1] not in subject_dict[subject]:
            subject_dict[subject][item["property"][1]] = {}

        try:
            year_1 = int(item["time"][0].split("-")[0])
            year_2 = (
                2025 if item["time"][1] is None else int(item["time"][1].split("-")[0])
            )

            if year_2 < year_1:
                continue
            elif year_2 < limit_past_time:
                continue
            elif year_1 < limit_past_time:
                year_1 = limit_past_time
        except AttributeError:
            # print("Error parsing time for item: ", item, v)
            count_error_parsing_time += 1
            continue
        times = [year for year in range(year_1, year_2 + 1)]
        for a_time in times:
            if a_time not in subject_dict[subject][item["property"][1]]:
                subject_dict[subject][item["property"][1]][a_time] = [
                    item["object"][-1]
                ]
            else:
                subject_dict[subject][item["property"][1]][a_time].append(
                    item["object"][-1]
                )

        if len(subject_dict[subject]) == 0:
            subject_dict.pop(subject)
            continue

        subject_dict[subject]["subject_rank"].append(item.get("subject_rank", 0))

    for subject in subject_dict:
        if len(subject_dict[subject]["subject_rank"]) == 0:
            subject_dict[subject]["subject_rank"] = -1
        else:
            subject_dict[subject]["subject_rank"] = sum(
                subject_dict[subject]["subject_rank"]
            ) / len(subject_dict[subject]["subject_rank"])

    print(f"Count error parsing time: {count_error_parsing_time}")
    print(f"Total subjects in subject_dict: {len(subject_dict)}")
    print(
        f"Proportion kept subjects: {len(subject_dict) * 100 / len(wiki_quad_data):.2f}%"
    )
    return subject_dict


def filter_subject_dict(subject_dict, perc_popularity: int = 20):
    filtered_subject_dict = {}
    for k, v in subject_dict.items():
        if set(v.keys()) == set(["subject_rank", "population"]):
            continue
        elif "subject_rank" not in v.keys():
            continue
        else:
            sub_dict = {}
            for rel in list(v.keys()):
                if rel == "population":
                    continue
                if rel == "subject_rank":
                    sub_dict[rel] = v[rel]
                    continue
                if (
                    len(
                        list(
                            set(
                                [
                                    answer[0] if isinstance(answer, list) else answer
                                    for year, answer in v[rel].items()
                                    if int(year) >= 2018 and int(year) <= 2025
                                ]
                            )
                        )
                    )
                    >= 3
                ):
                    sub_dict[rel] = v[rel]
                    continue
            if set(sub_dict.keys()) == set(["subject_rank", "population"]):
                continue
            elif list(sub_dict.keys()) == ["subject_rank"]:
                continue
            else:
                filtered_subject_dict[k] = sub_dict

    subject_ranks: defaultdict[str, int] = defaultdict(lambda: 0)
    for k in filtered_subject_dict.keys():
        subject_ranks[k] += filtered_subject_dict[k]["subject_rank"]
    percentile = 100 - perc_popularity
    # TOp 10% highest rank subjects
    top_threshold = np.percentile(list(subject_ranks.values()), percentile)
    top_percent_subjects = {
        k: v
        for k, v in filtered_subject_dict.items()
        if v["subject_rank"] >= top_threshold
    }
    return top_percent_subjects


""" Create evaluation samples """


def create_mcqa(
    subject,
    relation,
    client,
    subject_dict,
    model_name,
    n_d: int,
    out_queue: queue.Queue,
    predefined_q_templates: bool = False,
    limit_past_time: int = 2000,
    max_tokens: int = 128,
    top_p: float = 0.9,
    temperature: float = 0.7,
) -> None:
    answers_time = []
    subject_rank = subject_dict[subject].get("subject_rank", -1)
    if len(subject_dict[subject]) == 0:
        out_queue.put(None)

    all_answers = []
    for a_time, object_answer in subject_dict[subject][relation].items():
        if int(a_time) <= limit_past_time:
            continue
        else:
            if object_answer is None:
                continue
            elif isinstance(object_answer, str):
                selected_answer = [object_answer]
            elif isinstance(object_answer, list):
                if None in object_answer:
                    object_answer = [ans for ans in object_answer if ans is not None]
                    if len(object_answer) == 0:
                        continue
                selected_answer = list(set(object_answer))

            answers_time.append([selected_answer, a_time])
            if isinstance(object_answer, list):
                all_answers.extend(object_answer)
            else:
                all_answers.append(object_answer)

    if None in all_answers:
        all_answers = [ans for ans in all_answers if ans is not None]
        if len(all_answers) == 0:
            out_queue.put(None)

    if predefined_q_templates:
        question_template = PREDEFINED_TEMPLATE[relation]
        question = question_template.format(subject=subject)
    else:
        if relation == "position held":
            question_prompt = {
                "role": "user",
                "content": TEMPLATE_PH_QUESTION.format(subject=subject),
            }
        elif relation == "award received":
            question_prompt = {
                "role": "user",
                "content": TEMPLATE_AR_QUESTION.format(subject=subject),
            }
        else:
            question_prompt = {
                "role": "user",
                "content": TEMPLATE_QUESTION.format(
                    subject=subject,
                    relation=relation,
                    question=PREDEFINED_TEMPLATE[relation].format(subject=subject),
                ),
            }

        completion = client.chat.completions.create(
            model=model_name,
            messages=[question_prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1,
        )
        question = (
            completion.choices[0]
            .message.content.split("\nQuestion:")[0]
            .split("Question:")[-1]
            .strip()
        )

    mc_qa = {
        "question": question,
        "multi_choice_answers": answers_time,
        "subject": subject,
        "relation": relation,
        "subject_rank": subject_rank,
    }

    prompt = TEMPLATE_DISTRACTOR.format(
        question=question, answer=", ".join(list(set(all_answers)))
    )

    message_prompt = {
        "role": "user",
        "content": prompt,
    }

    counter = 0
    for k in range(2 * n_d):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[message_prompt],
            max_tokens=max_tokens,
            temperature=min(temperature + 0.1 * k, 1.8),
            top_p=top_p,
        )

        new_answer = (
            completion.choices[0]
            .message.content.split("\n\n")[0]
            .split("Distractor:")[-1]
            .strip()
        )

        if fuzzy_exact_match(new_answer, all_answers) > 0.8:
            continue
        else:
            counter += 1
            all_answers.append(new_answer)
            mc_qa["multi_choice_answers"].append([[new_answer], -1])
        if counter >= n_d:
            break

    out_queue.put((mc_qa, counter))


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--max_threads", type=int, default=64)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n_batch", type=int, default=10000)
    parser.add_argument("--outdir", type=str, default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument("--predefined_q_templates", action="store_true", default=False)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--n_d", type=int, default=3)
    parser.add_argument("--limit_past_time", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--subdict_path",
        type=str,
        default=DATA_PATH / "filtered_subject_dict.json",
    )
    parser.add_argument("--filter_subdict", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    if not Path(args.subdict_path).exists():
        dataset = []
        time_start = time.time()
        with Path.open(args.data_path, "r") as f:
            for i, line in enumerate(f):
                dataset.append(json.loads(line))
        print(
            f"Loaded dataset with {len(dataset)} samples in {time.time() - time_start:.2f} seconds."
        )
        refresh_time = time.time()
        subject_dict = create_subject_dict(
            dataset, limit_past_time=args.limit_past_time
        )
        print(
            f"Selected {len(subject_dict)} samples in {time.time() - refresh_time:.2f} seconds."
        )

        new_subject_dict: dict[str, dict] = {}
        for subject in subject_dict:
            new_subject_dict[subject] = {}
            for relation in subject_dict[subject]:
                new_subject_dict[subject][relation] = subject_dict[subject][relation]
            if len(new_subject_dict[subject]) == 0:
                new_subject_dict.pop(subject)
        subject_dict = new_subject_dict
        with Path.open(
            args.subdict_path,
            "w",
        ) as f:
            json.dump(subject_dict, f)
    else:
        refresh_time = time.time()
        with Path.open(args.subdict_path, "r") as f:
            subject_dict = json.load(f)

    if args.filter_subdict:
        refresh_time = time.time()
        subject_dict = filter_subject_dict(subject_dict)

    print(
        f"Created/Loaded subject dictionary in {time.time() - refresh_time:.2f} seconds."
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    random.seed(args.seed)

    mc_qa_dataset = []

    q: queue.Queue = queue.Queue()

    batched_pairs = []
    batch = []
    for subject, relations in subject_dict.items():
        for relation in relations:
            if relation == "subject_rank":
                continue
            else:
                batch.append((subject, relation))
                if len(batch) >= args.max_threads:
                    batched_pairs.append(batch)
                    batch = []

    if len(batch) > 0:
        batched_pairs.append(batch)

    for i, batch in enumerate(batched_pairs):
        if i == args.n_batch:
            break
        threads = []
        for subject, relation in batch:
            thread = threading.Thread(
                target=create_mcqa,
                kwargs={
                    "subject": subject,
                    "relation": relation,
                    "client": client,
                    "subject_dict": subject_dict,
                    "model_name": args.model_name,
                    "n_d": args.n_d,
                    "limit_past_time": args.limit_past_time,
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "out_queue": q,
                    "predefined_q_templates": args.predefined_q_templates,
                },
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        while not q.empty():
            mc_qa, counter = q.get()
            mc_qa_dataset.append(mc_qa)

    filtered_mcqa = []
    for sample in mc_qa_dataset:
        new_mc_answers = []
        for ans, y in sample["multi_choice_answers"]:
            if not isinstance(y, int):
                continue
            if y not in [-1] + list(range(2014, 2026)):
                continue
            new_mc_answers.append((ans, y))
        sample["multi_choice_answers"] = new_mc_answers
        filtered_mcqa.append(sample)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    output_path = Path(
        args.outdir,
        f"Temporal_eval_dataset_{len(filtered_mcqa)}_samples_seed_{args.seed}_{'genq' if not args.predefined_q_templates else 'predefined'}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
    )
    with Path.open(output_path, "w") as f:
        for item in filtered_mcqa:
            f.write(json.dumps(item) + "\n")

    print(f"Saved dataset to {output_path}")
