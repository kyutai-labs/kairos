import json
import os
import random
from collections import defaultdict
from pathlib import Path

import jax
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from kairos.evaluation.metrics import f1_score
from kairos.inference.generate import Generator, batching

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_DIR", "./data"))
timeaware_prompting_template = "Answer the following question: {question}\nAs of year {time}, the answer is: {answer}"
timenotaware_prompting_template = (
    "Answer the following question: {question}\nThe answer is: {answer}"
)

time_insensitive_exs = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Harry Potter?", "answer": "J.K. Rowling"},
    {"question": "Where did the Titanic sink?", "answer": "Atlantic Ocean"},
    {"question": "What is the gravity of earth?", "answer": "9.807 m/s^2"},
    {
        "question": "Is the speed of light faster than the speed of sound?",
        "answer": "Yes",
    },
]


def dataset_taqa(
    dataset: list,
    icl_dataset: list = [],
    icl_exs: int = 5,
    insensitive: bool = False,
    time_strat: str | None = None,
):
    icl_prompt = ""
    if icl_exs > 0 and not insensitive:
        random.seed(42)
        top_200 = sorted(
            icl_dataset, key=lambda x: x.get("avg_pageview", 0), reverse=True
        )[:200]
        selected_exs = random.sample(top_200, min(icl_exs, len(top_200)))
        for ex in selected_exs:
            question = ex["question"]
            answer = ex["answer"]
            if time_strat == "":
                icl_prompt += (
                    timenotaware_prompting_template.format(
                        question=question,
                        answer=answer[random.choice(list(answer.keys()))][0],
                    )
                    + "\n\n"
                )
            else:
                icl_prompt += (
                    timeaware_prompting_template.format(
                        question=question, time=time_strat, answer=answer[time_strat][0]
                    )
                    + "\n\n"
                )
    elif insensitive:
        selected_exs = time_insensitive_exs[: min(icl_exs, len(time_insensitive_exs))]
        for ex in selected_exs:
            question = ex["question"]
            answer = ex["answer"]
            if time_strat == "":
                icl_prompt += (
                    timenotaware_prompting_template.format(
                        question=question, answer=answer
                    )
                    + "\n\n"
                )
            else:
                icl_prompt += (
                    timeaware_prompting_template.format(
                        question=question, time=time_strat, answer=answer
                    )
                    + "\n\n"
                )

    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        if time_strat == "":
            prompt = timenotaware_prompting_template.format(
                question=question, answer=""
            )
        else:
            prompt = timeaware_prompting_template.format(
                question=question, time=time_strat, answer=""
            )
        yield icl_prompt + prompt, answer
    return None


def get_taqa_dataloader(
    dataset: list | None = None,
    icl_exs: int = 5,
    insensitive: bool = False,
    time_strat: str = "",
    taqa_path: Path = DATA_PATH / "taqa_test.jsonl",
    taqa_train_path: Path = DATA_PATH / "taqa_train.jsonl",
):
    if dataset is None:
        dataset = []
        with Path(taqa_path).open("r") as f:
            for line in f:
                dataset.append(json.loads(line))
        print(f"Loaded TAQA test dataset with {len(dataset)} examples.")
    icl_dataset = []
    if icl_exs > 0 and not insensitive:
        with Path(taqa_train_path).open("r") as f:
            for line in f:
                icl_dataset.append(json.loads(line))
        print(f"Loaded TAQA train dataset with {len(icl_dataset)} examples for ICL.")
    return (
        dataset_taqa(
            dataset,
            icl_dataset=icl_dataset,
            icl_exs=icl_exs,
            insensitive=insensitive,
            time_strat=time_strat,
        ),
        dataset,
    )


def eval_taqa(args, model, params, tokenizer, logger):
    key = jax.random.PRNGKey(1234)
    generator = Generator(model, params, tokenizer, topn=1, temp=0.7)

    dataset = None
    overall_metrics: defaultdict[str, float] = defaultdict(lambda: 0.0)
    verbose_list = []
    for insensitive in args.insensitive:
        for time_strat in args.time_strat:
            metrics: defaultdict[str, float] = defaultdict(lambda: 0.0)
            counter: defaultdict[str, int] = defaultdict(lambda: 0)
            logger.info(
                f"TAQA settings - insensitive: {insensitive}, time_strat: {time_strat}"
            )
            taqa_dataloader, dataset = get_taqa_dataloader(
                dataset=dataset,
                icl_exs=args.k,
                insensitive=insensitive,
                time_strat=time_strat,
            )

            overall_answers = []
            overall_generations = []
            logger.info("Starting TAQA evaluation...")
            for batch in batching(taqa_dataloader):
                prompts = [p for p, _ in batch]
                answers = [a for _, a in batch]
                key, subkey = jax.random.split(key)

                generations = generator.generate(prompts, length=64, key=subkey)
                generations = [
                    g[len(p) :]
                    .split("\n\n")[0]
                    .replace("The answer is: ", "")
                    .replace(", the answer is: ", "")
                    .strip()
                    for g, p in zip(generations, prompts)
                ]

                overall_answers.extend(answers)
                overall_generations.extend(generations)
                if args.verbose_log:
                    for ans, gen, p in zip(answers, generations, prompts):
                        verbose_list.append(
                            {
                                "prompt": p,
                                "generated_answer": gen,
                                "gold_answers": ans,
                                "insensitive": insensitive,
                                "time_strat": time_strat,
                                "f1_score": {
                                    y: f1_score(gen, a) for y, a in ans.items()
                                },
                            }
                        )

            logger.info("Computing TAQA metrics...")
            for generated_text, answer in zip(overall_generations, overall_answers):
                if time_strat == "":
                    for year, ans in answer.items():
                        metrics[f"{insensitive}_ts_none_{year}"] += f1_score(
                            generated_text, ans
                        )
                        counter[f"{insensitive}_ts_none_{year}"] += 1
                else:
                    if time_strat in answer:
                        ans = answer[time_strat]
                        metrics[f"{insensitive}_ts_{time_strat}"] += f1_score(
                            generated_text, ans
                        )
                        counter[f"{insensitive}_ts_{time_strat}"] += 1
            for metric_name in metrics.keys():
                if counter[metric_name] > 0:
                    metrics[metric_name] /= counter[metric_name]
                logger.info(f"TAQA Metric {metric_name}: {metrics[metric_name]:.4f}")
                overall_metrics[metric_name] += metrics[metric_name]

            if args.verbose_log:
                path_log = (
                    Path(args.verbose_path)
                    / "taqa"
                    / f"{args.model.split('/')[-1]}_taqa.jsonl"
                )
                with Path(path_log).open("a") as f:
                    for entry in verbose_list:
                        f.write(json.dumps(entry) + "\n")
                verbose_list = []

    if args.verbose_log and len(verbose_list) > 0:
        path_log = (
            Path(args.verbose_path) / "taqa" / f"{args.model.split('/')[-1]}_taqa.jsonl"
        )
        with Path(path_log).open("a") as f:
            for entry in verbose_list:
                f.write(json.dumps(entry) + "\n")
        verbose_list = []

    return overall_metrics


def eval_hf_taqa(args, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, logger):
    dataset = None
    overall_metrics: defaultdict[str, float] = defaultdict(lambda: 0.0)
    verbose_list = []
    for insensitive in args.insensitive:
        for time_strat in args.time_strat:
            metrics: defaultdict[str, float] = defaultdict(lambda: 0.0)
            counter: defaultdict[str, int] = defaultdict(lambda: 0)
            logger.info(
                f"TAQA settings - insensitive: {insensitive}, time_strat: {time_strat}"
            )
            taqa_dataloader, dataset = get_taqa_dataloader(
                dataset=dataset,
                icl_exs=args.k,
                insensitive=insensitive,
                time_strat=time_strat,
            )

            overall_answers = []
            overall_generations = []
            logger.info("Starting TAQA evaluation...")
            for batch in batching(taqa_dataloader):
                prompts = [p for p, _ in batch]
                answers = [a for _, a in batch]
                prompted_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
                input_len = prompted_tokens["input_ids"].shape[1]
                outputs = model.generate(
                    input_ids=prompted_tokens["input_ids"].cuda(),
                    attention_mask=prompted_tokens["attention_mask"].cuda(),
                    max_new_tokens=64,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                # Token-level slice: decoded prompt does not round-trip ("\n"
                # is dropped), so g[len(p):] cuts into the actual generation.
                gen_tokens = outputs[:, input_len:]
                gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                # The model greedy-decodes past the answer into further ICL-style
                # "Answer the following ..." continuations; keep only the first
                # answer line.
                generations = [
                    g.split("Answer the following")[0].split("\n")[0].strip()
                    for g in gen_texts
                ]

                overall_answers.extend(answers)
                overall_generations.extend(generations)

                if args.verbose_log:
                    for ans, gen, p in zip(answers, generations, prompts):
                        verbose_list.append(
                            {
                                "prompt": p,
                                "generated_answer": gen,
                                "gold_answers": ans,
                                "insensitive": insensitive,
                                "time_strat": time_strat,
                                "f1_score": {
                                    y: f1_score(gen, a) for y, a in ans.items()
                                },
                            }
                        )

            logger.info("Computing TAQA metrics...")
            for generated_text, answer in zip(overall_generations, overall_answers):
                for year, ans in answer.items():
                    metrics[f"{insensitive}_ts_{str(time_strat)}_" + str(year)] += (
                        f1_score(generated_text, ans)
                    )
                    counter[f"{insensitive}_ts_{str(time_strat)}_" + str(year)] += 1
            for metric_name in metrics.keys():
                if counter[metric_name] > 0:
                    metrics[metric_name] /= counter[metric_name]
                logger.info(f"TAQA Metric {metric_name}: {metrics[metric_name]:.4f}")
                overall_metrics[metric_name] += metrics[metric_name]

            if args.verbose_log:
                path_log = (
                    Path(args.verbose_path)
                    / "taqa"
                    / f"{args.model.split('/')[-1]}_taqa.jsonl"
                )
                with Path(path_log).open("a") as f:
                    for entry in verbose_list:
                        f.write(json.dumps(entry) + "\n")
                verbose_list = []

    if args.verbose_log and len(verbose_list) > 0:
        path_log = (
            Path(args.verbose_path) / "taqa" / f"{args.model.split('/')[-1]}_taqa.jsonl"
        )
        with Path(path_log).open("a") as f:
            for entry in verbose_list:
                f.write(json.dumps(entry) + "\n")
        verbose_list = []

    return overall_metrics
