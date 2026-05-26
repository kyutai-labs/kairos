import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
from dotenv import load_dotenv
from jaxtyping import PyTree
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kairos.evaluation import (
    OLMES_DATA,
    OlmesArgs,
    Task,
    eval_cf,
    eval_hf_cf,
    eval_hf_mcf,
    eval_hf_taqa,
    eval_mcf,
    eval_taqa,
    eval_task_hf_olmes,
    eval_task_olmes,
    f1_score,
    format_cf_prompt,
    get_cf_context_and_continuation,
    tasks,
)
from kairos.inference.generate import Generator, batching
from kairos.nn import load_model
from kairos.utils import ArgsParser, create_logger

load_dotenv()

DATA_PATH = Path(os.getenv("DATA_DIR", "./data"))


def eval_task_multi_choice(
    model: Any,
    tokenizer: Any,
    task: Task,
    datapath: str,
    logger,
    k_shot: int = 5,
    params: PyTree | None = None,
    target_year: int | None = None,
    use_prefix: bool = False,
    n_choices: int = 4,
    n_distractor: int = 1,
    verbose_log: bool = False,
) -> tuple[dict[str, float], list]:
    trainset = list(
        task.dataloader(
            datapath,
            target_year,
            n_choices,
            n_distractor,
            k_shot,
        )
    )
    evalset = list(
        task.dataloader(
            datapath,
            target_year,
            n_choices,
            n_distractor,
            -k_shot,
        )
    )
    trainset = trainset[:k_shot]
    if params:
        return eval_mcf(
            evalset,
            task.name,
            model,
            params,
            tokenizer,
            logger,
            few_shot_examples=trainset,
            prefix=task.prefix if use_prefix else None,
            verbose_log=verbose_log,
        )
    else:
        return eval_hf_mcf(
            evalset,
            task.name,
            model,
            tokenizer,
            logger,
            few_shot_examples=trainset,
            prefix=task.prefix if use_prefix else None,
            verbose_log=verbose_log,
        )


def eval_task_cloze(
    model: Any,
    tokenizer: Any,
    task: Task,
    datapath: str,
    logger,
    params: PyTree | None = None,
    normalization: str = "",
    k_shot: int = 0,
    target_year: int | None = None,
    n_choices: int = 4,
    n_distractor: int = 1,
    use_prefix: bool = False,
    verbose_log: bool = False,
) -> tuple[dict[str, float], list]:
    trainset = list(
        task.dataloader(
            datapath,
            target_year,
            n_choices,
            n_distractor,
            k_shot,
        )
    )
    trainset = trainset[:k_shot]
    evalset = list(
        task.dataloader(
            datapath,
            target_year,
            n_choices,
            n_distractor,
            -k_shot,
        )
    )
    if params:
        return eval_cf(
            evalset,
            task.name,
            normalization,
            model,
            params,
            tokenizer,
            logger,
            trainset,
            prefix=task.prefix if use_prefix else None,
            verbose_log=verbose_log,
        )
    else:
        return eval_hf_cf(
            evalset,
            task.name,
            normalization,
            model,
            tokenizer,
            logger,
            trainset,
            prefix=task.prefix if use_prefix else None,
            verbose_log=verbose_log,
        )


class Args(ArgsParser):
    model: str | None = None
    step: int | None = None
    subfolder: str | None = ""
    tasks: str | None = None
    data: Path | None = DATA_PATH / "KairosQA.jsonl"
    k: int = 0
    cloze: bool = False
    generate_task: bool = False

    # For one targeted year
    n_choices: int = 4
    n_distractor: int = 0  # Will be added if not enough to reach n_choices anyway but if n_distractor>0 impose to have at least n_distractor
    verbose_log: bool = False
    verbose_path: str = "./verbose_logs/"
    use_prefix: bool = False

    # TAQA params
    insensitive: bool | str | list[bool] = (
        False  # Whether to use time-insensitive examples for ICL; parsed by format()
    )
    time_strat: str | list[str] = (
        ""  # Time to precise in the question if None, neutral prompt; parsed by format()
    )

    # CSV logging
    results_dir: str = os.getenv("RESULTS_DIR", "./results")
    run_name: str | None = None
    log_file: str | None = None  # File logging

    def format(self):
        if isinstance(self.insensitive, bool):
            self.insensitive = [self.insensitive]
        elif isinstance(self.insensitive, str):
            self.insensitive = [
                s.lower() == "true" for s in self.insensitive.split(",")
            ]
        if isinstance(self.time_strat, str):
            self.time_strat = self.time_strat.split(",")


if __name__ == "__main__":
    args = Args.parse_args()
    args.format()

    logger = create_logger(
        name=__name__,
        rank_zero_only=True,
        log_file=args.log_file,
        results_dir=args.results_dir,
        run_name=args.run_name,
    )
    logger.log_config(vars(args))
    model: Any = None
    params: Any = None
    tokenizer: Any = None
    if args.step:
        # Orbax checkpoints
        model, params, tokenizer = load_model(Path(args.model), args.step, None)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, subfolder=args.subfolder
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        params = None

    model_name = args.model.split("/")[-1]

    all_metrics = {}
    for task in args.tasks.split(","):
        if "temporal" in task:
            temporal_task = tasks["temporal"]
            target_year = int(task.split("_")[-1])

            datapath = args.data
            if args.cloze:
                assert not args.generate_task, (
                    "cannot use both cloze and generate_task options"
                )
                for normalization in [
                    "character",
                    "none",
                ]:
                    logger.info(
                        f"Evaluating cloze task with normalization: {normalization}"
                    )

                    metrics, verbose_list = eval_task_cloze(
                        model,
                        tokenizer,
                        temporal_task,
                        datapath,
                        logger,
                        params=params,
                        normalization=normalization,
                        k_shot=args.k,
                        target_year=target_year,
                        n_choices=args.n_choices,
                        n_distractor=args.n_distractor,
                        use_prefix=args.use_prefix,
                        verbose_log=args.verbose_log,
                    )

                    logger.info(
                        f"{task.split('_')[0]} in {target_year}: {metrics['accuracy']}"
                    )
                    logger.log_metrics(
                        {
                            f"cloze_{normalization}_acc": metrics["accuracy"],
                            f"random_baseline_cloze_{normalization}": metrics[
                                "random_baseline"
                            ],
                            "number_of_samples": metrics["n_samples"],
                        },
                        step=target_year,
                    )
                    if args.verbose_log:
                        path_log = (
                            Path(args.verbose_path)
                            / "cloze"
                            / f"{model_name}_temporal_cloze_{normalization}_{target_year}.jsonl"
                        )
                        path_log.parent.mkdir(parents=True, exist_ok=True)
                        with path_log.open("w") as f:
                            for entry in verbose_list:
                                f.write(json.dumps(entry) + "\n")
            elif args.generate_task:
                trainset = list(
                    temporal_task.dataloader(
                        datapath,
                        target_year,
                        args.n_choices,
                        args.n_distractor,
                        args.k,
                    )
                )
                evalset = list(
                    temporal_task.dataloader(
                        datapath,
                        target_year,
                        args.n_choices,
                        args.n_distractor,
                        -args.k,
                    )
                )
                few_shot_prefix = (
                    temporal_task.prefix + "\n\n" if args.use_prefix else ""
                )

                if trainset:
                    few_shot_prompts = []
                    for ex in trainset:
                        choice_idx = ex["gold"] if "gold" in ex else ex["answer"]
                        prompt = format_cf_prompt(ex, temporal_task.name, choice_idx)
                        few_shot_prompts.append(prompt)
                    few_shot_prefix = "\n\n".join(few_shot_prompts) + "\n\n"

                if params:
                    key = jax.random.PRNGKey(1234)
                    generator = Generator(model, params, tokenizer, topn=1, temp=0.8)
                overall_answers = []
                overall_generations = []
                sub_ranks_and_relations = []
                for batch in tqdm(batching(evalset)):
                    prompts = []
                    gold_answers = []
                    generations = []
                    for example in batch:
                        choice_idx = (
                            example["gold"] if "gold" in example else example["answer"]
                        )
                        context, continuation = get_cf_context_and_continuation(
                            example, "", choice_idx
                        )
                        prompt = few_shot_prefix + context
                        prompts.append(prompt)
                        gold_answers.append(example["gt"])

                    if params:
                        key, subkey = jax.random.split(key)
                        generations = generator.generate(prompts, length=64, key=subkey)
                        generations = [
                            g[len(p) :].split("\n\n")[0].strip()
                            for g, p in zip(generations, prompts)
                        ]
                    else:
                        tokenized_prompts = tokenizer(
                            prompts, return_tensors="pt", padding=True
                        )
                        input_len = tokenized_prompts["input_ids"].shape[1]
                        outputs = model.generate(
                            input_ids=tokenized_prompts["input_ids"].cuda(),
                            attention_mask=tokenized_prompts["attention_mask"].cuda(),
                            max_new_tokens=64,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                        )
                        gen_tokens = outputs[:, input_len:]
                        gen_texts = tokenizer.batch_decode(
                            gen_tokens, skip_special_tokens=True
                        )
                        generations = [
                            g.split("\n\n")[0].split("Question:")[0].strip()
                            for g in gen_texts
                        ]
                    overall_answers.extend(gold_answers)
                    overall_generations.extend(generations)
                    if args.verbose_log:
                        for ex, gen in zip(batch, generations):
                            relation = ex.get("relation", "N/A")

                            sub_ranks_and_relations.append(
                                {
                                    "subject_rank": ex["subject_rank"],
                                    "relation": relation,
                                    "generated": gen,
                                    "gold_answer": ex["gt"],
                                    "f1_score": f1_score(gen, ex["gt"]),
                                }
                            )

                f1_score_value = sum(
                    [
                        f1_score(gen, ans)
                        for gen, ans in zip(overall_generations, overall_answers)
                    ]
                ) / len(overall_answers)
                print("F1 Score value", f1_score_value)

                logger.info(
                    f"Generative task on target year {target_year} with {len(evalset)} samples; F1-score: {f1_score_value:.1f}"
                )
                print("Before logging metrics")
                logger.log_metrics(
                    {
                        "generative_task_f1_score": f1_score_value,
                        "number_of_samples": len(evalset),
                    }
                )
                print("After logging metrics")
                if args.verbose_log:
                    path_log = (
                        Path(args.verbose_path)
                        / "gen"
                        / f"{model_name}_temporal_gen_{target_year}.jsonl"
                    )
                    path_log.parent.mkdir(parents=True, exist_ok=True)
                    with path_log.open("w") as f:
                        for entry in sub_ranks_and_relations:
                            f.write(json.dumps(entry) + "\n")

            else:
                metrics, verbose_list = eval_task_multi_choice(
                    model,
                    tokenizer,
                    temporal_task,
                    datapath,
                    logger,
                    args.k,
                    params=params,
                    target_year=target_year,
                    use_prefix=args.use_prefix,
                    n_choices=args.n_choices,
                    n_distractor=args.n_distractor,
                    verbose_log=args.verbose_log,
                )

                logger.info(
                    f"{task.split('_')[0]} in {task.split('_')[-1]} {metrics['accuracy']:.1f}"
                )

                logger.log_metrics(
                    {
                        "multi_choice_acc": metrics["accuracy"],
                        "random_baseline_multi_choice_acc": metrics["random_baseline"],
                        "number_of_samples": metrics["n_samples"],
                    },
                    step=target_year,
                )
                if args.verbose_log:
                    path_log = (
                        Path(args.verbose_path)
                        / "mc"
                        / f"{model_name}_temporal_mc_{target_year}.jsonl"
                    )
                    path_log.parent.mkdir(parents=True, exist_ok=True)
                    with path_log.open("w") as f:
                        for entry in verbose_list:
                            f.write(json.dumps(entry) + "\n")
        elif task == "taqa":
            if params:
                task_metrics = eval_taqa(args, model, params, tokenizer, logger)
            else:
                task_metrics = eval_hf_taqa(args, model, tokenizer, logger)

            logger.log_metrics(task_metrics)
        else:
            olmes_args = OlmesArgs(
                step=args.step,
                data=Path(OLMES_DATA),
                tasks="arc_challenge,arc_easy,boolq,csqa,hellaswag,mmlu,obqa,piqa,siqa,winogrande",
                k_shot=5,
                results_dir=args.results_dir,
                run_name=args.run_name,
            )
            for task in olmes_args.tasks.split(","):
                if params:
                    task_metrics = eval_task_olmes(
                        task, olmes_args, model, params, tokenizer, logger
                    )
                else:
                    task_metrics = eval_task_hf_olmes(
                        task, olmes_args, model, tokenizer, logger
                    )

                all_metrics.update(task_metrics)
            logger.log_metrics(all_metrics)

    if len(all_metrics) > 0:
        avg_accuracy = np.mean(
            [
                all_metrics[f"{task}_accuracy"]
                for task in tasks
                if f"{task}_accuracy" in all_metrics
            ]
        )
        logger.log_metrics({"olmes_score": avg_accuracy})
    logger.info("Evaluation complete!")
    logger.end_run()
