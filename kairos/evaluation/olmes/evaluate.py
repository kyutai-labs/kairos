import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jaxtyping import PyTree
from sentencepiece import SentencePieceProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from kairos.data.tokenization import tokenize
from kairos.evaluation.olmes.config import OLMES_TASKS, OlmesArgs
from kairos.evaluation.olmes.data import (
    batching,
    format_cf_prompt,
    format_mcf_prompt,
    load_fewshot_examples,
    sample_data,
)
from kairos.nn import load_model
from kairos.utils.logger import MultiLogger, create_logger
from kairos.utils.utils import cast_bf16, pad


def eval_mcf(
    examples: list[dict],
    task_name: str,
    model: nn.Module,
    params: PyTree,
    tokenizer: SentencePieceProcessor,
    logger: MultiLogger,
    few_shot_examples: list[dict] | None = None,
    prefix: str | None = None,
    verbose_log: bool = False,
) -> tuple[dict, list]:
    metrics = {"correct": 0, "total": 0, "random_correct": 0}
    few_shot_prefix = ""
    if prefix:
        few_shot_prefix = prefix + "\n\n"
    if few_shot_examples:
        few_shot_prompts = [
            format_mcf_prompt(ex, task_name, include_answer=True)
            for ex in few_shot_examples
        ]
        few_shot_prefix += "\n\n".join(few_shot_prompts) + "\n\n"
    # Tokenize with leading space because " A" may tokenize differently than "A"
    # depending on the tokenizer. This matches how letters appear in the prompt.
    # We take [-1] to get either the combined " A" token or just the "A" token.
    letter_tokens = [
        tokenizer.encode(f" {letter}")[-1]
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]
    ]

    @jax.jit
    def _get_last_token_logits(params, x, positions):
        logits, _ = model.apply(params, x)
        return logits[np.arange(x.shape[0]), positions]

    verbose_list = []
    for batch in batching(examples, batch_size=16):
        prompts = [
            few_shot_prefix + format_mcf_prompt(ex, task_name, include_answer=False)
            for ex in batch
        ]
        tokens = [tokenize(p, tokenizer) for p in prompts]
        last_token_idx = [len(t) - 1 for t in tokens]
        pad_length = 256 * math.ceil((max(last_token_idx) + 1) / 256)
        tokens_padded = [pad(t, tokenizer.pad_id(), pad_length) for t in tokens]

        x = jnp.asarray(tokens_padded)
        logits = _get_last_token_logits(params, x, last_token_idx)

        for i, example in enumerate(batch):
            num_choices = len(example["choices"])
            valid_letter_tokens = jnp.array(letter_tokens[:num_choices])
            pred = int(jnp.argmax(logits[i][valid_letter_tokens]))
            if pred == example["gold"]:
                metrics["correct"] += 1
            if np.random.randint(0, num_choices) == example["gold"]:
                metrics["random_correct"] += 1
            metrics["total"] += 1

            if verbose_log:
                verbose_list.append(
                    {
                        "question": example.get("question", ""),
                        "subject_rank": example.get("subject_rank", -1),
                        "relation": example.get("relation", ""),
                        "n_choices": num_choices,
                        "gt": example.get("gt", ""),
                        "generated": pred,
                        "gold_answer": example["gold"],
                        "score": int(pred == example["gold"]),
                    }
                )

    accuracy = (
        100.0 * metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    )
    random_accuracy = (
        100.0 * metrics["random_correct"] / metrics["total"]
        if metrics["total"] > 0
        else 0.0
    )
    logger.info(
        f"MCF Accuracy: {accuracy:.2f}%, Random Baseline: {random_accuracy:.2f}%, Samples: {metrics['total']}"
    )

    return {
        "accuracy": accuracy,
        "method": "mcf",
        "random_baseline": random_accuracy,
        "n_samples": metrics["total"],
    }, verbose_list


def eval_hf_mcf(
    examples: list[dict],
    task_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    logger: MultiLogger,
    few_shot_examples: list[dict] | None = None,
    prefix: str | None = None,
    verbose_log: bool = False,
) -> tuple[dict, list]:
    metrics = {"correct": 0, "total": 0, "random_correct": 0}
    few_shot_prefix = ""
    if prefix:
        few_shot_prefix = prefix + "\n\n"
    if few_shot_examples:
        few_shot_prompts = [
            format_mcf_prompt(ex, task_name, include_answer=True)
            for ex in few_shot_examples
        ]
        few_shot_prefix += "\n\n".join(few_shot_prompts) + "\n\n"

    verbose_list = []
    for batch in batching(examples, batch_size=16):
        prompts = [
            few_shot_prefix + format_mcf_prompt(ex, task_name, include_answer=False)
            for ex in batch
        ]
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
        input_len = tokenized_prompts["input_ids"].shape[1]
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"].cuda(),
            attention_mask=tokenized_prompts["attention_mask"].cuda(),
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Slice off the prompt at the token level (robust to tokenizer quirks),
        # then take the first non-whitespace character of what the model generated.
        generated_tokens = outputs[:, input_len:]
        generated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        preds = [t.strip()[:1] for t in generated_text]

        for i, example in enumerate(batch):
            num_choices = len(example["choices"])
            pred = preds[i]
            if pred == "ABCDEFGH"[example["gold"]]:
                metrics["correct"] += 1
            if np.random.randint(0, num_choices) == example["gold"]:
                metrics["random_correct"] += 1
            metrics["total"] += 1
            if verbose_log:
                verbose_list.append(
                    {
                        "question": example.get("question", ""),
                        "subject_rank": example.get("subject_rank", -1),
                        "relation": example.get("relation", ""),
                        "n_choices": num_choices,
                        "gt": example.get("gt", ""),
                        "generated": pred,
                        "gold_answer": "ABCDEFGH"[example["gold"]],
                        "score": int(pred == "ABCDEFGH"[example["gold"]]),
                    }
                )
    accuracy = (
        100.0 * metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    )
    random_accuracy = (
        100.0 * metrics["random_correct"] / metrics["total"]
        if metrics["total"] > 0
        else 0.0
    )
    logger.info(
        f"MCF Accuracy: {accuracy:.2f}%, Random Baseline: {random_accuracy:.2f}%, Samples: {metrics['total']}"
    )

    return {
        "accuracy": accuracy,
        "method": "mcf",
        "random_baseline": random_accuracy,
        "n_samples": metrics["total"],
    }, verbose_list


def get_cf_context_and_continuation(
    example: dict, task_name: str, choice_idx: int
) -> tuple[str, str]:
    question = example["question"]
    choice = example["choices"][choice_idx]

    if task_name == "piqa":
        context = f"Goal: {question}\nAnswer:"
        continuation = f" {choice}"
    elif task_name == "hellaswag":
        context = f"{question}"
        continuation = f" {choice}"
    elif task_name == "winogrande":
        pronoun_loc = question.index("_")
        context = question[:pronoun_loc] + choice
        continuation = " " + question[pronoun_loc + 1 :].strip()
    else:
        context = f"Question: {question}\nAnswer:"
        continuation = f" {choice}"

    return context, continuation


def eval_cf(
    examples: list[dict],
    task_name: str,
    normalization: str,
    model: nn.Module,
    params: PyTree,
    tokenizer: SentencePieceProcessor,
    logger: MultiLogger,
    few_shot_examples: list[dict] | None = None,
    prefix: str | None = None,
    verbose_log: bool = False,
) -> tuple[dict, list]:
    metrics = {"correct": 0, "total": 0, "random_correct": 0}

    few_shot_prefix = ""
    if prefix:
        few_shot_prefix = prefix + "\n\n"
    if few_shot_examples:
        few_shot_prompts = []
        for ex in few_shot_examples:
            choice_idx = ex["gold"] if "gold" in ex else ex["answer"]
            prompt = format_cf_prompt(ex, task_name, choice_idx)
            few_shot_prompts.append(prompt)
        few_shot_prefix = "\n\n".join(few_shot_prompts) + "\n\n"

    @partial(jax.jit, static_argnames=["pad_id"])
    def _get_continuation_scores(params, x, y, pad_id, continuation_start_indices):
        logits, _ = model.apply(params, x)
        labels = jax.nn.one_hot(y, logits.shape[-1])
        ce = optax.softmax_cross_entropy(logits, labels)
        # Mask context + padding
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)
        mask = positions[None, :] >= continuation_start_indices[:, None]
        mask = mask & (y != pad_id)
        ce = jnp.where(mask, ce, jnp.zeros_like(ce))
        return jnp.sum(ce, axis=1)

    verbose_list = []
    for example in examples:
        contexts_and_continuations = [
            get_cf_context_and_continuation(example, task_name, i)
            for i in range(len(example["choices"]))
        ]

        full_sequences = []
        continuation_start_indices = []
        continuation_lengths_chars = []

        for context, continuation in contexts_and_continuations:
            full_text = few_shot_prefix + context + continuation
            context_text = few_shot_prefix + context

            full_tokens = tokenize(full_text, tokenizer)
            full_sequences.append(full_tokens)

            context_tokens = tokenize(context_text, tokenizer)
            continuation_start_indices.append(
                len(context_tokens) - 1
            )  # -1 because y[i] corresponds to prediction for token i+1
            continuation_lengths_chars.append(len(continuation))

        pad_length = 64 * math.ceil(max(len(t) for t in full_sequences) / 64)
        tokens_padded = [pad(t, tokenizer.pad_id(), pad_length) for t in full_sequences]

        x = jnp.asarray(tokens_padded)
        cont_start_idx = jnp.asarray(continuation_start_indices)

        scores = np.asarray(
            _get_continuation_scores(
                params, x[:, :-1], x[:, 1:], tokenizer.pad_id(), cont_start_idx
            )
        )

        unconditional_scores = None
        if normalization == "pmi":
            uncond_sequences = []
            uncond_start_indices = []

            for context, continuation in contexts_and_continuations:
                uncond_text = "Answer:" + continuation
                uncond_context = "Answer:"

                uncond_tokens = tokenize(uncond_text, tokenizer)
                uncond_context_tokens = tokenize(uncond_context, tokenizer)
                uncond_sequences.append(uncond_tokens)
                uncond_start_indices.append(len(uncond_context_tokens) - 1)

            pad_length_u = 64 * math.ceil(max(len(t) for t in uncond_sequences) / 64)
            tokens_u_padded = [
                pad(t, tokenizer.pad_id(), pad_length_u) for t in uncond_sequences
            ]
            x_u = jnp.asarray(tokens_u_padded)
            uncond_start_idx = jnp.asarray(uncond_start_indices)

            unconditional_scores = np.asarray(
                _get_continuation_scores(
                    params,
                    x_u[:, :-1],
                    x_u[:, 1:],
                    tokenizer.pad_id(),
                    uncond_start_idx,
                )
            )

        if normalization == "character":
            lengths = np.array(continuation_lengths_chars)
            normalized_scores = scores / lengths
        elif normalization == "token":
            lengths = np.array(
                [
                    len(full_sequences[i]) - continuation_start_indices[i] - 1
                    for i in range(len(full_sequences))
                ]
            )
            lengths = np.maximum(lengths, 1)
            normalized_scores = scores / lengths
        elif normalization == "pmi":
            normalized_scores = scores - unconditional_scores
        else:
            normalized_scores = scores

        pred = int(np.argmin(normalized_scores))
        if pred == example["gold"]:
            metrics["correct"] += 1
        if np.random.randint(0, len(example["choices"])) == example["gold"]:
            metrics["random_correct"] += 1
        metrics["total"] += 1
        if verbose_log:
            verbose_list.append(
                {
                    "question": example.get("question", ""),
                    "subject_rank": example.get("subject_rank", -1),
                    "relation": example.get("relation", ""),
                    "n_choices": len(example["choices"]),
                    "generated": pred,
                    "gt": example.get("gt", ""),
                    "gold_answer": example["gold"],
                    "score": int(pred == example["gold"]),
                }
            )

    accuracy = (
        100.0 * metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    )
    random_accuracy = (
        100.0 * metrics["random_correct"] / metrics["total"]
        if metrics["total"] > 0
        else 0.0
    )
    logger.info(
        f"CF Accuracy ({normalization}): {accuracy:.2f}%, Random Baseline: {random_accuracy:.2f}%, Samples: {metrics['total']}"
    )

    return {
        "accuracy": accuracy,
        "method": f"cf_{normalization}",
        "random_baseline": random_accuracy,
        "n_samples": metrics["total"],
    }, verbose_list


def eval_hf_cf(
    examples: list[dict],
    task_name: str,
    normalization: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    logger: MultiLogger,
    few_shot_examples: list[dict] | None = None,
    prefix: str | None = None,
    verbose_log: bool = False,
) -> tuple[dict, list]:
    metrics = {"correct": 0, "total": 0, "random_correct": 0}
    few_shot_prefix = ""
    if prefix:
        few_shot_prefix = prefix + "\n\n"
    if few_shot_examples:
        few_shot_prompts = []
        for ex in few_shot_examples:
            choice_idx = ex["gold"] if "gold" in ex else ex["answer"]
            prompt = format_cf_prompt(ex, task_name, choice_idx)
            few_shot_prompts.append(prompt)
        few_shot_prefix = "\n\n".join(few_shot_prompts) + "\n\n"
    verbose_list = []

    def get_scores(sequences, continuation_start_indices):
        tokens = tokenizer(
            sequences, return_tensors="pt", padding=True, padding_side="right"
        )

        input_ids = tokens["input_ids"].cuda()
        attention_mask = tokens["attention_mask"].cuda()

        x, y = input_ids[:, :-1], input_ids[:, 1:]

        with torch.no_grad():
            logits = model(x).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(
                -1
            )

            seq_len = x.shape[1]
            positions = torch.arange(seq_len).unsqueeze(0).to(x.device)

            cont_start = (
                torch.tensor(continuation_start_indices).unsqueeze(1).to(x.device)
            )

            cont_mask = positions >= cont_start

            final_mask = cont_mask & attention_mask[:, 1:].bool()

            masked_log_probs = torch.where(
                final_mask, token_log_probs, torch.zeros_like(token_log_probs)
            )

            total_log_probs = -masked_log_probs.sum(dim=-1)

        return total_log_probs.float().cpu().numpy()

    for example in examples:
        contexts_and_continuations = [
            get_cf_context_and_continuation(example, task_name, i)
            for i in range(len(example["choices"]))
        ]

        full_sequences = []
        full_token_lengths = []
        continuation_lengths_chars = []
        continuation_start_indices = []

        for context, continuation in contexts_and_continuations:
            full_text = few_shot_prefix + context + continuation
            context_text = few_shot_prefix + context
            continuation_start_indices.append(
                len(tokenizer.encode(context_text, padding=False)) - 1
            )

            full_sequences.append(full_text)
            continuation_lengths_chars.append(len(continuation))
            full_token_lengths.append(len(tokenizer.encode(full_text, padding=False)))

        scores = get_scores(full_sequences, continuation_start_indices)

        if normalization == "character":
            lengths = np.array(continuation_lengths_chars)
            normalized_scores = scores / lengths
        elif normalization == "token":
            lengths = np.array(
                [
                    len(full_token_lengths[i]) - continuation_start_indices[i] - 1
                    for i in range(len(full_token_lengths))
                ]
            )
            lengths = np.maximum(lengths, 1)
            normalized_scores = scores / lengths
        elif normalization == "pmi":
            uncond_sequences = []
            uncond_start_indices = []

            for context, continuation in contexts_and_continuations:
                uncond_text = "Answer:" + continuation
                uncond_context = "Answer:"

                uncond_context_tokens = tokenizer.encode(uncond_context, padding=False)
                uncond_sequences.append(uncond_text)
                uncond_start_indices.append(len(uncond_context_tokens) - 1)

            unconditional_scores = get_scores(uncond_sequences, uncond_start_indices)
            normalized_scores = scores - unconditional_scores
        else:
            normalized_scores = scores

        pred = int(np.argmin(normalized_scores))
        if pred == example["gold"]:
            metrics["correct"] += 1
        if np.random.randint(0, len(example["choices"])) == example["gold"]:
            metrics["random_correct"] += 1
        metrics["total"] += 1

        if verbose_log:
            verbose_list.append(
                {
                    "question": example.get("question", ""),
                    "subject_rank": example.get("subject_rank", -1),
                    "relation": example.get("relation", ""),
                    "n_choices": len(example["choices"]),
                    "generated": pred,
                    "gt": example.get("gt", ""),
                    "gold_answer": example["gold"],
                    "score": int(pred == example["gold"]),
                }
            )

    accuracy = (
        100.0 * metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    )
    random_accuracy = (
        100.0 * metrics["random_correct"] / metrics["total"]
        if metrics["total"] > 0
        else 0.0
    )
    logger.info(
        f"CF Accuracy ({normalization}): {accuracy:.2f}%, Random Baseline: {random_accuracy:.2f}%, Samples: {metrics['total']}"
    )
    return {
        "accuracy": accuracy,
        "method": f"cf_{normalization}",
        "random_baseline": random_accuracy,
        "n_samples": metrics["total"],
    }, verbose_list


def eval_mmlu_olmes(
    args: OlmesArgs,
    model: nn.Module,
    params: PyTree,
    tokenizer: SentencePieceProcessor,
    logger: MultiLogger,
) -> dict:
    logger.info("Evaluating MMLU with macro averaging across subjects")

    task_config = OLMES_TASKS["mmlu"]
    datapath = args.data / task_config.filename

    normalization = task_config.normalization

    # Load all data and group by subject as OLMES uses the macro-average
    all_data = list(task_config.dataloader(datapath))
    subjects_data = defaultdict(list)
    for item in all_data:
        subjects_data[item["subject"]].append(item)

    subject_accuracies_mcf = {}
    subject_accuracies_cf = {}

    fewshot_examples = load_fewshot_examples("mmlu", args.k_shot)
    for subject, subject_examples in subjects_data.items():
        logger.info(f"Evaluating subject: {subject} ({len(subject_examples)} examples)")

        subject_formatted = " ".join(subject.split("_"))
        prefix = f"The following are multiple choice questions (with answers) about {subject_formatted}."

        mcf_result, _ = eval_mcf(
            subject_examples,
            "mmlu",
            model,
            params,
            tokenizer,
            logger,
            fewshot_examples[subject],
            prefix,
        )
        subject_accuracies_mcf[subject] = mcf_result["accuracy"]

        cf_result, _ = eval_cf(
            subject_examples,
            "mmlu",
            normalization,
            model,
            params,
            tokenizer,
            logger,
            fewshot_examples[subject],
        )
        subject_accuracies_cf[subject] = cf_result["accuracy"]

    macro_avg_mcf = np.mean(list(subject_accuracies_mcf.values()))
    macro_avg_cf = np.mean(list(subject_accuracies_cf.values()))

    logger.info(f"MMLU MCF Macro Average: {macro_avg_mcf:.2f}%")
    logger.info(f"MMLU CF Macro Average: {macro_avg_cf:.2f}%")

    best_accuracy = max(macro_avg_mcf, macro_avg_cf)
    best_method = "mcf" if macro_avg_mcf >= macro_avg_cf else f"cf_{normalization}"
    logger.info(f"Best result: {best_method} with accuracy {best_accuracy:.2f}%")
    return {
        "mmlu_accuracy": best_accuracy,
        "mmlu_mcf_accuracy": macro_avg_mcf,
        "mmlu_cf_accuracy": macro_avg_cf,
    }


def eval_hf_mmlu_olmes(
    args: OlmesArgs,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    logger: MultiLogger,
) -> dict:
    logger.info("Evaluating MMLU with macro averaging across subjects")

    task_config = OLMES_TASKS["mmlu"]
    datapath = args.data / task_config.filename

    normalization = task_config.normalization

    # Load all data and group by subject as OLMES uses the macro-average
    all_data = list(task_config.dataloader(datapath))
    subjects_data = defaultdict(list)
    for item in all_data:
        subjects_data[item["subject"]].append(item)

    subject_accuracies_mcf = {}
    subject_accuracies_cf = {}

    fewshot_examples = load_fewshot_examples("mmlu", args.k_shot)
    for subject, subject_examples in subjects_data.items():
        logger.info(f"Evaluating subject: {subject} ({len(subject_examples)} examples)")

        subject_formatted = " ".join(subject.split("_"))
        prefix = f"The following are multiple choice questions (with answers) about {subject_formatted}."

        mcf_result, _ = eval_hf_mcf(
            subject_examples,
            "mmlu",
            model,
            tokenizer,
            logger,
            fewshot_examples[subject],
            prefix,
        )
        subject_accuracies_mcf[subject] = mcf_result["accuracy"]

        cf_result, _ = eval_hf_cf(
            subject_examples,
            "mmlu",
            normalization,
            model,
            tokenizer,
            logger,
            fewshot_examples[subject],
        )
        subject_accuracies_cf[subject] = cf_result["accuracy"]

    macro_avg_mcf = np.mean(list(subject_accuracies_mcf.values()))
    macro_avg_cf = np.mean(list(subject_accuracies_cf.values()))

    logger.info(f"MMLU MCF Macro Average: {macro_avg_mcf:.2f}%")
    logger.info(f"MMLU CF Macro Average: {macro_avg_cf:.2f}%")

    best_accuracy = max(macro_avg_mcf, macro_avg_cf)
    best_method = "mcf" if macro_avg_mcf >= macro_avg_cf else f"cf_{normalization}"
    logger.info(f"Best result: {best_method} with accuracy {best_accuracy:.2f}%")
    return {
        "mmlu_accuracy": best_accuracy,
        "mmlu_mcf_accuracy": macro_avg_mcf,
        "mmlu_cf_accuracy": macro_avg_cf,
    }


def eval_task_hf_olmes(
    task_name: str,
    args: OlmesArgs,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    logger: MultiLogger,
) -> dict:
    logger.info(f"Evaluating task: {task_name}")

    task_config = OLMES_TASKS[task_name]

    if task_name == "mmlu":
        return eval_hf_mmlu_olmes(args, model, tokenizer, logger)

    datapath = Path(args.data) / task_config.filename

    test_data = list(task_config.dataloader(datapath))

    # Sample according to OLMES standard
    test_data = sample_data(test_data, task_config.max_samples, args.seed)
    logger.info(f"Loaded {len(test_data)} test examples")

    few_shot_examples = load_fewshot_examples(task_name, args.k_shot)
    if few_shot_examples:
        logger.info(
            f"Loaded {len(few_shot_examples)} few-shot examples for {task_name}"
        )

    normalization = task_config.normalization

    # Evaluate with CF
    cf_result, _ = eval_hf_cf(
        test_data,
        task_name,
        normalization,
        model,
        tokenizer,
        logger,
        few_shot_examples,
    )

    # Evaluate with MCF
    mcf_result, _ = eval_hf_mcf(
        test_data, task_name, model, tokenizer, logger, few_shot_examples
    )

    best_result = (
        mcf_result if mcf_result["accuracy"] >= cf_result["accuracy"] else cf_result
    )
    logger.info(
        f"Best result: {best_result['method']} with accuracy {best_result['accuracy']:.2f}%"
    )
    return {
        f"{task_name}_accuracy": best_result["accuracy"],
        f"{task_name}_mcf_accuracy": mcf_result["accuracy"],
        f"{task_name}_cf_accuracy": cf_result["accuracy"],
    }


def eval_task_olmes(
    task_name: str,
    args: OlmesArgs,
    model: nn.Module,
    params: PyTree,
    tokenizer: SentencePieceProcessor,
    logger: MultiLogger,
) -> dict:
    logger.info(f"Evaluating task: {task_name}")

    task_config = OLMES_TASKS[task_name]

    if task_name == "mmlu":
        return eval_mmlu_olmes(args, model, params, tokenizer, logger)

    datapath = Path(args.data) / task_config.filename

    test_data = list(task_config.dataloader(datapath))

    # Sample according to OLMES standard
    test_data = sample_data(test_data, task_config.max_samples, args.seed)
    logger.info(f"Loaded {len(test_data)} test examples")

    few_shot_examples = load_fewshot_examples(task_name, args.k_shot)
    if few_shot_examples:
        logger.info(
            f"Loaded {len(few_shot_examples)} few-shot examples for {task_name}"
        )

    normalization = task_config.normalization

    # Evaluate with MCF
    mcf_result, _ = eval_mcf(
        test_data, task_name, model, params, tokenizer, logger, few_shot_examples
    )

    # Evaluate with CF
    cf_result, _ = eval_cf(
        test_data,
        task_name,
        normalization,
        model,
        params,
        tokenizer,
        logger,
        few_shot_examples,
    )

    best_result = (
        mcf_result if mcf_result["accuracy"] >= cf_result["accuracy"] else cf_result
    )
    logger.info(
        f"Best result: {best_result['method']} with accuracy {best_result['accuracy']:.2f}%"
    )
    return {
        f"{task_name}_accuracy": best_result["accuracy"],
        f"{task_name}_mcf_accuracy": mcf_result["accuracy"],
        f"{task_name}_cf_accuracy": cf_result["accuracy"],
    }


def main(args: OlmesArgs, logger):
    logger.info("Starting OLMES evaluation")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"k-shot: {args.k_shot}")

    model, params, tokenizer = load_model(args.model, args.step, None)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
    params = cast_bf16(params)

    all_metrics = {}
    tasks = args.tasks.split(",")
    for task_name in tasks:
        if task_name == "mmlu":
            task_metrics = eval_mmlu_olmes(args, model, params, tokenizer, logger)
        else:
            task_metrics = eval_task_olmes(
                task_name, args, model, params, tokenizer, logger
            )
        all_metrics.update(task_metrics)

    logger.log_metrics(all_metrics)

    avg_accuracy = np.mean(
        [
            all_metrics[f"{task}_accuracy"]
            for task in tasks
            if f"{task}_accuracy" in all_metrics
        ]
    )
    logger.log_metrics({"olmes_score": avg_accuracy})


if __name__ == "__main__":
    args = OlmesArgs.parse_args()
    logger = create_logger(
        name=__name__,
        rank_zero_only=True,
        log_file=args.log_file,
        results_dir=args.results_dir,
        run_name=args.run_name,
    )
    main(args, logger)
    logger.info("Evaluation complete!")
    logger.end_run()
