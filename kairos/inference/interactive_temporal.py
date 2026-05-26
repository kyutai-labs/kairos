import os
import random
from pathlib import Path
from timeit import default_timer as timer

import jax
from dotenv import load_dotenv
from pydantic_settings import CliApp
from transformers import AutoModelForCausalLM, AutoTokenizer

from kairos.evaluation.evaluate_tasks import load_temporal
from kairos.inference.generate import Generator, color
from kairos.utils import ArgsParser

load_dotenv()  # Loads variables from .env
DATA_PATH = Path(os.getenv("DATA_DIR", "./data"))

DEFAULT_JSONL = DATA_PATH / "KairosQA.jsonl"
SYSTEM_PREFIX = "The following questions have been answered."


class InteractiveArgs(ArgsParser):
    model: str
    step: int = 0
    data: Path = Path(DEFAULT_JSONL)
    k_shot: int = 5
    start_year: int = 2018
    end_year: int = 2024
    n_choices: int = 5
    n_distractor: int = 1


def load_samples_for_year(
    data_path: Path,
    year: int,
    n_choices: int = 5,
    n_distractor: int = 1,
) -> list[dict]:
    return list(
        load_temporal(
            path=data_path,
            target_year=year,
            n_choices=n_choices,
            n_distractor=n_distractor,
            k_shot=-1,
        )
    )


def get_available_relations(samples: list[dict]) -> list[str]:
    relations = set()
    for sample in samples:
        if sample.get("relation"):
            relations.add(sample["relation"])
    return sorted(relations)


def build_few_shot_prefix(
    samples: list[dict], k_shot: int, exclude_idx: int | None = None
) -> str:
    if k_shot == 0:
        return SYSTEM_PREFIX + "\n\n"

    exclude = {exclude_idx} if exclude_idx is not None else set()
    available = [i for i in range(len(samples)) if i not in exclude]

    if len(available) < k_shot:
        selected = available
    else:
        selected = random.sample(available, k_shot)

    few_shot_prompts = []
    for idx in selected:
        sample = samples[idx]
        question = sample["question"]
        answer = sample["choices"][sample["gold"]]
        few_shot_prompts.append(f"Question: {question}\nAnswer: {answer}")

    return SYSTEM_PREFIX + "\n\n" + "\n\n".join(few_shot_prompts) + "\n\n"


def interactive_temporal(
    samples: list[dict],
    year: int,
    k_shot: int = 5,
    key=None,
    generator: Generator | None = None,
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    relation: str | None = None,
) -> dict | None:
    if relation is not None:
        filtered = [
            (i, s)
            for i, s in enumerate(samples)
            if s.get("relation", "").lower() == relation.lower()
        ]
        if not filtered:
            print(
                f"{color.RED}No valid sample found for relation '{relation}' "
                f"in year {year}{color.END}"
            )
            return None
        sample_idx, sample = random.choice(filtered)
    else:
        if not samples:
            print(f"{color.RED}No valid sample found for year {year}{color.END}")
            return None
        sample_idx = random.randint(0, len(samples) - 1)
        sample = samples[sample_idx]

    few_shot_prefix = build_few_shot_prefix(samples, k_shot, exclude_idx=sample_idx)
    question = sample["question"]
    correct_answer = sample["choices"][sample["gold"]]
    prompt = few_shot_prefix + f"Question: {question}\nAnswer:"

    print(f"\n{color.BOLD}{color.CYAN}═══ TEMPORAL EVAL ═══{color.END}")
    if sample.get("relation"):
        print(f"{color.BOLD}Relation:{color.END} {sample['relation']}")
    if sample.get("subject_rank"):
        print(f"{color.BOLD}Subject rank:{color.END} {sample['subject_rank']}")
    print(f"{color.BOLD}Year:{color.END} {year}")
    print(f"\n{color.BOLD}{color.YELLOW}Question:{color.END} {question}")
    print(f"\n{color.BOLD}{color.GREEN}Full prompt ({k_shot}-shot):{color.END}")
    print("─" * 60)
    print(prompt)
    print("─" * 60)

    if key is None:
        key = jax.random.PRNGKey(1234)

    t0 = timer()

    if generator:
        generation = generator.generate([prompt], length=128, key=key)[0]

        dt = timer() - t0

        generation_text = (
            generation[len(prompt) :].split("\n\n")[0].split("\n")[0].strip()
        )
    else:
        tokenized_prompts = tokenizer([prompt], return_tensors="pt", padding=True)
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"].cuda(),
            attention_mask=tokenized_prompts["attention_mask"].cuda(),
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generation_text = outputs[0][len(prompt) :].split("\n\n")[0].strip()

    print(f"\n{color.BOLD}{color.RED}Generation:{color.END} {generation_text}")
    print(f"{color.BOLD}{color.GREEN}Expected:{color.END} {correct_answer}")
    print(f"{color.BOLD}Time:{color.END} {dt:.2f}s")

    print(f"\n{color.BOLD}{color.PURPLE}All choices:{color.END}")
    for i, (choice, choice_year) in enumerate(
        zip(sample["choices"], sample["choice_years"])
    ):
        marker = " ◀ (correct)" if i == sample["gold"] else ""
        year_str = "distractor" if choice_year == -1 else str(choice_year)
        print(f"  {i + 1}. [{year_str}] {choice}{marker}")

    return {
        "sample": sample,
        "year": year,
        "prompt": prompt,
        "generation": generation_text,
        "expected": correct_answer,
        "choices": sample["choices"],
    }


def loop(
    generator: Generator | None = None,
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    data_path: Path = DEFAULT_JSONL,
    k_shot: int = 5,
    start_year: int = 2018,
    end_year: int = 2024,
    n_choices: int = 5,
    n_distractor: int = 1,
):
    key = jax.random.PRNGKey(42)

    if generator is None:
        assert not model is None and not tokenizer is None, (
            "Either use local orbax checkpoints or HF models"
        )
    current_relation = None
    samples_cache: dict[int, list[dict]] = {}

    def get_samples_for_year(year: int) -> list[dict]:
        if year not in samples_cache:
            print(f"Loading samples for year {year}...")
            samples_cache[year] = load_samples_for_year(
                data_path, year, n_choices, n_distractor
            )
            print(f"Loaded {len(samples_cache[year])} samples for year {year}")
        return samples_cache[year]

    available_relations = get_available_relations(get_samples_for_year(start_year))

    print(f"{color.BOLD}Interactive Temporal Eval{color.END}")
    print(f"Year range: {start_year}-{end_year}, {k_shot}-shot prompting")
    print("Commands:")
    print("  [Enter]        - new sample (random year)")
    print("  [year]         - specific year (e.g., 2020)")
    print("  r [relation]   - set relation filter (e.g., 'r member of sports team')")
    print("  r              - clear relation filter")
    print("  list           - list available relations")
    print("  q              - quit\n")

    while True:
        filter_str = (
            f" [{color.PURPLE}{current_relation}{color.END}]"
            if current_relation
            else ""
        )
        try:
            user_input = input(
                f"{color.BOLD}{color.DARKCYAN}>{filter_str} {color.END}"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "q":
            break

        if user_input.lower() == "list":
            print(f"\n{color.BOLD}Available relations:{color.END}")
            for rel in available_relations:
                print(f"  - {rel}")
            print()
            continue

        if user_input.lower().startswith("r "):
            relation = user_input[2:].strip()
            if relation.lower() in [r.lower() for r in available_relations]:
                current_relation = relation
                print(f"Relation filter set to: {color.PURPLE}{relation}{color.END}")
            else:
                print(f"{color.RED}Unknown relation: '{relation}'{color.END}")
                print("Use 'list' to see available relations.")
            continue

        if user_input.lower() == "r":
            current_relation = None
            print("Relation filter cleared.")
            continue

        if user_input.isdigit():
            year = int(user_input)
            if year < start_year or year > end_year:
                print(
                    f"{color.RED}Year must be between {start_year} and {end_year}{color.END}"
                )
                continue
        else:
            year = random.randint(start_year, end_year)

        samples = get_samples_for_year(year)

        key, subkey = jax.random.split(key)
        interactive_temporal(
            generator=generator,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            year=year,
            k_shot=k_shot,
            key=subkey,
            relation=current_relation,
        )
        print()


def main():
    from kairos.nn import load_model
    from kairos.utils import cast_bf16

    args = CliApp.run(InteractiveArgs)

    print(f"Loading model from {args.model}...")

    if args.step == 0:
        if len(args.model.split(",")) > 1:
            model, subfolder = args.model.split(",")
        else:
            model, subfolder = args.model, ""
        generator = None
        model = AutoModelForCausalLM.from_pretrained(model, subfolder=subfolder).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model, params, tokenizer = load_model(Path(args.model), args.step, False)
        params = cast_bf16(params)

        generator = Generator(model, params, tokenizer, topn=1, temp=0.8)
        model = None
        tokenizer = None

    print(f"Using temporal dataset from {args.data}")

    loop(
        generator,
        model,
        tokenizer,
        data_path=args.data,
        k_shot=args.k_shot,
        start_year=args.start_year,
        end_year=args.end_year,
        n_choices=args.n_choices,
        n_distractor=args.n_distractor,
    )


if __name__ == "__main__":
    main()
