import json
import os
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = Path(os.getenv("DATA_DIR", "./data"))
OLMES_PATH = DATA_PATH / "olmes"

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def dump_jsonl(items, path: Path) -> None:
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def load_arc_challenge():
    dump_jsonl(
        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"),
        OLMES_PATH / "arc_challenge.jsonl",
    )


def load_arc_easy():
    dump_jsonl(
        load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"),
        OLMES_PATH / "arc_easy.jsonl",
    )


def load_boolq():
    dump_jsonl(
        load_dataset("google/boolq", split="validation"),
        OLMES_PATH / "boolq.jsonl",
    )


def load_csqa():
    dump_jsonl(
        load_dataset("tau/commonsense_qa", split="validation"),
        OLMES_PATH / "csqa.jsonl",
    )


def load_hellaswag():
    dump_jsonl(
        load_dataset("Rowan/hellaswag", split="validation"),
        OLMES_PATH / "hellaswag.jsonl",
    )


def load_mmlu():
    ds = load_dataset("cais/mmlu", "all")
    dump_jsonl(ds["test"], OLMES_PATH / "mmlu.jsonl")

    fewshots_path = OLMES_PATH / "mmlu_fewshots"
    fewshots_path.mkdir(exist_ok=True)

    for subject in MMLU_SUBJECTS:
        subj_ds = load_dataset("cais/mmlu", subject, split="dev")
        dump_jsonl(subj_ds, fewshots_path / f"{subject}.jsonl")


def load_openbookqa():
    dump_jsonl(
        load_dataset("allenai/openbookqa", "main", split="test"),
        OLMES_PATH / "openbookqa.jsonl",
    )


def load_piqa():
    dump_jsonl(
        load_dataset("nthngdy/piqa", split="validation"),
        OLMES_PATH / "piqa.jsonl",
    )


def load_socialiqa():
    dump_jsonl(
        load_dataset("lighteval/siqa", split="validation"),
        OLMES_PATH / "socialiqa.jsonl",
    )


def load_winogrande():
    dump_jsonl(
        load_dataset("allenai/winogrande", "winogrande_debiased", split="validation"),
        OLMES_PATH / "winogrande.jsonl",
    )


LOADERS = {
    "arc_challenge": load_arc_challenge,
    "arc_easy": load_arc_easy,
    "boolq": load_boolq,
    "csqa": load_csqa,
    "hellaswag": load_hellaswag,
    "mmlu": load_mmlu,
    "openbookqa": load_openbookqa,
    "piqa": load_piqa,
    "socialiqa": load_socialiqa,
    "winogrande": load_winogrande,
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--only",
        type=lambda s: s.split(","),
        default=None,
        help=f"Comma-separated subset of: {', '.join(LOADERS)}. Defaults to all.",
    )
    args = parser.parse_args()

    OLMES_PATH.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(LOADERS)
    unknown = set(selected) - set(LOADERS)
    if unknown:
        parser.error(f"Unknown dataset(s): {sorted(unknown)}")

    for name in selected:
        print(f"Loading {name}...")
        LOADERS[name]()


if __name__ == "__main__":
    main()
