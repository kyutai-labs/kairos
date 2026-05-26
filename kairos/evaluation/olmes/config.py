import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from kairos.evaluation.olmes.data import (
    load_allenai,
    load_boolq,
    load_hellaswag,
    load_mmlu,
    load_piqa,
    load_siqa,
    load_winogrande,
)
from kairos.utils.args_parser import ArgsParser

load_dotenv()

OLMES_DATA = Path(os.getenv("DATA_DIR", "./data")) / "olmes"


class OlmesArgs(ArgsParser):
    model: Path | None = None
    step: int | None = None
    data: Path = Path(OLMES_DATA)
    tasks: str = (
        "arc_easy,arc_challenge,boolq,csqa,hellaswag,mmlu,obqa,piqa,siqa,winogrande"
    )
    k_shot: int = 5
    seed: int = 1234

    # CSV logging
    results_dir: str = os.getenv("RESULTS_DIR", "./results")
    run_name: str | None = None

    # File logging
    log_file: str | None = None


class TaskConfig(BaseModel):
    name: str
    dataloader: Callable
    normalization: str  # none, character, token, or pmi
    filename: str
    max_samples: int


OLMES_TASKS = {
    "arc_challenge": TaskConfig(
        name="arc_challenge",
        dataloader=load_allenai,
        normalization="pmi",
        filename="arc_challenge.jsonl",
        max_samples=1172,
    ),
    "arc_easy": TaskConfig(
        name="arc_easy",
        dataloader=load_allenai,
        normalization="character",
        filename="arc_easy.jsonl",
        max_samples=1000,
    ),
    "boolq": TaskConfig(
        name="boolq",
        dataloader=load_boolq,
        normalization="none",
        filename="boolq.jsonl",
        max_samples=1000,
    ),
    "csqa": TaskConfig(
        name="csqa",
        dataloader=load_allenai,
        normalization="pmi",
        filename="csqa.jsonl",
        max_samples=1221,
    ),
    "hellaswag": TaskConfig(
        name="hellaswag",
        dataloader=load_hellaswag,
        normalization="character",
        filename="hellaswag.jsonl",
        max_samples=1000,
    ),
    "mmlu": TaskConfig(
        name="mmlu",
        dataloader=load_mmlu,
        normalization="character",
        filename="mmlu.jsonl",
        max_samples=14042,
    ),
    "obqa": TaskConfig(
        name="obqa",
        dataloader=load_allenai,
        normalization="pmi",
        filename="openbookqa.jsonl",
        max_samples=500,
    ),
    "piqa": TaskConfig(
        name="piqa",
        dataloader=load_piqa,
        normalization="character",
        filename="piqa.jsonl",
        max_samples=1000,
    ),
    "siqa": TaskConfig(
        name="siqa",
        dataloader=load_siqa,
        normalization="character",
        filename="socialiqa.jsonl",
        max_samples=1000,
    ),
    "winogrande": TaskConfig(
        name="winogrande",
        dataloader=load_winogrande,
        normalization="none",
        filename="winogrande.jsonl",
        max_samples=1267,
    ),
}
