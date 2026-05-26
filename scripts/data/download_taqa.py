import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
DATA_PATH = Path(os.getenv("DATA_DIR", "./data"))


def main():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        path = hf_hub_download(
            repo_id="ROIM/temporal-alignment-qa",
            filename=f"data/{split}.jsonl",
            repo_type="dataset",
            local_dir=DATA_PATH,
        )
        target = DATA_PATH / f"taqa_{split}.jsonl"
        Path(path).rename(target)
        print(f"Downloaded TAQA {split} to {target}")


if __name__ == "__main__":
    main()
