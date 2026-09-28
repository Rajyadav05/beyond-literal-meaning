"""
scripts/download_assets.py
===========================
Phase 2 — Reproducible Dataset Downloader for Non-Literal Language Lab.

Downloads raw public datasets required for the project into `datasets/raw/<dataset>/`:
1. SARC (Self-Annotated Reddit Corpus)
2. iSarcasmEval
3. MAGPIE (Idiom usage)
4. SemEval-2022 Task 2 (Contextual Idiomaticity)
5. GoEmotions (Google Research)

Rules:
- Idempotent: Skips download if file already exists and is non-empty.
- Preserves raw original filenames and formats without altering them.
- Clear success/failure logging for each dataset.
- Does NOT fabricate or invent synthetic data.
- Handles direct public repository links + clear manual download instructions when Kaggle auth / large binary format is required.
"""

import os
import sys
import urllib.request
import urllib.error
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_assets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"

DATASET_CONFIGS = {
    "sarc": {
        "dir": RAW_DIR / "sarc",
        "files": {
            # Note: SARC's 1.3M row dataset (train-balanced-sarcasm.csv ~250MB) is hosted on Kaggle which requires auth/session cookies.
            # We attempt direct download mirrors if available, otherwise prompt manual download.
            "train-balanced-sarcasm.csv": "https://raw.githubusercontent.com/sherinclaudia/sarcastic-comments-on-reddit/master/train-balanced-sarcasm.csv",
        },
        "manual_instructions": (
            "SARC dataset 'train-balanced-sarcasm.csv' (~250 MB) requires manual download from Kaggle due to authentication rules:\n"
            "  1. Go to: https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit\n"
            "  2. Click 'Download' and extract 'train-balanced-sarcasm.csv'\n"
            "  3. Place the file at: datasets/raw/sarc/train-balanced-sarcasm.csv"
        )
    },
    "isarcasmeval": {
        "dir": RAW_DIR / "isarcasmeval",
        "files": {
            "train.En.csv": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv",
            "task_A_En_test.csv": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv",
            "task_B_En_test.csv": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_B_En_test.csv",
            "task_C_En_test.csv": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_C_En_test.csv",
        }
    },
    "magpie": {
        "dir": RAW_DIR / "magpie",
        "files": {
            "MAGPIE_filtered_split_random.jsonl": "https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_filtered_split_random.jsonl",
            "MAGPIE_filtered_split_typebased.jsonl": "https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_filtered_split_typebased.jsonl",
            "MAGPIE_unfiltered.jsonl": "https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_unfiltered.jsonl",
        }
    },
    "semeval_idiom": {
        "dir": RAW_DIR / "semeval_idiom",
        "files": {
            "train_zero_shot.csv": "https://raw.githubusercontent.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity/main/SubTaskA/Data/train_zero_shot.csv",
            "train_one_shot.csv": "https://raw.githubusercontent.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity/main/SubTaskA/Data/train_one_shot.csv",
            "dev.csv": "https://raw.githubusercontent.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity/main/SubTaskA/Data/dev.csv",
            "eval.csv": "https://raw.githubusercontent.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity/main/SubTaskA/Data/eval.csv",
        }
    },
    "goemotions": {
        "dir": RAW_DIR / "goemotions",
        "files": {
            "train.tsv": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv",
            "dev.tsv": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv",
            "test.tsv": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv",
            "emotions.txt": "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt",
        }
    }
}


def download_file(url: str, dest_path: Path) -> bool:
    """Downloads a file from url to dest_path if not already present."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("  [EXISTS] %s (%d bytes) - Skipping.", dest_path.name, dest_path.stat().st_size)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    logger.info("  [DOWNLOADING] %s from %s ...", dest_path.name, url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(temp_path, "wb") as out:
            block_size = 65536
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                out.write(buffer)

        temp_path.replace(dest_path)
        logger.info("  [SUCCESS] %s (%d bytes).", dest_path.name, dest_path.stat().st_size)
        return True
    except urllib.error.HTTPError as e:
        logger.warning("  [HTTP ERROR %d] %s for URL: %s", e.code, e.reason, url)
        if temp_path.exists():
            temp_path.unlink()
        return False
    except Exception as e:
        logger.warning("  [FAILED] %s: %s", dest_path.name, e)
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    logger.info("=== Starting Phase 2 Asset Downloader ===")
    summary = {}

    for ds_name, config in DATASET_CONFIGS.items():
        logger.info("\nProcessing dataset: %s", ds_name.upper())
        target_dir = config["dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        
        successes = 0
        total_files = len(config["files"])

        for filename, url in config["files"].items():
            dest = target_dir / filename
            ok = download_file(url, dest)
            if ok:
                successes += 1

        is_complete = successes == total_files
        summary[ds_name] = {
            "status": "COMPLETE" if is_complete else "MANUAL DOWNLOAD REQUIRED",
            "files_downloaded": f"{successes}/{total_files}",
            "target_dir": str(target_dir),
            "manual_instructions": config.get("manual_instructions"),
        }

    logger.info("\n=== Download Summary ===")
    for ds_name, res in summary.items():
        logger.info("%-15s : %s (%s files)", ds_name.upper(), res["status"], res["files_downloaded"])
        if res.get("manual_instructions") and "MANUAL" in res["status"]:
            logger.info("\n--- MANUAL ACTION REQUIRED FOR %s ---", ds_name.upper())
            logger.info(res["manual_instructions"])
            logger.info("------------------------------------\n")

    logger.info("Phase 2 Download script finished.")


if __name__ == "__main__":
    main()
