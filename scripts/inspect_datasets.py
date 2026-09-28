"""
scripts/inspect_datasets.py
============================
Phase 2 — Dataset Inspection & Manifest Generator.

Inspects raw dataset files in `datasets/raw/`:
1. SARC (if present)
2. iSarcasmEval (train.En.csv, task_A_En_test.csv, task_B_En_test.csv, task_C_En_test.csv)
3. MAGPIE (MAGPIE_filtered_split_random.jsonl, MAGPIE_filtered_split_typebased.jsonl, MAGPIE_unfiltered.jsonl)
4. SemEval-2022 Task 2 (train_zero_shot.csv, train_one_shot.csv, dev.csv, eval.csv)
5. GoEmotions (train.tsv, dev.tsv, test.tsv, emotions.txt)

Generates:
- Comprehensive stdout report for each dataset
- Manifest JSON saved to `datasets/processed/dataset_manifest.json`
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inspect_datasets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


def inspect_isarcasmeval() -> dict:
    logger.info("\n==========================================")
    logger.info("INSPECTING: iSarcasmEval")
    logger.info("==========================================")
    
    dir_path = RAW_DIR / "isarcasmeval"
    files_info = {}
    train_df = None

    for fname in ["train.En.csv", "task_A_En_test.csv", "task_B_En_test.csv", "task_C_En_test.csv"]:
        fpath = dir_path / fname
        if not fpath.exists():
            continue
        size_bytes = fpath.stat().st_size
        df = pd.read_csv(fpath)
        if fname == "train.En.csv":
            train_df = df

        files_info[fname] = {
            "size_bytes": size_bytes,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }
        logger.info("File: %s (%d bytes, %d rows)", fname, size_bytes, len(df))
        logger.info("  Columns: %s", list(df.columns))

    sample_examples = []
    label_dist = {}

    if train_df is not None:
        if "sarcastic" in train_df.columns:
            label_dist = {str(k): int(v) for k, v in train_df["sarcastic"].value_counts().to_dict().items()}
        sample_rows = train_df.head(5).to_dict(orient="records")
        for r in sample_rows:
            sample_examples.append({
                "tweet": r.get("tweet"),
                "sarcastic": int(r.get("sarcastic")) if pd.notnull(r.get("sarcastic")) else None,
                "rephrase": r.get("rephrase") if pd.notnull(r.get("rephrase")) else None,
            })

    return {
        "dataset_name": "iSarcasmEval",
        "source_url": "https://github.com/iabufarha/iSarcasmEval",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "files": files_info,
        "total_train_records": len(train_df) if train_df is not None else 0,
        "label_distribution": label_dist,
        "sample_examples": sample_examples,
        "context_available": False,
        "notes": (
            "iSarcasmEval contains intended sarcasm annotations directly from authors. "
            "Includes 'rephrase' column providing non-sarcastic intended meaning."
        )
    }


def inspect_magpie() -> dict:
    logger.info("\n==========================================")
    logger.info("INSPECTING: MAGPIE")
    logger.info("==========================================")

    dir_path = RAW_DIR / "magpie"
    files_info = {}
    main_df = None

    for fname in ["MAGPIE_filtered_split_random.jsonl", "MAGPIE_filtered_split_typebased.jsonl", "MAGPIE_unfiltered.jsonl"]:
        fpath = dir_path / fname
        if not fpath.exists():
            continue
        size_bytes = fpath.stat().st_size
        df = pd.read_json(fpath, lines=True)
        if fname == "MAGPIE_filtered_split_random.jsonl":
            main_df = df

        files_info[fname] = {
            "size_bytes": size_bytes,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }
        logger.info("File: %s (%d bytes, %d rows)", fname, size_bytes, len(df))

    sample_examples = []
    label_dist = {}
    context_available = False

    if main_df is not None:
        if "label" in main_df.columns:
            label_dist = {str(k): int(v) for k, v in main_df["label"].value_counts().to_dict().items()}
        context_available = "context" in main_df.columns
        sample_rows = main_df.head(5).to_dict(orient="records")
        for r in sample_rows:
            sample_examples.append({
                "idiom": r.get("idiom"),
                "label": r.get("label"),  # 'i' = idiomatic, 'l' = literal
                "context": r.get("context"),
            })

    return {
        "dataset_name": "MAGPIE",
        "source_url": "https://github.com/hslh/magpie-corpus",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "files": files_info,
        "total_records": len(main_df) if main_df is not None else 0,
        "label_distribution": label_dist,
        "sample_examples": sample_examples,
        "context_available": context_available,
        "notes": (
            "MAGPIE distinguishes between idiomatic (i) and literal (l) usages. "
            "Context contains preceding and succeeding sentences around the target expression."
        )
    }


def inspect_semeval_idiom() -> dict:
    logger.info("\n==========================================")
    logger.info("INSPECTING: SemEval-2022 Task 2 (Idiomaticity)")
    logger.info("==========================================")

    dir_path = RAW_DIR / "semeval_idiom"
    files_info = {}
    train_df = None

    for fname in ["train_zero_shot.csv", "train_one_shot.csv", "dev.csv", "eval.csv"]:
        fpath = dir_path / fname
        if not fpath.exists():
            continue
        size_bytes = fpath.stat().st_size
        df = pd.read_csv(fpath)
        if fname == "train_zero_shot.csv":
            train_df = df

        files_info[fname] = {
            "size_bytes": size_bytes,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }
        logger.info("File: %s (%d bytes, %d rows)", fname, size_bytes, len(df))

    sample_examples = []
    label_dist = {}
    context_available = False

    if train_df is not None:
        if "Label" in train_df.columns:
            label_dist = {str(k): int(v) for k, v in train_df["Label"].value_counts().to_dict().items()}
        context_available = "Previous" in train_df.columns or "Next" in train_df.columns
        sample_rows = train_df.head(5).to_dict(orient="records")
        for r in sample_rows:
            sample_examples.append({
                "MWE": r.get("MWE"),
                "Previous": r.get("Previous"),
                "Target": r.get("Target"),
                "Next": r.get("Next"),
                "Label": int(r.get("Label")) if pd.notnull(r.get("Label")) else None,
            })

    return {
        "dataset_name": "SemEval-2022 Task 2",
        "source_url": "https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "files": files_info,
        "total_train_records": len(train_df) if train_df is not None else 0,
        "label_distribution": label_dist,
        "sample_examples": sample_examples,
        "context_available": context_available,
        "notes": (
            "SubTask A binary idiomaticity classification dataset (0 = literal, 1 = idiomatic). "
            "Includes Previous, Target sentence, and Next sentence context."
        )
    }


def inspect_goemotions() -> dict:
    logger.info("\n==========================================")
    logger.info("INSPECTING: GoEmotions")
    logger.info("==========================================")

    dir_path = RAW_DIR / "goemotions"
    files_info = {}
    train_df = None
    emotion_labels = []

    emotions_txt_path = dir_path / "emotions.txt"
    if emotions_txt_path.exists():
        with open(emotions_txt_path, "r", encoding="utf-8") as f:
            emotion_labels = [line.strip() for line in f if line.strip()]

    for fname in ["train.tsv", "dev.tsv", "test.tsv"]:
        fpath = dir_path / fname
        if not fpath.exists():
            continue
        size_bytes = fpath.stat().st_size
        df = pd.read_csv(fpath, sep="\t", header=None, names=["text", "labels", "comment_id"])
        if fname == "train.tsv":
            train_df = df

        files_info[fname] = {
            "size_bytes": size_bytes,
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }
        logger.info("File: %s (%d bytes, %d rows)", fname, size_bytes, len(df))

    sample_examples = []
    if train_df is not None:
        sample_rows = train_df.head(5).to_dict(orient="records")
        for r in sample_rows:
            sample_examples.append({
                "text": r.get("text"),
                "labels": r.get("labels"),
            })

    return {
        "dataset_name": "GoEmotions",
        "source_url": "https://github.com/google-research/google-research/tree/master/goemotions",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "files": files_info,
        "total_train_records": len(train_df) if train_df is not None else 0,
        "emotion_categories_count": len(emotion_labels),
        "emotion_categories": emotion_labels,
        "sample_examples": sample_examples,
        "context_available": False,
        "notes": (
            f"Google Research GoEmotions dataset with {len(emotion_labels)} emotion categories (including neutral). "
            "Text column contains Reddit comments annotated for fine-grained emotions."
        )
    }


def inspect_sarc() -> dict:
    logger.info("\n==========================================")
    logger.info("INSPECTING: SARC (Self-Annotated Reddit Corpus)")
    logger.info("==========================================")

    dir_path = RAW_DIR / "sarc"
    sarc_file = dir_path / "train-balanced-sarcasm.csv"

    if sarc_file.exists():
        size_bytes = sarc_file.stat().st_size
        df = pd.read_csv(sarc_file)
        sample_rows = df.head(5).to_dict(orient="records")
        label_dist = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()} if "label" in df.columns else {}
        
        return {
            "dataset_name": "SARC",
            "source_url": "https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit",
            "download_date": datetime.now(timezone.utc).isoformat(),
            "status": "DOWNLOADED",
            "files": {
                "train-balanced-sarcasm.csv": {
                    "size_bytes": size_bytes,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
                }
            },
            "total_records": len(df),
            "label_distribution": label_dist,
            "sample_examples": sample_rows[:5],
            "context_available": "parent_comment" in df.columns,
            "notes": "SARC dataset with parent_comment context and comment target."
        }
    else:
        logger.info("SARC raw file 'train-balanced-sarcasm.csv' not yet placed in datasets/raw/sarc/.")
        return {
            "dataset_name": "SARC",
            "source_url": "https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit",
            "download_date": datetime.now(timezone.utc).isoformat(),
            "status": "MANUAL_DOWNLOAD_REQUIRED",
            "notes": (
                "SARC train-balanced-sarcasm.csv is hosted on Kaggle (~250MB). "
                "Download manually from Kaggle link and place in datasets/raw/sarc/train-balanced-sarcasm.csv."
            )
        }


def main():
    logger.info("=== Starting Dataset Inspection & Manifest Generation ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": "Non-Literal Language Lab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "isarcasmeval": inspect_isarcasmeval(),
            "magpie": inspect_magpie(),
            "semeval_idiom": inspect_semeval_idiom(),
            "goemotions": inspect_goemotions(),
            "sarc": inspect_sarc(),
        }
    }

    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("\n==========================================")
    logger.info("Manifest saved successfully to: %s", manifest_path)
    logger.info("==========================================")


if __name__ == "__main__":
    main()
