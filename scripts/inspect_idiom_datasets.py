"""
scripts/inspect_idiom_datasets.py
==================================
Phase 6 Step 1 — Inspection script for MAGPIE and SemEval-2022 Task 2 datasets.

Inspects raw dataset files, extracts schema details, label distributions,
context structures, and prints summary reports to enable manifest creation.

Usage:
    python scripts/inspect_idiom_datasets.py
"""

import json
import logging
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inspect_idiom_datasets")


def inspect_magpie():
    magpie_dir = PROJECT_ROOT / "datasets" / "raw" / "magpie"
    logger.info("=== Inspecting MAGPIE Dataset ===")
    
    files = list(magpie_dir.glob("*.jsonl"))
    results = {}
    
    for file_path in files:
        logger.info("Inspecting: %s", file_path.name)
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                rows.append(json.loads(line))
        
        df = pd.DataFrame(rows)
        logger.info("  Rows: %d", len(df))
        logger.info("  Columns: %s", list(df.columns))
        
        # Check label distribution if present
        label_dist = {}
        if "label" in df.columns:
            label_dist = df["label"].value_counts(dropna=False).to_dict()
            logger.info("  Label distribution: %s", label_dist)
        
        # Check split column if present
        split_dist = {}
        if "split" in df.columns:
            split_dist = df["split"].value_counts(dropna=False).to_dict()
            logger.info("  Split distribution: %s", split_dist)

        # Inspect first sample
        sample = rows[0] if rows else {}
        logger.info("  Sample keys: %s", list(sample.keys()))
        logger.info("  Sample idiom: %s | label: %s", sample.get("idiom"), sample.get("label"))
        if "sentence" in sample:
            logger.info("  Sample sentence: %s", sample.get("sentence")[:100])

        results[file_path.name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "split_distribution": {str(k): int(v) for k, v in split_dist.items()},
            "sample_keys": list(sample.keys())
        }
    return results


def inspect_semeval():
    semeval_dir = PROJECT_ROOT / "datasets" / "raw" / "semeval_idiom"
    logger.info("\n=== Inspecting SemEval-2022 Task 2 Dataset ===")
    
    files = list(semeval_dir.glob("*.csv"))
    results = {}
    
    for file_path in files:
        logger.info("Inspecting: %s", file_path.name)
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep="\t")
            
        logger.info("  Rows: %d", len(df))
        logger.info("  Columns: %s", list(df.columns))
        
        label_dist = {}
        if "label" in df.columns:
            label_dist = df["label"].value_counts(dropna=False).to_dict()
            logger.info("  Label distribution: %s", label_dist)
            
        sample = df.iloc[0].to_dict() if len(df) > 0 else {}
        logger.info("  Sample row: %s", sample)
        
        results[file_path.name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "sample_fields": list(sample.keys())
        }
    return results


def main():
    magpie_results = inspect_magpie()
    semeval_results = inspect_semeval()
    
    summary = {
        "magpie": magpie_results,
        "semeval2022_task2": semeval_results
    }
    
    logger.info("\n=== Inspection Summary Complete ===")


if __name__ == "__main__":
    main()
