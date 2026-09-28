"""
src/idiom/idiom_preprocessor.py
================================
Preprocessor for MAGPIE and SemEval-2022 Task 2 datasets.

Functions:
- load_and_clean_magpie: Processes MAGPIE jsonl dataset.
- load_and_clean_semeval: Processes SemEval Task 2 csv dataset.
- extract_idiom_span: Computes start_char and end_char offsets for a candidate idiom in text.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


def extract_idiom_span(sentence: str, idiom: str) -> Tuple[int, int, str]:
    """
    Finds exact character offsets (start_char, end_char) for an idiom in a sentence.
    Falls back to case-insensitive match if exact case match is not found.
    Returns (-1, -1, "") if not found.
    """
    if not sentence or not idiom:
        return (-1, -1, "")

    # 1. Try exact match
    idx = sentence.find(idiom)
    if idx != -1:
        return (idx, idx + len(idiom), sentence[idx:idx + len(idiom)])

    # 2. Try case-insensitive match
    pattern = re.compile(re.escape(idiom), re.IGNORECASE)
    match = pattern.search(sentence)
    if match:
        return (match.start(), match.end(), match.group(0))

    return (-1, -1, "")


def load_and_clean_magpie(file_path: Path) -> pd.DataFrame:
    """
    Loads MAGPIE jsonl dataset, filters out invalid rows, maps labels,
    and returns a cleaned DataFrame.
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rows.append(json.loads(line_str))

    df = pd.DataFrame(rows)

    # Filter out rows with invalid labels (keep only 'i' and 'l')
    df = df[df["label"].isin(["i", "l"])].copy()

    # Map binary label: 1 for idiomatic ('i'), 0 for literal ('l')
    df["binary_label"] = df["label"].map({"i": 1, "l": 0}).astype(int)

    # Clean text fields
    # MAGPIE context is a list of sentences where target sentence is context[sentence_no]
    def extract_magpie_text_and_context(row: pd.Series) -> Tuple[str, str]:
        ctx_list = row.get("context", [])

        target_sent = ""
        if isinstance(ctx_list, list) and len(ctx_list) > 0:
            target_idx = len(ctx_list) // 2  # MAGPIE target sentence is middle element (index 2 for 5 sentences)
            sent_elem = ctx_list[target_idx]
            if isinstance(sent_elem, list):
                target_sent = " ".join(sent_elem)
            else:
                target_sent = str(sent_elem)

        # Full context text (surrounding sentences)
        all_sents = []
        if isinstance(ctx_list, list):
            for elem in ctx_list:
                if isinstance(elem, list):
                    all_sents.append(" ".join(elem))
                else:
                    all_sents.append(str(elem))
        full_context = " ".join(all_sents)

        return target_sent, full_context

    extracted = df.apply(extract_magpie_text_and_context, axis=1)
    df["target_sentence"] = [e[0] for e in extracted]
    df["full_context"] = [e[1] for e in extracted]
    df["idiom_phrase"] = df["idiom"].astype(str)

    # Keep relevant normalized columns
    cleaned = pd.DataFrame({
        "example_id": df["id"].astype(str),
        "idiom": df["idiom_phrase"],
        "sentence": df["target_sentence"],
        "context": df["full_context"],
        "label": df["binary_label"],
        "raw_label": df["label"],
        "split": df["split"],
        "confidence": df.get("confidence", 1.0),
        "genre": df.get("genre", "unknown")
    })

    # Remove empty target sentences
    cleaned = cleaned[cleaned["sentence"].str.strip().str.len() > 0].copy()
    cleaned.drop_duplicates(subset=["example_id"], inplace=True)

    return cleaned


def load_and_clean_semeval(file_path: Path, language: str = "EN") -> pd.DataFrame:
    """
    Loads SemEval-2022 Task 2 CSV file, filters by language, maps labels,
    and returns a cleaned DataFrame.
    """
    df = pd.read_csv(file_path)

    if "Language" in df.columns and language:
        df = df[df["Language"] == language].copy()

    if "Label" in df.columns:
        df = df[df["Label"].isin([0, 1])].copy()
        df["binary_label"] = df["Label"].astype(int)
    else:
        df["binary_label"] = -1

    # Build context from Previous + Target + Next
    def build_sem_context(row: pd.Series) -> Tuple[str, str]:
        prev_txt = str(row["Previous"]) if pd.notna(row.get("Previous")) else ""
        tgt_txt = str(row["Target"]) if pd.notna(row.get("Target")) else ""
        next_txt = str(row["Next"]) if pd.notna(row.get("Next")) else ""

        full_ctx = " ".join([p for p in [prev_txt, tgt_txt, next_txt] if p])
        return tgt_txt, full_ctx

    extracted = df.apply(build_sem_context, axis=1)
    df["target_sentence"] = [e[0] for e in extracted]
    df["full_context"] = [e[1] for e in extracted]
    df["idiom_phrase"] = df["MWE"].astype(str)

    cleaned = pd.DataFrame({
        "example_id": df["DataID"].astype(str) if "DataID" in df.columns else df["ID"].astype(str),
        "idiom": df["idiom_phrase"],
        "sentence": df["target_sentence"],
        "context": df["full_context"],
        "label": df["binary_label"],
        "language": df["Language"],
        "setting": df.get("Setting", "zero_shot")
    })

    cleaned = cleaned[cleaned["sentence"].str.strip().str.len() > 0].copy()
    cleaned.drop_duplicates(subset=["example_id"], inplace=True)

    return cleaned
