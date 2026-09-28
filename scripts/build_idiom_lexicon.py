"""
scripts/build_idiom_lexicon.py
==============================
Phase 6 Step 5 — Builds the verified local idiom lexicon.

Aggregates unique idioms from MAGPIE, SemEval Task 2, and project demonstration idioms
into `datasets/processed/idiom_lexicon.json` and `models/idiom/idiom_lexicon.json`.

Usage:
    python scripts/build_idiom_lexicon.py
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
logger = logging.getLogger("build_idiom_lexicon")

# Manually verified definitions for demonstration & benchmark idioms
DEMO_IDIOM_DEFINITIONS = {
    "piece of cake": {
        "literal_meaning": "a piece of cake / a piece of baked food",
        "idiomatic_meaning": "very easy; simple to accomplish",
        "is_verified": True
    },
    "spill the beans": {
        "literal_meaning": "to knock over or spill physical bean seeds",
        "idiomatic_meaning": "reveal secret information prematurely or unintentionally",
        "is_verified": True
    },
    "spilled the beans": {
        "literal_meaning": "knocked over physical bean seeds",
        "idiomatic_meaning": "revealed secret information prematurely or unintentionally",
        "is_verified": True
    },
    "bite the bullet": {
        "literal_meaning": "to bite into a physical bullet cartridge",
        "idiomatic_meaning": "face a difficult situation with courage and get it over with",
        "is_verified": True
    },
    "break a leg": {
        "literal_meaning": "to fracture or break a leg bone",
        "idiomatic_meaning": "good luck (typically wished to performers before a show)",
        "is_verified": True
    },
    "under the weather": {
        "literal_meaning": "positioned physically under weather conditions",
        "idiomatic_meaning": "slightly unwell or sick",
        "is_verified": True
    },
    "off the beaten track": {
        "literal_meaning": "located away from a physical paved road or worn trail",
        "idiomatic_meaning": "in a remote, quiet, or unusual location",
        "is_verified": True
    },
    "in the running": {
        "literal_meaning": "actively participating in a foot race",
        "idiomatic_meaning": "having a realistic chance of winning or succeeding",
        "is_verified": True
    },
    "double dutch": {
        "literal_meaning": "a jump rope game using two ropes spinning in opposite directions",
        "idiomatic_meaning": "unintelligible or incomprehensible language / gibberish",
        "is_verified": True
    },
    "high life": {
        "literal_meaning": "living at a high physical elevation",
        "idiomatic_meaning": "a luxurious and expensive lifestyle",
        "is_verified": True
    }
}


def main():
    magpie_file = PROJECT_ROOT / "datasets" / "processed" / "magpie" / "train.parquet"
    semeval_file = PROJECT_ROOT / "datasets" / "processed" / "semeval_idiom" / "train.parquet"

    lexicon = {}

    # 1. Add demo idioms first
    for phrase, info in DEMO_IDIOM_DEFINITIONS.items():
        norm_p = phrase.lower().strip()
        lexicon[norm_p] = {
            "expression": phrase,
            "normalized_expression": norm_p,
            "source": "project_verified_lexicon",
            "known_usage": "figurative_and_literal",
            "literal_meaning": info["literal_meaning"],
            "idiomatic_meaning": info["idiomatic_meaning"],
            "has_dictionary_meaning": True
        }

    # 2. Add MAGPIE idioms
    if magpie_file.exists():
        df_m = pd.read_parquet(magpie_file)
        for idiom in df_m["idiom"].dropna().unique():
            norm_p = str(idiom).lower().strip()
            if norm_p not in lexicon:
                lexicon[norm_p] = {
                    "expression": str(idiom),
                    "normalized_expression": norm_p,
                    "source": "magpie",
                    "known_usage": "figurative_and_literal",
                    "literal_meaning": f"literal usage of phrase '{idiom}'",
                    "idiomatic_meaning": f"figurative usage of idiom '{idiom}'",
                    "has_dictionary_meaning": False
                }

    # 3. Add SemEval Task 2 idioms
    if semeval_file.exists():
        df_s = pd.read_parquet(semeval_file)
        for idiom in df_s["idiom"].dropna().unique():
            norm_p = str(idiom).lower().strip()
            if norm_p not in lexicon:
                lexicon[norm_p] = {
                    "expression": str(idiom),
                    "normalized_expression": norm_p,
                    "source": "semeval2022_task2",
                    "known_usage": "figurative_and_literal",
                    "literal_meaning": f"literal usage of multi-word expression '{idiom}'",
                    "idiomatic_meaning": f"figurative usage of multi-word expression '{idiom}'",
                    "has_dictionary_meaning": False
                }

    # Save to datasets/processed and models/idiom
    out_datasets = PROJECT_ROOT / "datasets" / "processed" / "idiom_lexicon.json"
    out_models_dir = PROJECT_ROOT / "models" / "idiom"
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_models = out_models_dir / "idiom_lexicon.json"

    with open(out_datasets, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2)

    with open(out_models, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2)

    logger.info("Saved idiom lexicon (%d entries) to:\n  - %s\n  - %s", len(lexicon), out_datasets, out_models)


if __name__ == "__main__":
    main()
