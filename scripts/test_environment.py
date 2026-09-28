"""
scripts/test_environment.py
============================
Phase 1 environment validation script.

Checks:
1. Python version (3.11+ required)
2. All required packages can be imported
3. FastAPI app can be instantiated without errors
4. Project folder structure is present
5. SQLite database can be opened

Run from the project root:
    python scripts/test_environment.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import importlib
import io
import sqlite3
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminals that default to cp1252 / cp850.
# This avoids UnicodeEncodeError for box-drawing and symbol characters.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Colour helpers (pure ANSI, no external deps)
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

OK_TAG = f"{GREEN}[PASS]{RESET}"
FAIL_TAG = f"{RED}[FAIL]{RESET}"
WARN_TAG = f"{YELLOW}[WARN]{RESET}"


def _pass(label: str) -> None:
    print(f"  {OK_TAG}  {label}")


def _fail(label: str, hint: str = "") -> None:
    msg = f"  {FAIL_TAG}  {label}"
    if hint:
        msg += f"\n         {YELLOW}-> {hint}{RESET}"
    print(msg)


def _warn(label: str, hint: str = "") -> None:
    msg = f"  {WARN_TAG}  {label}"
    if hint:
        msg += f"\n         {YELLOW}-> {hint}{RESET}"
    print(msg)


def _header(title: str) -> None:
    sep = "-" * 56
    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{sep}{RESET}")


# ---------------------------------------------------------------------------
# Project root  (scripts/ lives one level below root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Check 1 -- Python version
# ===========================================================================

def check_python_version() -> bool:
    _header("1. Python version")
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if major == 3 and minor >= 11:
        _pass(f"Python {version_str}  (3.11+ required -- OK)")
        return True
    _fail(
        f"Python {version_str}",
        hint="Python 3.11 or newer is required. Update your interpreter or venv.",
    )
    return False


# ===========================================================================
# Check 2 -- Required packages
# ===========================================================================

# (import_name, display_name, is_optional)
_PACKAGES: list[tuple[str, str, bool]] = [
    # Core web
    ("fastapi",              "fastapi",                     False),
    ("uvicorn",              "uvicorn",                     False),
    ("pydantic",             "pydantic v2",                 False),
    # ML
    ("torch",                "torch (PyTorch)",             False),
    ("transformers",         "transformers (Hugging Face)", False),
    ("sentence_transformers","sentence-transformers",       False),
    ("sklearn",              "scikit-learn",                False),
    ("datasets",             "datasets (Hugging Face)",     False),
    # Data
    ("pandas",               "pandas",                      False),
    ("numpy",                "numpy",                       False),
    ("pyarrow",              "pyarrow",                     False),
    # NLP utils
    ("nltk",                 "nltk",                        False),
    ("spacy",                "spaCy",                       False),
    # Explainability (large deps -- optional)
    ("shap",                 "shap",                        True),
    ("lime",                 "lime",                        True),
    # Visualisation
    ("plotly",               "plotly",                      False),
    # Utilities
    ("dotenv",               "python-dotenv",               False),
    ("requests",             "requests",                    False),
    ("tqdm",                 "tqdm",                        False),
    ("httpx",                "httpx",                       False),
    ("pytest",               "pytest",                      False),
]


def check_packages() -> bool:
    _header("2. Python package imports")
    all_ok = True
    for import_name, display_name, optional in _PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            _pass(f"{display_name} ({version})")
        except ImportError:
            if optional:
                _warn(
                    f"{display_name} -- not installed (optional)",
                    hint=f"pip install {import_name}",
                )
            else:
                _fail(
                    f"{display_name} -- NOT installed",
                    hint="pip install -r requirements.txt",
                )
                all_ok = False
    return all_ok


# ===========================================================================
# Check 3 -- FastAPI app can be imported
# ===========================================================================

def check_fastapi_app() -> bool:
    _header("3. FastAPI application import")
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from backend.main import app  # noqa: F401
        from backend.schemas import HealthResponse  # noqa: F401
        _pass("backend.main imported successfully")
        _pass("backend.schemas imported successfully")
        return True
    except Exception as exc:
        _fail(f"Import error: {exc}")
        return False


# ===========================================================================
# Check 4 -- Folder structure
# ===========================================================================

_REQUIRED_DIRS: list[str] = [
    "backend",
    "backend/services",
    "datasets/raw/sarc",
    "datasets/raw/isarcasmeval",
    "datasets/raw/magpie",
    "datasets/raw/semeval_idiom",
    "datasets/raw/goemotions",
    "datasets/processed",
    "models/sarcasm",
    "models/idiom",
    "models/sentiment",
    "models/emotion",
    "models/embeddings",
    "results/metrics",
    "results/predictions",
    "results/confusion_matrices",
    "results/experiments",
    "scripts",
    "src/preprocessing",
    "src/sarcasm",
    "src/idiom",
    "src/sentiment",
    "src/emotion",
    "src/semantics",
    "src/explainability",
    "src/evaluation",
    "notebooks",
    "tests",
]

_REQUIRED_FILES: list[str] = [
    "requirements.txt",
    "README.md",
    ".gitignore",
    "backend/__init__.py",
    "backend/main.py",
    "backend/schemas.py",
    "backend/services/__init__.py",
    "scripts/test_environment.py",
    "scripts/download_assets.py",
]


def check_structure() -> bool:
    _header("4. Project folder structure")
    all_ok = True

    for rel in _REQUIRED_DIRS:
        path = PROJECT_ROOT / rel
        if path.is_dir():
            _pass(f"dir   {rel}/")
        else:
            _fail(f"dir   {rel}/  -- MISSING")
            all_ok = False

    for rel in _REQUIRED_FILES:
        path = PROJECT_ROOT / rel
        if path.is_file():
            _pass(f"file  {rel}")
        else:
            _fail(f"file  {rel}  -- MISSING")
            all_ok = False

    return all_ok


# ===========================================================================
# Check 5 -- SQLite
# ===========================================================================

def check_sqlite() -> bool:
    _header("5. SQLite database")
    db_path = PROJECT_ROOT / "backend" / "lab.db"
    try:
        conn = sqlite3.connect(str(db_path))
        version = conn.execute("SELECT sqlite_version()").fetchone()[0]
        conn.close()
        _pass(f"SQLite {version}  -- backend/lab.db is accessible")
        return True
    except Exception as exc:
        _fail(f"SQLite error: {exc}")
        return False


# ===========================================================================
# Check 6 -- PyTorch device (informational only, never a failure)
# ===========================================================================

def check_torch_device() -> None:
    _header("6. PyTorch device (informational)")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
                _pass(f"CUDA GPU {i}: {name} ({vram} MB VRAM)")
        else:
            _warn(
                "No CUDA GPU detected -- will run on CPU.",
                hint="CPU inference is fully supported. Training will be slower.",
            )
    except ImportError:
        _warn("torch not installed -- skipping device check.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    print(f"\n{BOLD}Non-Literal Language Lab -- Phase 1 Environment Check{RESET}")
    print(f"Project root: {PROJECT_ROOT}\n")

    results = [
        check_python_version(),
        check_packages(),
        check_fastapi_app(),
        check_structure(),
        check_sqlite(),
    ]
    check_torch_device()  # informational only

    passed = sum(results)
    total = len(results)

    sep = "-" * 56
    print(f"\n{BOLD}{sep}{RESET}")
    if all(results):
        print(f"{GREEN}{BOLD}  All {total} checks passed. Phase 1 environment is ready.{RESET}")
        print()
        print("  Next step -- start the API server:")
        print(f"  {BOLD}uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000{RESET}")
        print()
        print("  Then verify: http://localhost:8000/api/health")
        print()
        return 0
    else:
        failed = total - passed
        print(f"{RED}{BOLD}  {failed}/{total} check(s) failed.{RESET}")
        print()
        print("  Fix the issues above, then re-run:")
        print(f"  {BOLD}python scripts/test_environment.py{RESET}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
