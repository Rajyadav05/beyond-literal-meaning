# Non-Literal Language Lab — Beyond Literal Meaning
### Context-Aware Detection and Interpretation of Sarcasm and Idioms Using NLP

> **Research Question:** Can contextual NLP improve the detection and interpretation of non-literal language such as sarcasm and idiomatic expressions?  
> **Product Name:** Non-Literal Language Lab  
> **Backend:** FastAPI + PyTorch + Hugging Face Transformers + Scikit-Learn + SQLite  
> **Frontend:** Google Stitch Exported Vanilla HTML5 + ES6 JavaScript + Tailwind CSS  
> **Status:** All Core Phases Complete | 83/83 Pytest Backend & API Tests Passing

---

## 🌟 Overview & Architecture

**Non-Literal Language Lab** is an end-to-end, context-aware NLP system for detecting and interpreting non-literal language (verbal irony/sarcasm, idiomatic expressions, sentiment polarity shifts, fine-grained emotion distributions, and semantic vector divergence).

The frontend intentionally contains four unified user-facing pages:
1. **Analyzer (`analyzer.html`)**: Core multi-signal inference pipeline (`POST /api/analyze`) combining Sarcasm (TF-IDF & DistilBERT), Idiom Candidate Detection (PIE Lexicon), Surface Sentiment (Twitter-RoBERTa), GoEmotions (28 native classes), Semantic Vector Distance (`all-MiniLM-L6-v2`), Signal Alignment, and Evidence Fusion Synthesis.
2. **Context Lab (`context_lab.html`)**: Side-by-side comparison (`POST /api/analyze/context`) evaluating target utterances with and without preceding context to measure probability deltas and contextual incongruity shifts.
3. **Idiom Lab (`idiom_lab.html`)**: Standalone idiom detection (`POST /api/idiom`) with exact phrase span offsets, literal vs. idiomatic definitions, and usage classification (`IDIOMATIC` vs `LITERAL`).
4. **Challenge (`challenge.html`)**: Human-in-the-loop prediction review system (`POST /api/challenge`, `GET /api/challenge/stats`) persisting user evaluations (`correct` / `incorrect`) and optional feedback notes into a local SQLite database (`backend/lab.db`), displaying real-time accuracy and submission history logs.

---

## 🏗️ Project Architecture & Directory Layout

```
beyond-literal-meaning/
├── backend/                   # FastAPI Backend Application
│   ├── main.py                # FastAPI routes, CORS, startup initialization
│   ├── schemas.py             # Pydantic v2 request & response models
│   ├── database.py            # SQLite connection and CRUD methods
│   ├── lab.db                 # SQLite database for challenge feedback
│   └── services/              # High-level application service wrappers
│       ├── analysis_service.py # Multi-signal evidence fusion pipeline
│       └── idiom_service.py    # Lexicon & model-based idiom detection
├── frontend/                  # Google Stitch Exported HTML5/JS Frontend
│   ├── analyzer.html          # Page 1: Core Analyzer Dashboard
│   ├── context_lab.html       # Page 2: Bimodal Context Differential Studio
│   ├── idiom_lab.html         # Page 3: Idiom Disambiguation & Span Isolation
│   ├── challenge.html         # Page 4: Human-in-the-Loop Challenge & SQLite Log
│   ├── logo.svg               # Non-Literal Language Lab Logo
│   └── js/
│       └── api.js             # Centralized API Client Module (window.AppAPI)
├── src/                       # Validated NLP Core Engine Modules
│   ├── sarcasm/               # TF-IDF & DistilBERT sarcasm classifiers
│   ├── idiom/                 # Candidate detector, classifier & interpreter
│   ├── sentiment/             # Twitter-RoBERTa sentiment classifier
│   ├── emotion/               # 28-class GoEmotions multi-label classifier
│   ├── semantics/             # Sentence-transformers embedding & similarity
│   └── explainability/        # Evidence Fusion & Signal Alignment Engine
├── models/                    # Saved pretrained weight files and tokenizers
├── datasets/                  # Datasets (SARC, iSarcasmEval, MAGPIE, SemEval, GoEmotions)
├── results/                   # Evaluation metrics, predictions, and smoke-test snapshots
│   ├── metrics/               # Model evaluation JSON metrics
│   └── predictions/           # Snapshot contract and e2e test artifacts
├── tests/                     # Comprehensive Pytest Suite (83 Passing Tests)
├── scripts/                   # Verification, training & asset download scripts
├── requirements.txt           # Python dependencies
└── README.md                  # Comprehensive Documentation
```

---

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI (Uvicorn ASGI server)
- **Deep Learning / NLP**: PyTorch, Hugging Face Transformers (`distilbert-base-uncased`, `twitter-roberta-base-sentiment-latest`, `roberta-base-go_emotions`, `all-MiniLM-L6-v2`)
- **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer + Logistic Regression)
- **Database**: SQLite3 (`backend/lab.db`)
- **Validation**: Pydantic v2
- **Frontend**: Google Stitch UI Export — Vanilla HTML5, ES6 JavaScript (`fetch()`), Tailwind CSS (CDN), Material Symbols Outlined icons, Google Fonts (`Inter`, `JetBrains Mono`)
- **Test Suite**: Pytest (83/83 unit and API tests passing)

---

## 🚀 Step-by-Step Guide to Run the Project

### 1. Prerequisites
- Python **3.11** or **3.13** installed
- Git installed
- Modern Web Browser (Chrome, Edge, Firefox, Safari)

---

### 2. Set Up Virtual Environment

```bash
# Clone or navigate to the repository
cd beyond-literal-meaning

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Start the FastAPI Backend Server

Run the Uvicorn dev server from the project root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

> The API server will start on **`http://localhost:8000`**.  
> Interactive OpenAPI / Swagger documentation is available at **`http://localhost:8000/docs`**.

---

### 5. Launch the Frontend Application

In a separate terminal window (with virtual environment active), serve the `frontend/` directory using Python's built-in HTTP server:

```bash
python -m http.server 3000 --directory frontend
```

Now open your web browser and navigate to:
👉 **`http://localhost:3000/analyzer.html`**

#### Navigation between Pages:
- **Analyzer**: [http://localhost:3000/analyzer.html](http://localhost:3000/analyzer.html)
- **Context Lab**: [http://localhost:3000/context_lab.html](http://localhost:3000/context_lab.html)
- **Idiom Lab**: [http://localhost:3000/idiom_lab.html](http://localhost:3000/idiom_lab.html)
- **Challenge**: [http://localhost:3000/challenge.html](http://localhost:3000/challenge.html)

---

## 🧪 Running Unit & Integration Tests

The project includes a test suite covering all backend NLP modules, evidence fusion logic, SQLite persistence, and FastAPI endpoints.

Run the test suite:

```bash
pytest tests/
```

**Expected Result:**
```
================= 83 passed in 208.36s ==================
```

---

## 📡 API Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System readiness & SQLite database probe |
| `POST` | `/api/analyze` | Unified NLP analysis pipeline (sarcasm, idiom, sentiment, emotion, similarity, evidence fusion) |
| `POST` | `/api/analyze/context` | Side-by-side comparison with & without context (delta metrics, prediction shift) |
| `POST` | `/api/idiom` | Standalone idiom detection, span offsets, & literal vs. idiomatic definitions |
| `POST` | `/api/sentiment` | Twitter-RoBERTa 3-class sentiment classification |
| `POST` | `/api/emotion` | GoEmotions 28-class multi-label emotion prediction |
| `POST` | `/api/challenge` | Submit human reviewer verdict (`correct` / `incorrect`) to SQLite |
| `GET` | `/api/challenge/stats` | Retrieve aggregate SQLite challenge stats & recent submissions |
| `GET` | `/api/challenge/recent` | Retrieve 10 most recent challenge evaluation records |

---

## 🛡️ Key Principles & Scientific Integrity

1. **No External Inference APIs**: All inference runs locally on CPU/GPU via PyTorch and Transformers. No runtime calls to OpenAI, Gemini, or external APIs.
2. **No Fabricated Data**: All probabilities, sentiment scores, semantic distances, and SQLite statistics displayed in the frontend originate directly from live backend model outputs or database queries.
3. **Reproducibility**: Models and evaluation pipelines use fixed random seeds for deterministic reproducibility.

---

## 📄 License

This project is developed for academic research in Context-Aware Natural Language Processing.
