# Humanoo Case Study – Applied AI Engineer

## Overview
This repository contains my submission for the Humanoo Applied AI Engineer case study.  
The objective was to address **early user drop-off within the first two weeks** by designing and implementing an AI/ML-powered prototype that demonstrates how personalization can improve engagement.

This project includes:
- **Code** (Python scripts with a Makefile workflow)
- **README.md** (setup, run, testing, coverage, guidelines)
- **docs/** (screenshots and PDF summary)
- **PDF summary** covering problem exploration, solution proposal, prototype implementation, and trade-offs

---

## Problem Exploration (Guideline)
- **Core problem**: low retention during onboarding.  
- **Assumptions**: users churn when activities feel generic, reminders are mistimed, motivation is missing.  
- **Signals**: login frequency, time-of-day usage, content preference, onboarding answers, early conversions.

---

## Solution Proposal (Guideline)
- **Main approach**: Learning-to-rank recommender using XGBoost.  
- **How it fits**: onboarding-tailored starter pack → personalized updates in first 2 weeks → context-aware notifications.  
- **Alternatives considered**: clustering cohorts, motivational LLM prompts, popularity-based fallbacks.

---

## Project Structure
```
.
├── Makefile
├── requirements.txt
├── scripts/
│   ├── train_ltr.py       # Training pipeline
│   ├── evaluate.py        # Offline evaluation
│   ├── run_api.py         # API runner
│   ├── ltr.py             # XGBoost helper functions
│   ├── schemas.py         # Pydantic schemas
│   └── generate_data.py   # Dummy data generation
├── src/
│   ├── features/          # Feature engineering modules
│   ├── models/            # Bandit, recommender, LTR model
│   └── service/           # API & schemas
├── tests/                 # Unit tests
├── data/                  # Dummy data
├── models/                # Saved artifacts
└── docs/
    ├── Humanoo_Case_Study_Summary.pdf
    ├── coverage_console.png
    └── coverage_html.png
```

---

## How to Run (Makefile-driven)

### 1. Setup Environment
```bash
make setup
```

### 2. Run Tests with Coverage
```bash
make test
```
- Generates a coverage report (HTML in `htmlcov/index.html`).  
- Screenshots of the 92% coverage are stored in `docs/`.  
- I wrote **comprehensive test cases**, covering edge cases and ensuring correctness of both training and API.

### 3. Start the Application
```bash
make run
```
- Launches FastAPI backend at `http://localhost:8000`  
- Interact with API or UI

### 4. Evaluate Offline
Open a **new terminal** and run:
```bash
make eval
```

### 5. Optional Commands
- `make lint` – code linting  
- `make format` – auto-format source code  
- `make clean` – remove build artifacts  

---

## Example Request
```bash
curl -X POST "http://localhost:8000/recommend"      -H "Content-Type: application/json"      -d '{"user_id": 123, "k": 5}'
```

---

## Trade-offs & Risks (Guideline)
- Cold start – mitigated with onboarding defaults and popularity priors  
- Explainability – SHAP values needed for interpretation  
- Fairness/diversity – prevent skew toward one content type  
- Latency – addressed with caching and feature pre-compute  
- Scalability – production would need distributed feature pipelines  

---

## Next Steps (Guideline)
With more time I would:  
- Add embeddings for users/items to improve retrieval  
- Build an A/B testing harness for online evaluation  
- Integrate contextual bandits for adaptive personalization  
- Expand simulated data to reflect complex user journeys  

---

## Deliverables
- **README.md** – this file  
- **docs/Humanoo_Case_Study_Summary.pdf** – structured reflection (problem, solution, prototype, trade-offs)  
- **docs/** – screenshots of 92% test coverage  
- **Codebase** – Python scripts with recommender prototype  

---
