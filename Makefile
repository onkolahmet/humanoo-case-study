# ==== Humanoo Case — Makefile (ML-powered) ====================================
# End-to-end: data → personas → learned scorer (XGBoost/LogReg) → bandit → API
# =============================================================================

# ---- Python & venv -----------------------------------------------------------
VENV_DIR      := .venv
VENV_PYTHON   := $(VENV_DIR)/bin/python
VENV_PIP      := $(VENV_DIR)/bin/pip
PYTHON        := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

# All project Python invocations go through RUNPY so PYTHONPATH is set.
RUNPY         := PYTHONPATH=. $(VENV_PYTHON)

# ---- Service URL -------------------------------------------------------------
URL           := http://127.0.0.1:8000

# ---- Colors ------------------------------------------------------------------
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RED := \033[0;31m
NC := \033[0m

# ---- Phony targets -----------------------------------------------------------
.PHONY: all setup check-venv \
        data train-personas train-ltr train-bandit train \
        api run \
        eval helper \
        test lint lint-fix format format-check type-check coverage \
        clean clean-all help

all: setup

# ---- Venv guard --------------------------------------------------------------
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(RED)Virtual environment not found.$(NC)"; \
		echo "$(YELLOW)Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi

# ---- One-time setup ----------------------------------------------------------
setup:
	@echo "$(GREEN)Setting up project environment…$(NC)"
	@if [ -z "$(PYTHON)" ]; then \
		echo "$(RED)Error: Python 3 is required$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	@echo "$(GREEN)Installing requirements…$(NC)"
	$(VENV_PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Setup complete$(NC)"
	@echo "$(YELLOW)Try: make run$(NC)"

# ==============================================================================
#                               DATA & TRAINING
# ==============================================================================

# Generate synthetic dataset (users, content, interactions) into ./data
data: check-venv
	@echo "$(GREEN)⧗ Generating mock data → ./data$(NC)"
	$(RUNPY) scripts/generate_data.py
	@echo "$(GREEN)✓ Data ready: ./data/users.csv, ./data/content_catalog.csv, ./data/interactions.csv$(NC)"

# # Train user personas (encoder + KMeans) → ./artifacts
# train-personas: check-venv
# 	@echo "$(GREEN)⧗ Training personas (KMeans) → ./artifacts$(NC)"
# 	$(RUNPY) scripts/train_personas.py
# 	@echo "$(GREEN)✓ Personas saved (encoder + kmeans)$(NC)"

# Train learned scorer (prefers XGBoost, falls back to Logistic Regression)
train-ltr: check-venv
	@echo "$(GREEN)⧗ Training learned scorer (XGBoost→LogReg) → ./artifacts/ltr_model.joblib$(NC)"
	$(RUNPY) scripts/train_ltr.py

# # Train bandit policy from historical interactions (offline init)
# train-bandit: check-venv
# 	@echo "$(GREEN)⧗ Training bandit (LinTS) → ./artifacts$(NC)"
# 	$(RUNPY) scripts/train_bandit.py
# 	@echo "$(GREEN)✓ Bandit saved$(NC)"

# Full training pipeline in correct order
train: train-ltr
	@echo "$(GREEN)✓ Training pipeline finished (personas → learned scorer → bandit)$(NC)"

# ==============================================================================
#                                 RUN / API
# ==============================================================================

# Start FastAPI without touching data/models (assumes artifacts exist)
api: check-venv
	@echo "$(GREEN)⧗ Starting FastAPI (CTRL+C to stop)…$(NC)"
	@echo "$(GREEN)Open $(URL)/docs$(NC)"
	$(RUNPY) scripts/run_api.py

# End-to-end: generate data → train all → start API
run: check-venv data train
	@echo "$(GREEN)⧗ Starting FastAPI (CTRL+C to stop)…$(NC)"
	@echo "$(GREEN)Open $(URL)/docs$(NC)"
	$(RUNPY) scripts/run_api.py

# ==============================================================================
#                               EVALUATION / QA
# ==============================================================================

# Offline evaluation: writes ./artifacts/metrics.json, then prints it
eval: check-venv
	@echo "$(GREEN)⧗ Running offline evaluation → artifacts/metrics.json$(NC)"
	$(RUNPY) scripts/evaluate.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"
	@echo "$(YELLOW)⧗ Metrics (artifacts/metrics.json)$(NC)"
	@cat artifacts/metrics.json



# Quick helper: show one consolidated helper bundle (requires API running)
helper:
	@echo "$(GREEN)⧗ GET $(URL)/helper$(NC)"
	@curl -s $(URL)/helper | python - <<'PY'
	import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))
	PY

# ==============================================================================
#                               DEV UTILITIES
# ==============================================================================

test: check-venv
	@echo "$(GREEN)Running tests…$(NC)"
	@if $(VENV_PYTHON) -m pytest --help | grep -q -- '--cov'; then \
		echo "$(YELLOW)pytest-cov detected — running with coverage$(NC)"; \
		PYTHONPATH=. $(VENV_PYTHON) -m pytest tests/ -v \
			--cov=src --cov=scripts \
			--cov-branch \
			--cov-report=term-missing \
			--cov-report=html; \
	else \
		PYTHONPATH=. $(VENV_PYTHON) -m pytest tests/ -v; \
	fi
	@echo "$(GREEN)✓ Tests completed$(NC)"
	@if [ -d "htmlcov" ]; then \
		echo "$(YELLOW)📊 Coverage report: htmlcov/index.html$(NC)"; \
	fi

lint: check-venv
	@echo "$(GREEN)Ruff lint (check only)…$(NC)"
	$(RUNPY) -m ruff check src scripts tests

lint-fix: check-venv
	@echo "$(GREEN)Ruff lint (auto-fix)…$(NC)"
	$(RUNPY) -m ruff check --fix src scripts tests

format: check-venv
	@echo "$(GREEN)Black (apply formatting)…$(NC)"
	$(RUNPY) -m black -l 120 src scripts tests

format-check: check-venv
	@echo "$(GREEN)Black (check only)…$(NC)"
	$(RUNPY) -m black --check -l 120 src scripts tests

type-check: check-venv
	@echo "$(GREEN)mypy (static type checks)…$(NC)"
	MYPYPATH=src $(RUNPY) -m mypy src

coverage: check-venv
	@if [ ! -f "htmlcov/index.html" ]; then \
		echo "$(RED)No coverage report found. Run 'make test' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Opening coverage report…$(NC)"
	@if command -v open >/dev/null 2>&1; then open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then xdg-open htmlcov/index.html; \
	else echo "$(YELLOW)Please open htmlcov/index.html in your browser$(NC)"; fi

# ==============================================================================
#                                 CLEANUP
# ==============================================================================

clean:
	@echo "$(GREEN)Cleaning temporary files…$(NC)"
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage .mypy_cache/ .ruff_cache/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +

clean-all: clean
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Removing $(VENV_DIR)…$(NC)"; \
		rm -rf $(VENV_DIR); \
	fi

# ==============================================================================
#                                   HELP
# ==============================================================================

help:
	@echo "$(BLUE)Humanoo Case — ML pipeline$(NC)"
	@echo "$(BLUE)===========================$(NC)\n"
	@echo "$(GREEN)Setup & Dev$(NC)"
	@echo "  $(YELLOW)make setup$(NC)          - Create venv & install requirements"
	@echo "  $(YELLOW)make data$(NC)           - Generate mock dataset to ./data"
	@echo "  $(YELLOW)make train$(NC)          - Train personas → learned scorer → bandit"
	@echo "  $(YELLOW)make api$(NC)            - Start FastAPI (assumes artifacts exist)"
	@echo "  $(YELLOW)make run$(NC)            - Data + train + start FastAPI"
	@echo ""
	@echo "$(GREEN)Evaluation$(NC)"
	@echo "  $(YELLOW)make eval$(NC)           - Offline metrics to artifacts/metrics.json"
	@echo "  $(YELLOW)make helper$(NC)         - Fetch consolidated helper bundle from /helper"
	@echo ""
	@echo "$(GREEN)Quality Tools$(NC)"
	@echo "  $(YELLOW)make test$(NC)           - Run unit tests (with coverage if available)"
	@echo "  $(YELLOW)make lint$(NC)           - Ruff lint (check)"
	@echo "  $(YELLOW)make lint-fix$(NC)       - Ruff lint (auto-fix)"
	@echo "  $(YELLOW)make format$(NC)         - Black (apply)"
	@echo "  $(YELLOW)make format-check$(NC)   - Black (check only)"
	@echo "  $(YELLOW)make type-check$(NC)     - mypy static type checks"
	@echo "  $(YELLOW)make coverage$(NC)       - Open HTML coverage report"
	@echo ""
	@echo "$(GREEN)Cleaning$(NC)"
	@echo "  $(YELLOW)make clean$(NC)          - Remove caches/artifacts (not venv)"
	@echo "  $(YELLOW)make clean-all$(NC)      - Also remove $(VENV_DIR)"
