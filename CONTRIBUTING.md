# Contributing to Multimodal Agentic RAG Platform

Thank you for your interest in contributing to the **Multimodal Agentic RAG Platform**! This document provides guidelines and setup instructions to help you get started with development.

---

## 1. Development Standards & Philosophy

This project adheres to production-grade engineering principles:
- **Strict Typing**: All new functions and methods must have comprehensive Python type hints (`mypy` clean).
- **Zero Raw Exceptions**: Always catch and wrap failures using the custom exception hierarchy in `src/rag_platform/exceptions.py`.
- **Structured Observability**: Never use `print()` statements in library code. Use structured JSON loggers via `src/rag_platform/logging_config.py` with correlation ID propagation.
- **Pydantic Validation**: All API requests, responses, tool arguments, and ingestion chunks must use typed Pydantic models.
- **Tested Logic**: Every feature and fix must include comprehensive unit and/or integration tests in `tests/unit/` or `tests/integration/`.

---

## 2. Local Environment Setup

### Prerequisites
- Python 3.11 or 3.12
- Git
- Docker & Docker Compose (optional, for container testing)
- Tesseract OCR (for OCR testing: `apt install tesseract-ocr` or `brew install tesseract`)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/mubashir-yaseen/RAG_chatbot.git
cd RAG_chatbot

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install package in editable mode with development dependencies
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# 4. Install pre-commit git hooks
pre-commit install
```

---

## 3. Environment Variables

Copy the example template and fill in your development keys:

```bash
cp .env.example .env
```

Key variables:
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (for pgvector store)
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY` (for LLM inference)
- `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` (optional, for remote tracing)

---

## 4. Code Quality & Formatting

We enforce consistent formatting and linting via **Ruff** and type safety via **Mypy**.

```bash
# Check code style and formatting
ruff check .

# Automatically apply formatting fixes
ruff format .

# Run static type checking
mypy src
```

---

## 5. Running the Test Suite

Run unit and integration tests using `pytest`:

```bash
# Run all unit tests
python -m pytest -v tests/unit/

# Run tests with code coverage report
python -m pytest -v --cov=src/rag_platform --cov-report=term-missing tests/unit/

# Run live database integration tests (requires live Supabase credentials in .env)
python -m pytest -v tests/integration/
```

---

## 6. Running Local Services

### Start FastAPI Backend Server
```bash
python -m uvicorn rag_platform.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

### Start Streamlit Web UI
```bash
streamlit run ui/app.py
```

### Run Multi-Container Stack with Docker Compose
```bash
docker-compose up --build
```

---

## 7. Pull Request Process

1. **Create a branch**: Follow standard naming conventions:
   - `feat/feature-name`
   - `fix/bug-description`
   - `docs/documentation-update`
2. **Ensure tests pass**: Run `ruff check .`, `mypy src`, and `pytest` locally.
3. **Commit with Conventional Commits**:
   - `feat: add semantic caching layer`
   - `fix: handle OCR edge case on blank pages`
   - `docs: update deployment instructions`
4. **Open a Pull Request**: Provide a concise description of changes, context, and test validation.
