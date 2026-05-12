# LuminaQ Platform — Installation Guide

## Quick start (Windows / macOS / Linux)

```bash
# 1. Extract the ZIP into your repo root, then:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install "jax[cpu]" numpy scipy matplotlib pydantic
pip install fastapi "uvicorn[standard]" websockets pydantic-settings python-multipart anyio
pip install pytest pytest-asyncio httpx

pip install -e packages/lumq-photonics
pip install -e packages/lumq-compiler
pip install -e packages/lumq-backends
pip install -e packages/lumq-api
pip install -e packages/lumq-algorithms

# 2. Run all tests
pytest packages/ -v --tb=short

# 3. Start API server
uvicorn lumq.api.app:app --reload --port 8000
# Swagger UI -> http://localhost:8000/docs
```

## Troubleshooting

**ModuleNotFoundError: No module named 'lumq'**
Make sure venv is active and you ran `pip install -e packages/...`

**VS Code does not recognise the Python files**
Ctrl+Shift+P -> Python: Select Interpreter -> choose .venv

**jax install fails**
Try: `pip install --upgrade pip` then `pip install "jax[cpu]"`


