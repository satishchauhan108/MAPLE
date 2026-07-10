# Local LLM setup (Ollama / LM Studio) — no OpenAI API required

## Quick start (Windows 11)

```powershell
# 1. Install Ollama from https://ollama.com/download/windows

# 2. Setup project + pull model
cd automl-agent
.\setup_ollama.ps1

# 3. Run pipeline
.\run_local.ps1
```

## Configuration (`configs.py`)

| Flag | Default | Purpose |
|------|---------|---------|
| `USE_LOCAL_LLM` | `True` | Use Ollama/LM Studio instead of OpenAI |
| `LOCAL_LLM_BACKEND` | `"ollama"` | `"ollama"` or `"lmstudio"` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `DEFAULT_LOCAL_MODEL` | `llama3.1:8b` | Default Ollama model |
| `USE_OPENAI_PARSER` | `False` when local | Uses `parse_local()` |

## Available local LLM keys

| Key | Ollama model |
|-----|--------------|
| `local` | `llama3.1:8b` (default) |
| `llama3.1` | `llama3.1:8b` |
| `mistral` | `mistral:7b` |
| `qwen2.5` | `qwen2.5:7b` |

## Switch back to OpenAI

Set in `configs.py`:

```python
USE_LOCAL_LLM = False
USE_OPENAI_PARSER = True
OPENAI_KEY = "sk-..."
```

## LM Studio

```powershell
$env:LOCAL_LLM_BACKEND = "lmstudio"
$env:DEFAULT_LOCAL_MODEL = "your-loaded-model-name"
.\run_local.ps1
```

LM Studio OpenAI endpoint: `http://localhost:1234/v1`
