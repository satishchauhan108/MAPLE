# AutoML-Agent — Final Windows 11 Setup Guide

This guide reflects the Windows-compatible configuration in this repository. **vLLM is not required** on Windows. Prompt parsing uses OpenAI when `Configs.USE_OPENAI_PARSER` is `True` (default on Windows).

---

## Prerequisites

- Windows 11
- Python 3.11.x
- PowerShell
- OpenAI API key (required for all agents)
- Optional: SerpAPI key (web search during planning), Kaggle credentials (Kaggle retrieval)

---

## 1. Clone and enter the project

```powershell
cd "c:\Users\satis\OneDrive\Pictures\Desktop\MAPLE\automl-agent"
```

---

## 2. Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 3. Install dependencies

**Recommended (minimal — tabular / core tasks):**

```powershell
.\setup_windows.ps1 -Profile minimal
```

**Full profile (image, text, graph templates):**

```powershell
.\setup_windows.ps1 -Profile full
```

**Manual equivalent:**

```powershell
pip install --upgrade pip setuptools wheel
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r minimal_requirements.txt
pip install torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.5.3 -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torchtext==0.18.0
```

Use `minimal_requirements.txt` for the agent framework. Use `windows_requirements.txt` for the full PyTorch task stack.

---

## 4. Configure API keys

Edit **`configs.py`**:

```python
class Configs:
    OPENAI_KEY = "sk-your-openai-key-here"   # REQUIRED
    SEARCHAPI_API_KEY = "your-serpapi-key"   # optional, for Google web search in RAG
    USE_OPENAI_PARSER = True                 # default True on Windows; no vLLM needed
```

| Setting | File | Purpose |
|---------|------|---------|
| OpenAI key | `configs.py` → `Configs.OPENAI_KEY` | All LLM agents + prompt parsing |
| SerpAPI key | `configs.py` → `Configs.SEARCHAPI_API_KEY` | Web search during knowledge retrieval |
| Kaggle | `%USERPROFILE%\.kaggle\kaggle.json` | Optional Kaggle notebook/dataset retrieval |
| Parser mode | `configs.py` → `Configs.USE_OPENAI_PARSER` | `True` = OpenAI (`parse_openai`), `False` = vLLM (`parse`) |

On Windows, **`USE_OPENAI_PARSER` defaults to `True`**. You do not need to run vLLM or change agent code.

---

## 5. Place datasets

Put your data under:

```
automl-agent/agent_workspace/datasets/
```

Example:

```
agent_workspace/datasets/banana_quality.csv
```

The notebook and launcher pick the first entry in that folder if you do not specify a path.

---

## 6. Verify installation

```powershell
.\venv\Scripts\python.exe -c "from configs import Configs; from agent_manager import AgentManager; from operation_agent import OperationAgent; from model_agent import ModelAgent; print('USE_OPENAI_PARSER=', Configs.USE_OPENAI_PARSER); print('All imports OK')"
```

Expected on Windows:

```
USE_OPENAI_PARSER= True
All imports OK
```

---

## 7. Launch the notebook

```powershell
.\run_project.ps1 -RunNotebook
```

Or:

```powershell
jupyter notebook AutoMLAgent.ipynb
```

In the notebook, set:

- `task` — one of: `tabular_classification`, `tabular_regression`, `tabular_clustering`, `image_classification`, `text_classification`, `node_classification`, `ts_forecasting`
- `llm` — e.g. `"gpt-4"`
- `data_path` — path to your dataset file or folder

Run the cell to start the pipeline.

---

## 8. Run the pipeline from PowerShell (no notebook)

```powershell
.\run_project.ps1 -RunExample `
  -Task tabular_classification `
  -DataPath "agent_workspace\datasets\banana_quality.csv" `
  -Llm gpt-4
```

Generated code is written to:

```
agent_workspace/exp/
```

Trained models (when saved by generated code):

```
agent_workspace/trained_models/
```

---

## Parser behavior (Windows vs Linux)

| Platform | Default | Method used |
|----------|---------|-------------|
| Windows | `USE_OPENAI_PARSER = True` | `parse_openai()` via GPT-4 |
| Linux | `USE_OPENAI_PARSER = False` | `parse()` via local vLLM prompt-llm |

To force OpenAI parsing on Linux, set `Configs.USE_OPENAI_PARSER = True` in `configs.py`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Client.__init__() got an unexpected keyword argument 'proxies'` | Reinstall: `pip install "httpx>=0.23.0,<0.28.0"` |
| Kaggle error at startup | Fixed — Kaggle loads lazily; only needed when retrieval runs |
| PapersWithCode errors | `_data/paperswithcode/` is optional; web/arXiv/Kaggle still work |
| High API cost | Use `n_plans=1`, `n_revise=1` in notebook or `-RunExample` |
| vLLM mentioned in README | Linux/WSL optional path only; not used on Windows |

---

## Quick reference — final run command

```powershell
cd "c:\Users\satis\OneDrive\Pictures\Desktop\MAPLE\automl-agent"
.\venv\Scripts\Activate.ps1
# After setting Configs.OPENAI_KEY in configs.py:
.\run_project.ps1 -RunExample -Task tabular_classification -DataPath "agent_workspace\datasets\your_data.csv"
```
