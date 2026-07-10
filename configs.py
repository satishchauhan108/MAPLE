import os
import sys


class Configs:
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_KEY", "")
    HF_KEY = ""
    PWC_KEY = ""
    SEARCHAPI_API_KEY = ""
    TAVILY_API_KEY = ""

    # --- Local LLM (Ollama / LM Studio) — default on Windows ---
    USE_LOCAL_LLM = True
    LOCAL_LLM_BACKEND = os.environ.get("LOCAL_LLM_BACKEND", "ollama")  # "ollama" or "lmstudio"
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234")
    DEFAULT_LOCAL_MODEL = os.environ.get("DEFAULT_LOCAL_MODEL", "llama3.1:8b")
    DEFAULT_LOCAL_LLM = os.environ.get("DEFAULT_LOCAL_LLM", "local")

    # Parser: local -> parse_local(); OpenAI cloud -> parse_openai(); vLLM -> parse()
    USE_OPENAI_PARSER = False if USE_LOCAL_LLM else (sys.platform != "win32")


def _local_entry(model: str) -> dict:
    base_url = (
        f"{Configs.LMSTUDIO_BASE_URL.rstrip('/')}/v1"
        if Configs.LOCAL_LLM_BACKEND == "lmstudio"
        else f"{Configs.OLLAMA_BASE_URL.rstrip('/')}/v1"
    )
    return {"api_key": "ollama-local", "base_url": base_url, "model": model}


def build_available_llms() -> dict:
    if Configs.USE_LOCAL_LLM:
        return {
            "local": _local_entry(Configs.DEFAULT_LOCAL_MODEL),
            "llama3.1": _local_entry("llama3.1:8b"),
            "mistral": _local_entry("mistral:7b"),
            "qwen2.5": _local_entry("qwen2.5:7b"),
        }
    return {
        "prompt-llm": {
            "api_key": "empty",
            "model": "prompt-llama",
            "base_url": "http://localhost:8000/v1",
        },
        "gpt-4.1": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4.1"},
        "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
        "gpt-3.5": {"api_key": Configs.OPENAI_KEY, "model": "gpt-3.5-turbo"},
    }


AVAILABLE_LLMs = build_available_llms()

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}
