"""Local LLM integration via Ollama or LM Studio (OpenAI-compatible APIs)."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import requests
from openai import OpenAI

from configs import Configs


def get_local_base_url() -> str:
    if Configs.LOCAL_LLM_BACKEND == "lmstudio":
        base = Configs.LMSTUDIO_BASE_URL.rstrip("/")
    else:
        base = Configs.OLLAMA_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def get_local_client() -> OpenAI:
    return OpenAI(base_url=get_local_base_url(), api_key="ollama-local")


def get_ollama_client() -> OpenAI:
    return get_local_client()


def is_local_llm_running() -> bool:
    try:
        if Configs.LOCAL_LLM_BACKEND == "lmstudio":
            url = f"{Configs.LMSTUDIO_BASE_URL.rstrip('/')}/v1/models"
        else:
            url = f"{Configs.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_local_models() -> list[str]:
    try:
        if Configs.LOCAL_LLM_BACKEND == "lmstudio":
            url = f"{Configs.LMSTUDIO_BASE_URL.rstrip('/')}/v1/models"
            payload = requests.get(url, timeout=5).json()
            return [m.get("id", "") for m in payload.get("data", [])]
        url = f"{Configs.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        payload = requests.get(url, timeout=5).json()
        return [m.get("name", "") for m in payload.get("models", [])]
    except requests.RequestException:
        return []


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Any:
    client = get_local_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        if json_mode:
            kwargs.pop("response_format", None)
            return client.chat.completions.create(**kwargs)
        raise


def extract_json_text(content: str) -> str:
    content = content.strip()
    pattern = r"^```(?:json)?\s*\n(.*?)(?=^```)```"
    results = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if results:
        return results[0].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def parse_json_content(content: str) -> dict:
    text = extract_json_text(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.split("\n\n")[0].strip())


def resolve_model(model_key: Optional[str] = None) -> str:
    from configs import AVAILABLE_LLMs

    key = model_key or Configs.DEFAULT_LOCAL_LLM
    if key in AVAILABLE_LLMs:
        return AVAILABLE_LLMs[key]["model"]
    return Configs.DEFAULT_LOCAL_MODEL
