import requests

from serpapi import GoogleSearch
from openai import OpenAI
from configs import AVAILABLE_LLMs, Configs


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print_message("system", f"Kaggle unavailable (skipped): {e}")
        return None


def search_web(query):
    if not Configs.SEARCHAPI_API_KEY:
        return []
    params = {
        "engine": "google",
        "q": query,
        "api_key": Configs.SEARCHAPI_API_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])


def print_message(sender, msg, pid=None):
    pid = f"-{pid}" if pid else ""
    sender_color = {
        "user": color.PURPLE,
        "system": color.RED,
        "manager": color.GREEN,
        "model": color.BLUE,
        "data": color.DARKCYAN,
        "prompt": color.CYAN,
        "operation": color.YELLOW,
    }
    sender_label = {
        "user": "💬 You:",
        "system": "⚠️ SYSTEM NOTICE ⚠️\n",
        "manager": "🕴🏻 Agent Manager:",
        "model": f"🦙 Model Agent{pid}:",
        "data": f"🦙 Data Agent{pid}:",
        "prompt": "🦙 Prompt Agent:",
        "operation": f"🦙 Operation Agent{pid}:",
    }
    msg = f"{color.BOLD}{sender_color[sender]}{sender_label[sender]}{color.END}{color.END} {msg}"
    print(msg)
    print()


def safe_usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        return usage.to_dict(mode="json")
    except Exception:
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }


def get_client(llm: str = "local"):
    if Configs.USE_LOCAL_LLM:
        from local_llm import get_local_client

        return get_local_client()
    if llm.startswith("gpt"):
        return OpenAI(api_key=AVAILABLE_LLMs[llm]["api_key"])
    return OpenAI(
        base_url=AVAILABLE_LLMs[llm]["base_url"],
        api_key=AVAILABLE_LLMs[llm]["api_key"],
    )
