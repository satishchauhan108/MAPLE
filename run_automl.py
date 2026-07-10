"""AutoML-Agent CLI entry point (Windows-safe multiprocessing)."""
import argparse
import multiprocessing
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run AutoML-Agent pipeline")
    parser.add_argument("--task", default="tabular_classification")
    parser.add_argument("--llm", default="local")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--n-plans", type=int, default=1)
    parser.add_argument("--n-revise", type=int, default=1)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)

    from configs import Configs, AVAILABLE_LLMs
    from local_llm import is_local_llm_running, list_local_models

    if Configs.USE_LOCAL_LLM and not is_local_llm_running():
        print("ERROR: Local LLM is not running.")
        print("Start Ollama: ollama serve")
        print("Pull a model: ollama pull llama3.1:8b")
        sys.exit(1)

    if Configs.USE_LOCAL_LLM:
        print(f"Backend: {Configs.LOCAL_LLM_BACKEND}")
        print(f"Model: {AVAILABLE_LLMs[args.llm]['model']}")
        models = list_local_models()
        if models:
            print(f"Available: {', '.join(models[:8])}")

    from agent_manager import AgentManager

    user_prompt = args.prompt or (
        "Build a tabular classification model for the uploaded dataset. "
        "Use the dataset at the provided path."
    )
    manager = AgentManager(
        task=args.task,
        llm=args.llm,
        interactive=False,
        data_path=args.data_path,
        n_revise=args.n_revise,
        n_plans=args.n_plans,
    )
    manager.initiate_chat(user_prompt)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
