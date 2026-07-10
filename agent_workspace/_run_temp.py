import os
import sys

sys.path.insert(0, r"C:\Users\satis\OneDrive\Pictures\Desktop\MAPLE\automl-agent")
os.chdir(r"C:\Users\satis\OneDrive\Pictures\Desktop\MAPLE\automl-agent")

from agent_manager import AgentManager

data_path = r"agent_workspace\datasets\sample_banana.csv"
user_prompt = (
    "Build a tabular classification model to predict banana quality (good/bad) "
    "from numerical features. The dataset is at the provided path."
)
manager = AgentManager(
    task="tabular_classification",
    llm="gpt-4",
    interactive=False,
    data_path=data_path,
    n_revise=1,
    n_plans=1,
)
manager.initiate_chat(user_prompt)