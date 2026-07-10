import os
import subprocess
import sys


def execute_script(script_name, work_dir=".", device="0"):
    script_path = os.path.join(work_dir, script_name)
    if not os.path.exists(script_path):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        env = os.environ.copy()
        if device is not None and str(device) != "":
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        result = subprocess.run(
            [sys.executable, "-u", script_name],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return_code = result.returncode
        if return_code != 0:
            observation = result.stderr or result.stdout
        else:
            observation = result.stdout or result.stderr
        return return_code, "The script has been executed. Here is the output:\n" + observation

    except Exception as e:
        print("++++", "Wrong!")
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."
