"""Launcher module - initiates and coordinates the GAIA evaluation process."""

import multiprocessing
import json
import os
from pathlib import Path

from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a


async def launch_evaluation():
    data_path = os.getenv("GAIA_PLUS_DATA", "data/gaia_plus.jsonl")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"

    print("Launching green agent...")
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("gaia_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    print("Launching white agent...")
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("gaia_white_agent", *white_address)
    )
    p_white.start()
    assert await my_a2a.wait_agent_ready(white_url), "White agent not ready in time"
    print("White agent is ready.")

    print("Sending GAIA task description to green agent...")
    task_config = {
        "white_agent_url": f"http://{white_address[0]}:{white_address[1]}",
        "data_path": data_path,
    }
    task_text = json.dumps(task_config)
    response = await my_a2a.send_message(green_url, task_text)
    print("Response from green agent:", response)

    # Wait a moment to let green flush outputs before termination
    print("Evaluation triggered. Waiting briefly before shutdown...")
    p_green.join(timeout=2)
    p_white.join(timeout=2)
    p_green.terminate()
    p_white.terminate()
    p_green.join()
    p_white.join()

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f).get("summary", {})
        print("Final metrics:", metrics)
    else:
        print("metrics.json not found; check green agent logs.")

    print("Agents terminated.")
