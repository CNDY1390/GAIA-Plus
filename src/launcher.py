"""Launcher module - URL-driven evaluation trigger for GAIA-Plus."""

import json
import os
import asyncio
from pathlib import Path
from typing import Optional

import httpx
from a2a.client import A2AClient
from a2a.types import SendMessageSuccessResponse, Message


async def check_agent_readiness(url: str) -> bool:
    """Check if agent is ready by requesting agent-card."""
    try:
        # Normalize URL (remove trailing slash, add protocol if missing)
        normalized_url = url.rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            normalized_url = f"http://{normalized_url}"
        
        agent_card_url = f"{normalized_url}/.well-known/agent-card.json"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(agent_card_url)
            if response.status_code == 200:
                card = response.json()
                print(f"[OK] Agent ready at {normalized_url}: {card.get('name', 'unknown')}")
                return True
            else:
                print(f"[FAIL] Agent card returned status {response.status_code} at {agent_card_url}")
                return False
    except Exception as e:
        print(f"[FAIL] Cannot reach agent at {url}: {e}")
        return False


async def launch_evaluation():
    """Launch GAIA evaluation using URL-driven configuration."""
    # Read environment variables
    green_url = os.getenv("GREEN_URL")
    white_url = os.getenv("WHITE_URL")
    data_path = os.getenv("GAIA_PLUS_DATA", "data/gaia_plus.jsonl")
    print(
        f"[Launcher] env GREEN_URL={green_url!r}, WHITE_URL={white_url!r}, GAIA_PLUS_DATA={os.getenv('GAIA_PLUS_DATA')!r}"
    )
    
    if not green_url:
        print("[FAIL] GREEN_URL environment variable not set")
        print("Example: $env:GREEN_URL='http://localhost:9001'")
        return
    
    if not white_url:
        print("[FAIL] WHITE_URL environment variable not set")
        print("Example: $env:WHITE_URL='http://localhost:9002'")
        return
    
    # Normalize URLs
    green_url = green_url.rstrip("/")
    white_url = white_url.rstrip("/")
    
    if not green_url.startswith(("http://", "https://")):
        green_url = f"http://{green_url}"
    if not white_url.startswith(("http://", "https://")):
        white_url = f"http://{white_url}"
    
    print(f"[Launcher] Green URL: {green_url}")
    print(f"[Launcher] White URL: {white_url}")
    print(f"[Launcher] Data path: {data_path}")
    
    # Check readiness
    print("\n[1] Checking green agent readiness...")
    if not await check_agent_readiness(green_url):
        print("[FAIL] Green agent is not ready. Aborting.")
        return
    
    print("\n[2] Checking white agent readiness...")
    if not await check_agent_readiness(white_url):
        print("[FAIL] White agent is not ready. Aborting.")
        return
    
    # Construct task config
    task_config = {
        "white_agent_url": white_url,
        "data_path": data_path,
    }
    task_text = json.dumps(task_config)
    
    print(f"\n[3] Sending GAIA task to green agent...")
    print(f"Task config: {task_text}")
    
    try:
        # Send message using A2A client
        client = A2AClient(base_url=green_url)
        response = await client.send_message(task_text)
        
        # Handle response (may be wrapped or direct)
        if hasattr(response, 'root'):
            response_obj = response.root
        else:
            response_obj = response
        
        if isinstance(response_obj, SendMessageSuccessResponse):
            print("\n[SUCCESS] Task sent successfully!")
            result = response_obj.result
            if isinstance(result, Message):
                print(f"Response context ID: {result.context_id}")
            else:
                print(f"Response result: {result}")
            
            # Check for metrics file (wait a bit for it to be created)
            print("\n[4] Checking for metrics file...")
            await asyncio.sleep(2)
            metrics_path = Path("outputs/metrics.json")
            if metrics_path.exists():
                print("[OK] Metrics file found")
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                    summary = metrics_data.get("summary", {})
                    print(f"EM Mean: {summary.get('em_mean', 'N/A')}")
                    print(f"Latency Mean: {summary.get('latency_mean', 'N/A')}s")
                    print(f"Items: {summary.get('n_items', 'N/A')}")
                    print(f"Success: {summary.get('n_success', 'N/A')}")
            else:
                print("[INFO] Metrics file not found yet. Check outputs/ after evaluation completes.")
            
            print("\n[INFO] Evaluation triggered. Monitor outputs/metrics.json for results.")
        else:
            print(f"\n[WARN] Unexpected response type: {type(response_obj)}")
            print(f"Response: {response_obj}")
            
    except Exception as e:
        print(f"\n[FAIL] Error sending message: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    asyncio.run(launch_evaluation())


if __name__ == "__main__":
    main()
