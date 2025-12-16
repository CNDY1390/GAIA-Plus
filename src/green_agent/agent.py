"""GAIA-Plus green agent - orchestrates benchmark over GAIA JSONL."""

import json
import os
import time
import re
import sys
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
import tomllib
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, Message, SendMessageSuccessResponse
from a2a.utils import get_text_parts, new_agent_text_message
from pydantic import BaseModel

from my_util import my_a2a

dotenv.load_dotenv()

DEFAULT_DATA_PATH = os.getenv("GAIA_PLUS_DATA", "data/gaia_plus.jsonl")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OUTPUTS_DIR = Path("outputs")


class GaiaItem(BaseModel):
    id: str
    question: str
    answer: str
    level: str | None = None
    meta: Dict[str, Any] | None = None


def ensure_required_envs():
    missing: List[str] = []
    if not os.getenv("OPENAI_API_KEY") and os.getenv("WHITE_MODE") != "dummy_correct":
        missing.append("OPENAI_API_KEY (required unless WHITE_MODE=dummy_correct)")
    if not os.getenv("OPENAI_MODEL"):
        print("[green] OPENAI_MODEL not set; defaulting to gpt-4o-mini.")
    else:
        print(f"[green] Using OPENAI_MODEL={os.getenv('OPENAI_MODEL')}")
    if os.getenv("OPENAI_BASE_URL"):
        print(f"[green] OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')}")
    if os.getenv("WHITE_MODE"):
        print(f"[green] WHITE_MODE={os.getenv('WHITE_MODE')}")
    if missing:
        print(f"[green] Missing required env: {', '.join(missing)}. Exiting.")
        sys.exit(1)
    data_path = os.getenv("GAIA_PLUS_DATA", DEFAULT_DATA_PATH)
    if not Path(data_path).exists():
        print(f"[green] GAIA_PLUS_DATA not found at {data_path}. Exiting.")
        sys.exit(1)
    return data_path


def get_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def load_agent_card_toml(agent_name: str):
    current_dir = Path(__file__).parent
    with open(current_dir / f"{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def load_dataset(path: str | Path) -> tuple[List[GaiaItem], int]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"GAIA data not found at {dataset_path}")
    items: List[GaiaItem] = []
    skipped = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item = GaiaItem(**obj)
                # ensure required fields
                if not item.id or not item.question or not item.answer:
                    raise ValueError("missing id/question/answer")
                items.append(item)
            except Exception as e:
                skipped += 1
                print(f"[green] WARNING: skip line {lineno} due to {e}")
                continue
    return items, skipped


def normalize_text(text: str) -> str:
    # Lowercase, strip whitespace, drop surrounding punctuation-like chars.
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = cleaned.strip(" .,!?:;\"'()[]{}")
    return cleaned


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


async def ask_white_for_answer(
    white_agent_url: str, question: str, context_id: str | None = None
) -> tuple[str, str | None]:
    print(
        f"[green] -> white url={white_agent_url} ctx={context_id} q_preview={question[:120]!r}"
    )
    white_agent_response = await my_a2a.send_message(
        white_agent_url, question, context_id=context_id
    )
    res_root = white_agent_response.root
    assert isinstance(res_root, SendMessageSuccessResponse)
    res_result = res_root.result
    assert isinstance(res_result, Message)
    text_parts = get_text_parts(res_result.parts)
    assert (
        len(text_parts) >= 1
    ), "Expecting at least one text part from the white agent"
    answer = text_parts[0]
    print(
        f"[green] <- white ctx={res_result.context_id} ans_preview={answer[:120]!r}"
    )
    return answer, res_result.context_id


class GaiaGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        self.dataset_path = DEFAULT_DATA_PATH
        self.white_url = os.getenv("WHITE_AGENT_URL")
        self.retries = int(os.getenv("GREEN_RETRY", "1"))
        self.model = DEFAULT_MODEL

    def _parse_task_config(self, user_input: str):
        # Accept either JSON dict or tag-wrapped url/config.
        try:
            cfg = json.loads(user_input)
            self.white_url = cfg.get("white_agent_url", self.white_url)
            self.dataset_path = cfg.get("data_path", self.dataset_path)
            return
        except Exception:
            pass

        if "<white_agent_url>" in user_input:
            start = user_input.split("<white_agent_url>", 1)[1]
            white_url = start.split("</white_agent_url>", 1)[0].strip()
            if white_url:
                self.white_url = white_url
        if "<gaia_data_path>" in user_input:
            start = user_input.split("<gaia_data_path>", 1)[1]
            data_path = start.split("</gaia_data_path>", 1)[0].strip()
            if data_path:
                self.dataset_path = data_path

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        self._parse_task_config(user_input)

        # Env validation
        self.dataset_path = ensure_required_envs()
        
        if not self.white_url:
            error_msg = "WHITE_AGENT_URL not set. Must be provided via environment variable or task config."
            print(f"[green] ERROR: {error_msg}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: {error_msg}")
            )
            return

        print(
            f"Green agent: Starting GAIA evaluation with data={self.dataset_path}, white={self.white_url}, model={self.model}"
        )
        items, skipped = load_dataset(self.dataset_path)
        total_items = skipped + len(items)
        OUTPUTS_DIR.mkdir(exist_ok=True)

        per_item: List[Dict[str, Any]] = []
        latencies: List[float] = []
        ems: List[float] = []
        started_at = datetime.utcnow().isoformat() + "Z"
        run_id = uuid.uuid4().hex
        total_start = time.perf_counter()
        errors = 0
        successes = 0

        for item in items:
            q_text = f"Question: {item.question}\nPlease output only the final short answer."
            # In dummy_correct mode we optionally embed the gold for the white to echo.
            if os.getenv("WHITE_MODE") == "dummy_correct":
                q_text += f"\nGOLD: {item.answer}"
            t0 = time.perf_counter()
            pred = ""
            ctx_id = None
            error_msg = None
            for attempt in range(self.retries + 1):
                try:
                    pred, ctx_id = await ask_white_for_answer(
                        self.white_url, q_text, context_id=None
                    )
                    break
                except Exception as e:
                    error_msg = str(e)
                    print(f"[green] ERROR attempt {attempt+1} on {item.id}: {e}")
                    if attempt >= self.retries:
                        pred = ""
            latency = time.perf_counter() - t0

            em = exact_match(pred, item.answer) if pred else 0.0
            if em == 1.0:
                successes += 1
            else:
                errors += 1 if error_msg else 0
            per_item.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "pred": pred,
                    "gold": item.answer,
                    "em": em,
                    "latency": latency,
                    "context_id": ctx_id,
                    "error": error_msg,
                    "level": item.level,
                }
            )
            ems.append(em)
            latencies.append(latency)
            print(f"[GAIA] {item.id}: em={em} latency={latency:.3f}s pred={pred}")

        em_mean = sum(ems) / len(ems) if ems else 0.0
        latency_mean = sum(latencies) / len(latencies) if latencies else 0.0
        metrics = {
            "em_mean": em_mean,
            "latency_mean": latency_mean,
            "n_items": len(items),
            "n_total_lines": total_items,
            "n_skipped": skipped,
            "n_success": successes,
            "n_failed": len(items) - successes,
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat() + "Z",
            "total_duration": time.perf_counter() - total_start,
            "git_commit": get_git_commit(),
        }

        # persist metrics
        metrics_path = OUTPUTS_DIR / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"summary": metrics, "details": per_item}, f, ensure_ascii=False, indent=2)
        # lightweight csv
        csv_path = OUTPUTS_DIR / "metrics.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,pred,gold,em,latency\n")
            for row in per_item:
                f.write(
                    f"{row['id']},{row['pred'].replace(',',';')},{row['gold'].replace(',',';')},{row['em']},{row['latency']}\n"
                )

        print("Green agent: Evaluation complete.", metrics)
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished GAIA evaluation. em_mean={em_mean:.3f}, latency_mean={latency_mean:.3f}, n_items={len(items)}"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def _resolve_binding(default_host: str, default_port: int) -> tuple[str, int]:
    env_host = os.getenv("HOST")
    env_port = os.getenv("AGENT_PORT")
    host = env_host or default_host
    port = int(env_port) if env_port else default_port
    return host, port


def start_green_agent(
    agent_name: str = "gaia_green_agent",
    host: str | None = None,
    port: int | None = None,
):
    print("Starting green agent...")
    resolved_host, resolved_port = _resolve_binding(
        host or "localhost",
        port if port is not None else 9001,
    )
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{resolved_host}:{resolved_port}"
    agent_card_dict["url"] = url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=GaiaGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=resolved_host, port=resolved_port)


def main():
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "9001"))
    start_green_agent("gaia_green_agent", host, port)


if __name__ == "__main__":
    main()
