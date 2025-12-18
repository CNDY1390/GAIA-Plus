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


def _normalize_raw_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw JSONL object into the GaiaItem schema.

    Supports:
    - Native GAIA-Plus lines with id/question/answer/level/meta.
    - NewDataset-style lines with task_id/Question/Final answer/Level
      plus nested Annotator/Efficiency/Evidence blocks.
    """
    # Native GAIA-Plus schema.
    if "id" in obj and "question" in obj and "answer" in obj:
        return {
            "id": obj["id"],
            "question": obj["question"],
            "answer": obj["answer"],
            "level": obj.get("level"),
            "meta": obj.get("meta"),
        }

    # NewDataset-style schema.
    if "task_id" in obj and "Question" in obj and "Final answer" in obj:
        meta: Dict[str, Any] = {}
        if "Annotator Metadata" in obj:
            meta["annotator"] = obj["Annotator Metadata"]
        if "Efficiency Metrics" in obj:
            meta["efficiency"] = obj["Efficiency Metrics"]
        if "Evidence Metrics" in obj:
            meta["evidence"] = obj["Evidence Metrics"]
        if meta:
            meta["source"] = "NewDataset"

        level_value = obj.get("Level")
        level_str = str(level_value) if level_value is not None else None

        return {
            "id": str(obj["task_id"]),
            "question": obj["Question"],
            "answer": obj["Final answer"],
            "level": level_str,
            "meta": meta or None,
        }

    # For any other schema we just pass through; GaiaItem validation will decide.
    return obj


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
                normalized = _normalize_raw_item(obj)
                item = GaiaItem(**normalized)
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


def _safe_parse_task_payload(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}
    except json.JSONDecodeError:
        return {"text": raw}


WHITE_URL_TAG_RE = re.compile(
    r"<white_agent_url>(.*?)</white_agent_url>", re.IGNORECASE | re.DOTALL
)
GAIA_DATA_PATH_TAG_RE = re.compile(
    r"<gaia_data_path>(.*?)</gaia_data_path>", re.IGNORECASE | re.DOTALL
)


def parse_task(user_input: str) -> Dict[str, Any]:
    """
    Parse the inbound task description.

    1) First, try strict JSON parsing.
    2) If JSON fails, fall back to extracting tags like
       <white_agent_url>...</white_agent_url> and
       <gaia_data_path>...</gaia_data_path> via regex.
    """
    raw_text = user_input or ""
    text = raw_text.strip()
    result: Dict[str, Any] = {
        "raw_text": raw_text,
        "payload": {},
        "white_urls": [],
        "gaia_data_path": None,
    }

    if not text:
        return result

    # 1) JSON branch
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            payload: Dict[str, Any] = parsed
        else:
            payload = {"data": parsed}
        result["payload"] = payload
        print("[green] inbound task payload (JSON):")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        # Let the structured extractor handle URL discovery.
        result["white_urls"] = extract_white_urls(None, payload)
        return result
    except json.JSONDecodeError:
        # 2) Non-JSON branch; use tag-based extraction.
        result["payload"] = {"text": raw_text}
        print("[green] inbound task payload (non-JSON, raw preview):")
        print(raw_text[:300])

        white_urls = [
            m.strip() for m in WHITE_URL_TAG_RE.findall(text) if m.strip()
        ]
        result["white_urls"] = white_urls

        m = GAIA_DATA_PATH_TAG_RE.search(text)
        if m:
            gaia_path = m.group(1).strip()
            if gaia_path:
                result["gaia_data_path"] = gaia_path

        return result


def extract_white_urls(request_dump: Dict[str, Any] | None, task_payload: Dict[str, Any]) -> List[str]:
    def _extract_from_obj(obj: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        for key in ["participants", "participant_agents", "agents", "white_agents"]:
            if key in obj and isinstance(obj[key], list):
                for entry in obj[key]:
                    if isinstance(entry, dict) and "url" in entry:
                        urls.append(entry["url"])
                    elif isinstance(entry, str) and entry.startswith("http"):
                        urls.append(entry)
        cfg = obj.get("config")
        if isinstance(cfg, dict):
            for key in ["white_agent_url", "white_url", "white"]:
                value = cfg.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    urls.append(value)
                elif isinstance(value, list):
                    urls.extend([v for v in value if isinstance(v, str) and v.startswith("http")])
        return urls

    # 1) try from request dump
    if request_dump:
        urls = _extract_from_obj(request_dump)
        if urls:
            return urls

    # 2) then from task payload
    urls = _extract_from_obj(task_payload)
    if urls:
        return urls

    raise RuntimeError("No white agent URL found in request/task payload.")


def _snapshot_env_for_report() -> Dict[str, Any]:
    """Collect non-sensitive environment variables for logging."""
    hidden_keys = {"OPENAI_API_KEY"}
    env_snapshot: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if any(s in key.upper() for s in hidden_keys):
            continue
        env_snapshot[key] = value
    return env_snapshot


async def ask_white_for_answer(
    white_agent_url: str, question: str, context_id: str | None = None
) -> tuple[str, str | None, Dict[str, Any]]:
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

    # Per-response HTML snapshot for later debugging.
    per_call_dir = OUTPUTS_DIR / "calls"
    per_call_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    html_path = per_call_dir / f"call_{ts}_{res_result.context_id or 'noctx'}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
        f.write("<title>GAIA-Plus White Call</title></head><body>")
        f.write("<h1>White agent call snapshot</h1>")
        f.write("<h2>Question</h2><pre>")
        f.write(question)
        f.write("</pre>")
        f.write("<h2>Answer</h2><pre>")
        f.write(answer)
        f.write("</pre>")
        f.write("<h2>Environment</h2><pre>")
        f.write(json.dumps(_snapshot_env_for_report(), indent=2, ensure_ascii=False))
        f.write("</pre>")
        f.write("</body></html>")

    return answer, res_result.context_id, _snapshot_env_for_report()


class GaiaGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        self.dataset_path = DEFAULT_DATA_PATH
        self.white_url = None
        self.retries = int(os.getenv("GREEN_RETRY", "1"))
        self.model = DEFAULT_MODEL

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("Green agent: Received a task, parsing...")

        user_input = context.get_user_input()
        print(f"[green] user_input_repr={user_input!r}")

        raw_request = getattr(context, "request", None)
        print(f"[green] inbound RequestContext.request type={type(raw_request)}")
        request_dump: Dict[str, Any] | None = None
        if raw_request is not None:
            if hasattr(raw_request, "model_dump"):
                try:
                    request_dump = raw_request.model_dump()  # type: ignore[assignment]
                    print("[green] inbound request model_dump:")
                    try:
                        print(json.dumps(request_dump, indent=2, ensure_ascii=False, default=str))
                    except TypeError:
                        print(request_dump)
                except Exception as e:  # noqa: BLE001
                    print(f"[green] ERROR while dumping request via model_dump: {e}")
            elif hasattr(raw_request, "__dict__"):
                request_dump = {
                    k: v for k, v in raw_request.__dict__.items() if not k.startswith("_")
                }
                print("[green] inbound request __dict__:")
                try:
                    print(json.dumps(request_dump, indent=2, ensure_ascii=False, default=str))
                except TypeError:
                    print(request_dump)

        parsed_task = parse_task(user_input)
        task_payload = parsed_task.get("payload") or {}

        white_urls: List[str] = list(parsed_task.get("white_urls") or [])
        if not white_urls and request_dump is not None:
            try:
                white_urls = extract_white_urls(request_dump, task_payload)
            except RuntimeError:
                white_urls = []

        if not white_urls:
            raw_preview = (parsed_task.get("raw_text") or "")[:300]
            raise RuntimeError(
                f"[green] No white_agent_url found in task. raw_text_preview={raw_preview!r}"
            )

        self.white_url = white_urls[0]

        print(f"[green] resolved white URLs={white_urls}")

        # Connectivity check to ensure white agent can receive A2A call.
        ping_question = "[green] connectivity check: please reply with 'pong'."
        print(f"[green] sending connectivity ping to white url={self.white_url!r}")
        ping_answer, _, _ = await ask_white_for_answer(
            self.white_url, ping_question, context_id=None
        )
        print(f"[green] connectivity ping answer_preview={ping_answer[:120]!r}")

        # Env validation
        self.dataset_path = ensure_required_envs()
        
        if not self.white_url:
            error_msg = "white_agent_url not provided in task config."
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
        baseline_in_tokens_total = 0
        baseline_out_tokens_total = 0
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
                    pred, ctx_id, _ = await ask_white_for_answer(
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

            baseline_in_tokens = None
            baseline_out_tokens = None
            if isinstance(item.meta, dict):
                efficiency = item.meta.get("efficiency")
                if isinstance(efficiency, dict):
                    in_tokens = efficiency.get("Baseline_Input_Tokens")
                    out_tokens = efficiency.get("Baseline_Output_Tokens")
                    if isinstance(in_tokens, (int, float)):
                        baseline_in_tokens = int(in_tokens)
                        baseline_in_tokens_total += baseline_in_tokens
                    if isinstance(out_tokens, (int, float)):
                        baseline_out_tokens = int(out_tokens)
                        baseline_out_tokens_total += baseline_out_tokens

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
                    "baseline_input_tokens": baseline_in_tokens,
                    "baseline_output_tokens": baseline_out_tokens,
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
            "baseline_input_tokens_total": baseline_in_tokens_total,
            "baseline_output_tokens_total": baseline_out_tokens_total,
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
            f.write("id,pred,gold,em,latency,baseline_input_tokens,baseline_output_tokens\n")
            for row in per_item:
                f.write(
                    f"{row['id']},"
                    f"{str(row['pred']).replace(',',';')},"
                    f"{str(row['gold']).replace(',',';')},"
                    f"{row['em']},"
                    f"{row['latency']},"
                    f"{row['baseline_input_tokens'] if row['baseline_input_tokens'] is not None else ''},"
                    f"{row['baseline_output_tokens'] if row['baseline_output_tokens'] is not None else ''}\n"
                )

        # Static HTML report for quick inspection.
        report_path = OUTPUTS_DIR / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n")
            f.write("<title>GAIA-Plus Evaluation Report</title>\n")
            f.write(
                "<style>"
                "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
                "margin:24px;background:#0b1120;color:#e5e7eb;}"
                "h1{font-size:24px;margin-bottom:8px;}"
                "h2{font-size:18px;margin-top:24px;margin-bottom:8px;}"
                ".summary{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:16px;}"
                ".card{background:#020617;border-radius:8px;padding:12px 16px;"
                "box-shadow:0 10px 15px -3px rgba(15,23,42,0.4);}"
                ".label{font-size:11px;text-transform:uppercase;color:#9ca3af;letter-spacing:.08em;}"
                ".value{font-size:18px;font-weight:600;margin-top:2px;}"
                "table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px;}"
                "th,td{padding:6px 8px;border-bottom:1px solid #1f2937;vertical-align:top;}"
                "th{position:sticky;top:0;background:#020617;color:#d1d5db;text-align:left;}"
                "tr:nth-child(even){background:#020617;}"
                "tr:nth-child(odd){background:#020617;}"
                ".em-good{color:#4ade80;font-weight:600;}"
                ".em-bad{color:#f97373;font-weight:600;}"
                ".mono{font-family:ui-monospace,Menlo,Monaco,Consolas,monospace;}"
                "</style>\n</head>\n<body>\n"
            )
            f.write("<h1>GAIA-Plus Evaluation Report</h1>\n")
            f.write("<div class='summary'>\n")
            f.write(
                f"<div class='card'><div class='label'>Exact match (mean)</div>"
                f"<div class='value'>{em_mean:.3f}</div></div>\n"
            )
            f.write(
                f"<div class='card'><div class='label'>Latency mean (s)</div>"
                f"<div class='value'>{latency_mean:.3f}</div></div>\n"
            )
            f.write(
                f"<div class='card'><div class='label'>Items</div>"
                f"<div class='value'>{len(items)}</div></div>\n"
            )
            f.write(
                f"<div class='card'><div class='label'>Skipped lines</div>"
                f"<div class='value'>{skipped}</div></div>\n"
            )
            f.write(
                f"<div class='card'><div class='label'>Baseline tokens (input / output)</div>"
                f"<div class='value'>{baseline_in_tokens_total} / {baseline_out_tokens_total}</div></div>\n"
            )
            f.write("</div>\n")

            f.write("<h2>Per-question details</h2>\n")
            f.write("<div style='max-height:480px;overflow:auto;border-radius:8px;border:1px solid #1f2937;'>\n")
            f.write(
                "<table>\n<thead><tr>"
                "<th>ID</th><th>Level</th><th>EM</th><th>Latency (s)</th>"
                "<th>Baseline tokens (in/out)</th><th>Pred</th><th>Gold</th>"
                "</tr></thead>\n<tbody>\n"
            )
            for row in per_item:
                em_val = row["em"]
                em_class = "em-good" if em_val == 1.0 else "em-bad"
                baseline_pair = ""
                if row["baseline_input_tokens"] is not None or row["baseline_output_tokens"] is not None:
                    baseline_pair = f"{row['baseline_input_tokens'] or 0}/{row['baseline_output_tokens'] or 0}"
                pred_preview = (str(row["pred"]) or "")[:160]
                gold_preview = (str(row["gold"]) or "")[:160]
                f.write("<tr>")
                f.write(f"<td class='mono'>{row['id']}</td>")
                f.write(f"<td>{row['level'] or ''}</td>")
                f.write(f"<td class='{em_class}'>{em_val:.0f}</td>")
                f.write(f"<td>{row['latency']:.3f}</td>")
                f.write(f"<td class='mono'>{baseline_pair}</td>")
                f.write(f"<td>{pred_preview}</td>")
                f.write(f"<td>{gold_preview}</td>")
                f.write("</tr>\n")
            f.write("</tbody>\n</table>\n</div>\n")
            f.write("</body>\n</html>\n")

        print("Green agent: Evaluation complete.", metrics)
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished GAIA evaluation. em_mean={em_mean:.3f}, latency_mean={latency_mean:.3f}, n_items={len(items)}"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(
    agent_name: str = "gaia_green_agent",
    host: str | None = None,
    port: int | None = None,
) -> None:
    # Resolve binding purely from env + optional explicit args
    env_host = os.getenv("HOST")
    env_port = os.getenv("AGENT_PORT")
    resolved_host = host or env_host or "0.0.0.0"
    resolved_port = port or (int(env_port) if env_port else 9001)

    print("Starting green agent...")
    print(
        f"[green] env HOST={env_host!r}, AGENT_PORT={env_port!r}, "
        f"AGENT_URL={os.getenv('AGENT_URL')!r}, "
        f"CLOUDRUN_HOST={os.getenv('CLOUDRUN_HOST')!r}, "
        f"HTTPS_ENABLED={os.getenv('HTTPS_ENABLED')!r}, "
        f"resolved_host={resolved_host!r}, resolved_port={resolved_port!r}"
    )

    agent_card_dict = load_agent_card_toml(agent_name)
    public_url = os.getenv("AGENT_URL") or f"http://{resolved_host}:{resolved_port}"
    agent_card_dict["url"] = public_url  # complete all required card fields

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
