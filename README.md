# GAIA-Plus Agentified Benchmark

GAIA/GAIA-Plus benchmark on A2A: **Green orchestrator** reads GAIA JSONL, calls **White baseline** (LLM passthrough or dummy self-check), scores normalized exact match (EM) + latency, and writes metrics.

## Overview
- Green agent: orchestrates GAIA items, calls white, scores EM/latency, writes outputs.
- White agent: minimal LLM wrapper with `dummy_correct` mode for self-check.
- Ready for AgentBeats v2: exposes agent-card, health, and accepts white URL/config via task input.

## Project structure
```
├─main.py                     # CLI entry (launch/green/white)
├─pyproject.toml              # Project metadata
├─uv.lock                     # Locked dependencies
├─data/
│   └─gaia_plus.jsonl         # Sample GAIA-Plus data (10 rows)
├─outputs/                    # Metrics written after runs
│   ├─metrics.json
│   └─metrics.csv
└─src/
    ├─launcher.py             # Starts green/white processes
    ├─my_util/
    │   └─my_a2a.py           # Helper for A2A integration
    ├─green_agent/
    │   ├─agent.py            # Orchestrator: loads data, calls white, scores
    │   ├─gaia_green_agent.toml
    │   └─tau_green_agent.toml
    └─white_agent/
        └─agent.py            # Baseline white agent (LLM or dummy_correct)
```

## Requirements
- Python 3.11/3.12 recommended (works with uv; lockfile is 3.13-compatible).
- [`uv`](https://github.com/astral-sh/uv) for dependency/env management.
- Windows: run in PowerShell; uv handles venv creation automatically.

## Setup
```bash
uv sync
```
Create a `.env` (do **not** commit it; already in `.gitignore`):
```
OPENAI_API_KEY=sk-...             # required unless WHITE_MODE=dummy_correct
OPENAI_MODEL=gpt-4o-mini          # default shown at startup
OPENAI_BASE_URL=...               # optional (Azure / proxy)
OPENAI_PROVIDER=openai            # optional (litellm provider)
GAIA_PLUS_DATA=data/gaia_plus.jsonl
WHITE_AGENT_URL=http://localhost:9002
WHITE_MODE=dummy_correct          # optional: white echoes GOLD for self-check
GREEN_RETRY=1                     # optional retry for white calls
```

## Run locally
- One-shot (starts green+white, runs all items):
```bash
uv run python main.py launch
```
- Individually:
```bash
uv run python main.py white   # baseline only
uv run python main.py green   # orchestrator only
```
Expected: prints `em_mean`, `latency_mean`, `n_items`, writes outputs under `outputs/`.

## Outputs
- `outputs/metrics.json`: `summary` + `details` per item.
  - Summary fields: `em_mean`, `latency_mean`, `n_items`, `n_total_lines`, `n_skipped`, `n_success`, `n_failed`, `run_id`, `started_at`, `finished_at`, `total_duration`, `git_commit`.
  - Details per item: `id`, `question`, `pred`, `gold`, `em`, `latency`, `context_id`, `error`, `level`.
- `outputs/metrics.csv`: lightweight view (`id,pred,gold,em,latency`).

## Deploy / AgentBeats v2
1) Start locally (or on a VM):
```bash
uv run python main.py green   # listens on 9001
uv run python main.py white   # listens on 9002
```
2) Expose both via Cloudflare (two tunnels):
```bash
cloudflared tunnel --url http://localhost:9001   # green public URL
cloudflared tunnel --url http://localhost:9002   # white public URL
```
3) Register agents in AgentBeats v2 UI:
   - Green agent: use its public URL; card served at `/.well-known/agent-card.json`.
   - White agent: same.
4) Launch an assessment/battle in AgentBeats UI:
   - Choose green as the task owner; provide the white agent URL (public tunnel) and optional `gaia_data_path` if overriding default.
   - Start assessment; platform calls green, which calls white per item; metrics stored by platform.
5) Copy the assessment link for submission.

## Troubleshooting
- Missing key/model: startup prints clear errors and exits; set `OPENAI_API_KEY` (unless `dummy_correct`) and `OPENAI_MODEL`.
- Base URL/proxy issues: set `OPENAI_BASE_URL`; ensure reachable from runner.
- Agent card unreachable: verify `/.well-known/agent-card.json` via the public URL (cloudflared tunnel).
- Port conflicts: change ports in `start_green_agent` / `start_white_agent` or set different tunnels.
- uv/Python mismatch: install Python 3.11/3.12; rerun `uv sync`.
- No outputs: check that `GAIA_PLUS_DATA` exists and is valid JSONL; warnings indicate skipped rows.

## Data schema (simplified)
Small sample (10 rows) in `data/gaia_plus.jsonl`:
```
{"id": "...", "level": "L1", "question": "...", "answer": "...", "meta": {...}}
```
Required fields: `id`, `question`, `answer`. Optional: `level`, `meta`. Rows missing required fields are skipped with a warning. To swap in full GAIA data, replace the JSONL file or point `GAIA_PLUS_DATA` to your path (same keys recommended). Answer normalization: lowercase, trim, collapse spaces, strip leading/trailing punctuation.
