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
Copy `.env.example` to `.env` (already ignored) and fill in your keys:
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

## Local development (one-shot launcher)
- Launches green+white locally, runs GAIA sample, writes metrics:
```bash
uv run python main.py launch
```
- Individual agents for debugging (uses env HOST/AGENT_PORT defaults):
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

## AgentBeats v2 Remote deployment (Controller)
Remote mode expects a publicly reachable controller URL per agent. The platform injects `HOST` (default `0.0.0.0`) and `AGENT_PORT` when it boots your process; our agents read these env vars automatically.

### AgentBeats Remote Evaluation

We expose the Green (assessment controller) agent via a public endpoint
(e.g., ngrok) for AgentBeats remote evaluation.

The White agent runs on the same host and is accessed by the Green agent
via a local URL (e.g., http://localhost:9002). This design follows the
controller–worker pattern and avoids exposing multiple public endpoints.

AgentBeats interacts only with the Green agent. All benchmark execution,
including calls to the White agent and metric aggregation, occurs inside
the Green agent process.

### A) Local controller + Cloudflare tunnel
1. Start each agent via the Linux scripts (they run `uv` with the configured environment defaults and respect controller-provided overrides):
   ```bash
   ./run_white.sh   # default port 9002
   ./run_green.sh   # default port 9001
   ```
2. For local testing, export your own ports before running (optional):
   ```bash
   HOST=127.0.0.1 AGENT_PORT=9101 ./run_white.sh
   ```
3. Expose each agent with separate tunnels:
   ```bash
   cloudflared tunnel --url http://localhost:9001   # green controller URL
   cloudflared tunnel --url http://localhost:9002   # white controller URL
   ```

### B) Cloud VM (direct HTTPS)
1. Provision a VM (Ubuntu LTS recommended) with ports 80/443 opened.
2. Install `uv`, `python3.11+`, repo, and `.env`.
3. Run `./run_green.sh` and `./run_white.sh` in tmux/screen; use a reverse proxy (nginx / Caddy / Traefik) for HTTPS termination and to map stable URLs to `HOST=0.0.0.0` with the injected `AGENT_PORT`.

### AgentBeats UI checklist
1. Create two remote agents (green + white).
2. For each card, paste the corresponding public controller URL (from Cloudflare or your VM) into the Controller URL field.
3. Click **Check Again** until `Controller Reachable: Yes`.
4. Start an assessment: choose green as task owner, paste the white controller URL into the run configuration (or rely on default `WHITE_AGENT_URL`).
5. Wait for completion, copy the assessment link, and submit for Q11.

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
