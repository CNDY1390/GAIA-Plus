## GAIA-Plus: Green–White Agent Benchmark on AgentBeats

Small GAIA-style short-answer benchmark wrapped as a green–white agent pair for AgentBeats. The green agent reads a JSONL dataset, calls a white LLM agent over HTTP, scores normalized exact match + latency + approximate token usage, and writes machine-readable metrics and static HTML reports.

### Project layout

```text
main.py                 CLI (green / white / local launcher)
pyproject.toml          Project metadata
uv.lock                 Locked dependencies
data/
  gaia_plus.jsonl       GAIA-Plus mini benchmark (20 Q&A items)
src/
  launcher.py           Local URL-driven launcher (no AgentBeats)
  my_util/my_a2a.py     A2A client helper
  green_agent/agent.py  Green orchestrator (this repo’s main entry)
  green_agent/gaia_green_agent.toml  Agent card for green
  white_agent/agent.py  White baseline (LLM or dummy_correct)
ctrl_green.ps1          Windows controller script for green
ctrl_white.ps1          Windows controller script for white
setup_all.ps1           One-shot Windows setup + controller bootstrap
run.sh                  POSIX entry (ROLE=green|white)
README_RUN.md           Short run cookbook
```

### Requirements

- Python 3.11 or 3.12 (managed via `uv`)
- `uv` installed (`pip install uv` or official installer)
- Windows + PowerShell for local/controller scripts (evaluation cluster uses `run.sh`)

Install dependencies from the project root:

```bash
uv sync
```

### Configuration

The green and white agents are driven entirely by environment variables (no hard-coded URLs or ports). The most important ones are:

```text
OPENAI_API_KEY=sk-...             # required unless WHITE_MODE=dummy_correct
OPENAI_MODEL=gpt-4o-mini          # default if unset
OPENAI_BASE_URL=...               # optional (proxy / Azure endpoint)
OPENAI_PROVIDER=openai            # optional (litellm provider name)

GAIA_PLUS_DATA=data/gaia_plus.jsonl   # JSONL dataset path used by green
GREEN_RETRY=1                         # retry count for white calls

HOST=0.0.0.0                     # bind host for agents (set by controller)
AGENT_PORT=9001                  # port for green (9002 for white)
AGENT_URL=...                    # public URL for agent-card (set by controller)
WHITE_MODE=dummy_correct         # optional: white echoes gold answers for sanity check
```

All other values (including controller URLs, tunnels, and cluster-specific settings) are handled in `ctrl_green.ps1`, `ctrl_white.ps1`, and `run.sh`.

### Data: GAIA-Plus mini benchmark

The benchmark lives in `data/gaia_plus.jsonl`. Each line is a JSON object:

```text
{"id": "agentbeats_001", "level": "L1", "question": "...", "answer": "...", "meta": {...}}
```

- `id` (string): unique identifier.
- `question` (string): natural-language GAIA-style question.
- `answer` (string): short, canonical gold answer (number / short phrase).
- `level` (string, optional): difficulty tag.
- `meta` (object, optional): extra metadata. For some tasks adapted from GAIA-style `NewDataset.jsonl`, this includes
  - `meta.efficiency.Baseline_Input_Tokens`
  - `meta.efficiency.Baseline_Output_Tokens`
  - `meta.evidence.Required_Sources` / `Required_Facts`

Rows missing any of `id`, `question`, or `answer` are skipped with a warning when the green agent loads the dataset.

### Green agent behavior

`src/green_agent/agent.py` implements the green agent as an A2A HTTP service:

- On each incoming task from AgentBeats, it parses either JSON or a plain-text prompt containing a `<white_agent_url>...</white_agent_url>` tag to locate the white agent.
- It pings the white agent once with a connectivity check prompt (“please reply with 'pong'”) to verify that the A2A call succeeds.
- It validates required environment variables and the GAIA-Plus dataset path.
- It iterates over all items in `GAIA_PLUS_DATA`, synthesizes a prompt that asks for “only the final short answer”, and sends it to the white agent.
- For each white response, it:
  - extracts the text answer and computes normalized exact match (0.0 or 1.0);
  - measures per-item latency in seconds;
  - pulls baseline token counts from `meta.efficiency` if present, otherwise approximates them from the answer length with a small per-item perturbation (and a minimum of 5 tokens);
  - logs an item record into memory.
- After the dataset is exhausted, it aggregates:

  - `em_mean` (average exact match),
  - `latency_mean`,
  - counts of items, skipped rows, successes, and failures,
  - totals of baseline input and output tokens,
  - run identifiers and timestamps.

The run’s artifacts are written under a timestamped directory, e.g. `outputs/20251218T123456Z/`:

- `metrics.json`: `{"summary": {...}, "details": [...]}`
- `metrics.csv`: per-item flat view (`id,pred,gold,em,latency,baseline_input_tokens,baseline_output_tokens`)
- `report.html`: static dark-themed summary page with key metrics and a scrollable per-item table
- `calls/call_<timestamp>_<context_id>.html`: one HTML snapshot per white call containing the exact question, white answer, and a non-sensitive environment-variable dump for debugging.

### White agent behavior

`src/white_agent/agent.py` implements a minimal LLM wrapper:

- In normal mode, it calls a model via `litellm.completion` with a short system prompt that enforces “ONLY the final short answer” and deterministic decoding (`temperature=0.0`).
- In `WHITE_MODE=dummy_correct`, it looks for a `GOLD:` marker in the user message and simply echoes the gold answer; this is useful for checking that the green agent’s EM and logging logic behave as expected.
- It exposes an A2A HTTP interface with a simple skill card and supports streaming texts back to the green agent.

### Local development (one-shot launcher)

For local end-to-end testing without AgentBeats, use the CLI in `main.py`:

```bash
uv run python main.py white   # start white on HOST/AGENT_PORT (default 0.0.0.0:9002)
uv run python main.py green   # start green on HOST/AGENT_PORT (default 0.0.0.0:9001)

# or launch both + one evaluation in a single command:
uv run python main.py launch
```

The `launch` subcommand reads `GREEN_URL`, `WHITE_URL`, and `GAIA_PLUS_DATA` from the environment, verifies readiness of both agents, sends a JSON config task to the green agent, and then waits for `outputs/<run_timestamp>/metrics.json` to appear, printing a short summary (EM, latency, item count) to the console.

### AgentBeats integration (controllers)

On Windows, controllers and tunnels are scripted in `README_RUN.md`, `ctrl_green.ps1`, `ctrl_white.ps1`, and `setup_all.ps1`:

1. Use `setup_all.ps1` once to prepare `uv`, dependencies, work directories, and cloudflared tunnels.
2. Start green and white controllers with:

   ```powershell
   .\ctrl_green.ps1   # runs agentbeats run_ctrl for green
   .\ctrl_white.ps1   # runs agentbeats run_ctrl for white
   ```

   Each controller runs in its own working directory under `.ab-work/`, uses `run.sh` as the starting command, and forwards cloudflared HTTPS to the local agent.

3. In the AgentBeats UI, register the green controller URL only (the white agent stays on localhost behind the controller). The platform will send tasks to the green agent, which discovers and calls the white agent using the URL embedded in the task input.

Full, step-by-step screenshots and exact PowerShell commands are documented in `README_RUN.md` and `WORKFLOW.md` in this repo.

### Reproducing a GAIA-Plus run

1. Ensure `GAIA_PLUS_DATA` points to `data/gaia_plus.jsonl`.
2. Start a white agent (either dummy-correct or a real LLM configuration).
3. Start the green agent and send it a task that contains the white agent URL (e.g., via the local launcher or AgentBeats).
4. After the run finishes, open the newest directory under `outputs/` and inspect:

   - `metrics.json` and `metrics.csv` for numeric results,
   - `report.html` for a human-friendly summary,
   - `calls/` for individual question–answer snapshots.

These artifacts are the same ones we use for grading, debugging, and the demo video in Q9. 
