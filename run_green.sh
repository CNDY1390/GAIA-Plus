#!/usr/bin/env bash
set -euo pipefail

export HOST="${HOST:-0.0.0.0}"
: "${AGENT_PORT:?AGENT_PORT not set}"
export PYTHONPATH="src"

exec uv run python -m green_agent.agent
