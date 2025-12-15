#!/usr/bin/env bash
set -e

export HOST="${HOST:-0.0.0.0}"
export AGENT_PORT="${AGENT_PORT:-9002}"
export PYTHONPATH="src"

uv run python -m white_agent.agent
