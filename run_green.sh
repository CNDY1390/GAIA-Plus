#!/usr/bin/env bash
set -euo pipefail

export ROLE="green"
export AGENT_PORT="9001"
export PYTHONPATH="src"
export CLOUDRUN_HOST="scenarios-liquid-hierarchy-isa.trycloudflare.com"
export HTTPS_ENABLED="true"
export PORT="8010"
export HOST="0.0.0.0"

uv run agentbeats run_ctrl

