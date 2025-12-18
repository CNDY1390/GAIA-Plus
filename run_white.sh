#!/usr/bin/env bash
set -euo pipefail

export ROLE="white"
export AGENT_PORT="9002"
export PYTHONPATH="src"
export CLOUDRUN_HOST="noble-receipt-resolved-trademarks.trycloudflare.com"
export HTTPS_ENABLED="true"
export PORT="8011"
export HOST="0.0.0.0"

uv run agentbeats run_ctrl


