#!/usr/bin/env bash
set -e

# If running on Windows (Git Bash / MSYS / Cygwin), delegate to run.cmd
case "$(uname -s 2>/dev/null || echo "")" in
  MINGW*|MSYS*|CYGWIN*)
    # run.cmd is Windows-native entrypoint for controller on Windows
    cmd.exe /c "%~dp0run.cmd"
    exit $?
    ;;
esac

export HOST="${HOST:-0.0.0.0}"
if [ -z "${AGENT_PORT:-}" ]; then
  echo "AGENT_PORT not set"
  exit 1
fi
export PYTHONPATH="src"

if [ "${ROLE:-}" = "green" ]; then
  uv run python -m green_agent.agent
elif [ "${ROLE:-}" = "white" ]; then
  uv run python -m white_agent.agent
else
  echo "ROLE must be green or white"
  exit 1
fi
