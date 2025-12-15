"""CLI entry point for GAIA-Plus agentified benchmark."""

import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

app = typer.Typer(help="GAIA-Plus A2A benchmark - Green (orchestrator) + White baseline")


@app.command()
def green():
    """Start the green agent (GAIA benchmark orchestrator)."""
    start_green_agent()


@app.command()
def white():
    """Start the white agent (LLM baseline)."""
    start_white_agent()


@app.command()
def launch():
    """Launch the complete GAIA evaluation workflow (local green+white)."""
    asyncio.run(launch_evaluation())


if __name__ == "__main__":
    app()
