"""GAIA-Plus white agent - minimal LLM wrapper with dummy mode."""

import os
import uvicorn
import dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion

dotenv.load_dotenv()


def ensure_required_envs():
    if os.getenv("WHITE_MODE") == "dummy_correct":
        print("[white] WHITE_MODE=dummy_correct (OPENAI_API_KEY not required).")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("[white] ERROR: OPENAI_API_KEY is required unless WHITE_MODE=dummy_correct.")
        raise SystemExit(1)
    if not os.getenv("OPENAI_MODEL"):
        print("[white] OPENAI_MODEL not set; defaulting to gpt-4o-mini.")
    else:
        print(f"[white] Using OPENAI_MODEL={os.getenv('OPENAI_MODEL')}")
    if os.getenv("OPENAI_BASE_URL"):
        print(f"[white] OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')}")


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="gaia_answering",
        name="GAIA short answering",
        description="Given a question, reply with only the final short answer.",
        tags=["gaia-plus"],
        examples=[],
    )
    card = AgentCard(
        name="gaia_white_agent",
        description="Minimal GAIA-Plus baseline (LLM passthrough).",
        url=url,
        version="0.2.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


def _maybe_dummy_answer(prompt: str) -> str | None:
    if os.getenv("WHITE_MODE") != "dummy_correct":
        return None
    # When in dummy mode, if prompt contains GOLD: <answer>, echo it; else echo "dummy".
    marker = "GOLD:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip().splitlines()[0]
    return "dummy"


class GaiaWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        ensure_required_envs()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.provider = os.getenv("OPENAI_PROVIDER", "openai")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        print(
            f"[white] starting with model={self.model}, provider={self.provider}, base_url={self.base_url or 'default'}"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        print(
            f"[white] recv ctx={context.context_id} len={len(user_input)} preview={user_input[:120]!r}"
        )

        # Dummy correct mode for quick EM self-check.
        dummy_ans = _maybe_dummy_answer(user_input)
        if dummy_ans is not None:
            print("[white] dummy_correct mode hit, echoing gold.")
            await event_queue.enqueue_event(
                new_agent_text_message(dummy_ans, context_id=context.context_id)
            )
            return

        system_prompt = (
            "You are a GAIA short-answer agent. "
            "Read the question and respond with ONLY the final short answer. "
            "Do not add explanations, punctuation, or multiple words unless required."
        )
        response = completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            model=self.model,
            custom_llm_provider=self.provider,
            base_url=self.base_url,
            temperature=0.0,
        )
        next_message = response.choices[0].message.model_dump()  # type: ignore
        print(
            f"[white] send ctx={context.context_id} ans_preview={next_message['content'][:120]!r}"
        )
        await event_queue.enqueue_event(
            new_agent_text_message(
                next_message["content"], context_id=context.context_id
            )
        )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(
    agent_name: str = "gaia_white_agent",
    host: str | None = None,
    port: int | None = None,
) -> None:
    # Resolve binding purely from env + optional explicit args
    env_host = os.getenv("HOST")
    env_port = os.getenv("AGENT_PORT")
    resolved_host = host or env_host or "0.0.0.0"
    resolved_port = port or (int(env_port) if env_port else 9002)

    print("Starting white agent...")
    print(
        f"[white] env HOST={env_host!r}, AGENT_PORT={env_port!r}, "
        f"AGENT_URL={os.getenv('AGENT_URL')!r}, "
        f"CLOUDRUN_HOST={os.getenv('CLOUDRUN_HOST')!r}, "
        f"HTTPS_ENABLED={os.getenv('HTTPS_ENABLED')!r}, "
        f"resolved_host={resolved_host!r}, resolved_port={resolved_port!r}"
    )

    public_url = os.getenv("AGENT_URL") or f"http://{resolved_host}:{resolved_port}"
    card = prepare_white_agent_card(public_url)

    request_handler = DefaultRequestHandler(
        agent_executor=GaiaWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=resolved_host, port=resolved_port)


def main():
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "9002"))
    start_white_agent("gaia_white_agent", host, port)


if __name__ == "__main__":
    main()
