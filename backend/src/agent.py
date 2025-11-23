import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from datetime import datetime


@dataclass
class OrderState:
    drinkType: str | None = None   # e.g. "latte", "cappuccino"
    size: str | None = None        # e.g. "small", "medium", "large"
    milk: str | None = None        # e.g. "regular", "oat milk"
    extras: list[str] = field(default_factory=list)  # e.g. ["extra shot", "whipped cream"]
    name: str | None = None        # customer's name


logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
                You are a friendly barista for the coffee brand BrewBuddies Coffee.

                Your job is to take a coffee order and fill in the following fields:

                - drinkType (e.g. latte, cappuccino, americano, cold brew)
                - size (small, medium, large)
                - milk (e.g. regular, skim, oat, almond, soy)
                - extras (list of strings: extra shot, syrup, whipped cream, etc. can be empty)
                - name (customer's name for the cup)

                Ask clarifying follow-up questions until you are confident all required fields
                (drinkType, size, milk, name) are filled.

                Use the tools you have to:
                - record each part of the order into the order state
                - optionally record extras
                - when the order is fully captured, confirm the order, and then finalize it.

                Always speak naturally like a kind barista.
                Do NOT invent values: if something is unclear, ask the customer.
                """
        )
    async def on_enter(self) -> None:
        # This runs when the call/session starts
        await self.session.generate_reply(
            instructions=(
                "Greet the customer warmly as a barista at BrewBuddies Coffee "
                "and ask what they'd like to order."
            )
        )
    @function_tool()
    async def set_drink_type(
        self,
        context: RunContext[OrderState],
        drink_type: str,
    ) -> None:
        """Use this when the customer tells you what drink they want (latte, cappuccino, etc.)."""
        context.userdata.drinkType = drink_type
        await self._maybe_continue_or_finalize(context)
    @function_tool()
    async def set_size(
        self,
        context: RunContext[OrderState],
        size: str,
    ) -> None:
        """Use this when the customer specifies the drink size (small, medium, large)."""
        context.userdata.size = size
        await self._maybe_continue_or_finalize(context)
    @function_tool()
    async def set_milk(
        self,
        context: RunContext[OrderState],
        milk: str,
    ) -> None:
        """Use this when the customer specifies their milk preference (regular, oat, etc.)."""
        context.userdata.milk = milk
        await self._maybe_continue_or_finalize(context)
    @function_tool()
    async def add_extra(
        self,
        context: RunContext[OrderState],
        extra: str,
    ) -> None:
        """
        Use this when the customer asks for any extras or customizations
        like extra shot, extra hot, sugar, syrup, whipped cream, etc.
        If they say they don't want extras, call this with 'no extras'.
        """
        context.userdata.extras.append(extra)
        await self._maybe_continue_or_finalize(context)
    @function_tool()
    async def set_name(
        self,
        context: RunContext[OrderState],
        name: str,
    ) -> None:
        """Use this when the customer tells you their name for the cup."""
        context.userdata.name = name
        await self._maybe_continue_or_finalize(context)

    async def _maybe_continue_or_finalize(
        self,
        context: RunContext[OrderState],
    ) -> None:
        """
        Check which fields are missing and either:
        - Ask clarifying questions, or
        - Finalize the order and save it to JSON.
        """
        order = context.userdata
        missing_parts: list[str] = []

        if not order.drinkType:
            missing_parts.append(
                "what drink they want (e.g. latte, cappuccino, americano)"
            )
        if not order.size:
            missing_parts.append(
                "what size they want (small, medium, or large)"
            )
        if not order.milk:
            missing_parts.append(
                "their milk preference (regular, skim, oat, etc.)"
            )
        if not order.name:
            missing_parts.append("their name for the cup")

        # extras are optional – empty list is fine
        if missing_parts:
            await context.session.generate_reply(
                instructions=(
                    "You have already collected some parts of the order. "
                    "Now politely ask follow-up questions to collect the missing details: "
                    + ", ".join(missing_parts)
                    + ". Ask just one or two things at a time, in a friendly way."
                )
            )
            return

        # All required fields present; finalize
        await self._finalize_order(context)

    async def _finalize_order(
        self,
        context: RunContext[OrderState],
    ) -> None:
        """Write the order to a JSON file and verbally summarize it to the user."""
        order = context.userdata

        # Convert dataclass to a plain dict for JSON
        order_dict = asdict(order)
        order_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Save/append to backend/src/orders.json
        orders_path = Path(__file__).parent / "orders.json"

        existing: list[dict] = []
        if orders_path.exists():
            try:
                existing = json.loads(orders_path.read_text())
            except json.JSONDecodeError:
                existing = []

        existing.append(order_dict)
        orders_path.write_text(json.dumps(existing, indent=2))
        
        # Build a human-friendly summary
        summary_str = (
            f"{order.size or ''} {order.drinkType or ''} "
            f"with {order.milk or ''} milk"
        ).strip()

        extras_part = ""
        if order.extras:
            extras_part = " with " + ", ".join(order.extras)

        await context.session.generate_reply(
            instructions=(
                "Let the customer know their order is complete. "
                f"Summarize it clearly as: {summary_str}{extras_part}, "
                f"under the name {order.name}. "
                "Confirm that everything looks right and tell them their drink will be ready shortly."
            )
        )






    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession[OrderState](
        userdata=OrderState(),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
