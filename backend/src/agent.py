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

from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# -----------------------------
# WELLNESS STATE
# -----------------------------
@dataclass
class WellnessState:
    last_mood: str | None = None
    last_energy: str | None = None
    last_goals: list[str] = field(default_factory=list)


logger = logging.getLogger("agent")

load_dotenv(".env.local")

# -----------------------------
# WELLNESS AGENT
# -----------------------------

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a gentle, supportive Health & Wellness Companion.
Your job is to guide a short daily check-in with the user.

RULES:
- You are NOT a medical expert.
- Do NOT diagnose or give medical claims.
- Keep everything simple, supportive, grounded, and practical.

CHECK-IN FLOW:
1. Ask about mood and energy.
2. Ask what is stressing them (optional).
3. Ask for 1–3 goals/intentions for today.
4. Offer simple practical advice.
5. Summarize mood + goals.
6. Save the check-in to wellness_log.json.
7. Next time: mention something from the last check-in.
"""
        )

    # -----------------------------
    # SESSION START MEMORY
    # -----------------------------
    async def on_enter(self) -> None:
        """Triggered when user joins session."""
        path = Path("wellness_log.json").resolve()

        if path.exists():
            try:
                logs = json.loads(path.read_text())
                if logs:
                    last = logs[-1]
                    msg = (
                        f"Last time, you said your mood was '{last.get('mood')}'. "
                        "How are you feeling today?"
                    )
                else:
                    msg = "How are you feeling today?"
            except:
                msg = "How are you feeling today?"
        else:
            msg = "How are you feeling today?"

        await self.session.generate_reply(instructions=msg)

    # -----------------------------
    # TOOLS
    # -----------------------------
    @function_tool()
    async def record_mood(
        self,
        context: RunContext[WellnessState],
        mood: str,
        energy: str,
    ):
        """Record today's mood & energy."""
        context.userdata.last_mood = mood
        context.userdata.last_energy = energy

        await context.session.generate_reply(
            instructions="Thank you. What are 1–3 things you'd like to get done today?"
        )

    @function_tool()
    async def record_goals(
        self,
        context: RunContext[WellnessState],
        goals: list[str],
    ):
        """Record today's goals."""
        context.userdata.last_goals = goals

        await context.session.generate_reply(
            instructions="Got it. Here's a small suggestion for the day:"
        )

        await context.session.generate_reply(
            instructions="Try breaking tasks into small steps, and take short breaks if you feel overwhelmed."
        )

        await self._finalize_checkin(context)

    # -----------------------------
    # RETURN LAST CHECK-IN
    # -----------------------------
    @function_tool()
    async def get_last_checkin(
        self,
        context: RunContext[WellnessState],
    ):
        """Return the most recent check-in."""
        path = Path("wellness_log.json").resolve()

        if not path.exists():
            return "There is no previous check-in recorded."

        try:
            logs = json.loads(path.read_text())
            if not logs:
                return "There is no previous check-in recorded."

            last = logs[-1]
            mood = last.get("mood")
            energy = last.get("energy")
            goals = ", ".join(last.get("goals", []))
            time = last.get("timestamp")

            return (
                f"Your last check-in was on {time}. "
                f"You felt {mood} with {energy} energy, "
                f"and your goals were: {goals}."
            )

        except Exception as e:
            return f"Error reading check-in history: {e}"

    # -----------------------------
    # FINALIZE CHECK-IN
    # -----------------------------
    async def _finalize_checkin(
        self,
        context: RunContext[WellnessState],
    ):
        data = context.userdata

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mood": data.last_mood,
            "energy": data.last_energy,
            "goals": data.last_goals,
            "summary": f"Mood: {data.last_mood}, Goals: {', '.join(data.last_goals)}"
        }

        # --- FIXED PATH ---
        path = Path("wellness_log.json").resolve()
        logs = []

        if path.exists():
            try:
                logs = json.loads(path.read_text())
            except:
                logs = []

        logs.append(entry)
        path.write_text(json.dumps(logs, indent=2))

        await context.session.generate_reply(
            instructions=(
                f"Here's your check-in summary:\n"
                f"• Mood: {data.last_mood}\n"
                f"• Energy: {data.last_energy}\n"
                f"• Goals: {', '.join(data.last_goals)}\n"
                "You've got this. Would you like to adjust anything?"
            )
        )


# -------------------------
# PREWARM
# -------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# -------------------------
# ENTRYPOINT
# -------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession[WellnessState](
        userdata=WellnessState(),
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
