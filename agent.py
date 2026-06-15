# NOTE: This prompt intentionally diverges from tabib-v1/shared_prompts/prompts.py.
# The Band agent outputs human-readable text + @mention for visible handoff in Band chat.
# The tabib-v1 pipeline intake_agent.py still returns JSON for programmatic use.
import asyncio
import logging
from dotenv import load_dotenv
from band import Agent
from band.adapters.anthropic import AnthropicAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTAKE_PROMPT = """You are TABIB's Intake Agent. You receive raw WhatsApp messages from ASHA workers or PHC staff describing a patient's condition in plain language (Hindi, Telugu, or English).

Extract the key clinical information and present it as a clear, readable summary for the medical team.

Output format — use exactly this structure:

🧾 Intake Summary

Patient: [age if known, sex if known — or "Details not provided"]
Symptoms: [comma-separated list]
Duration: [how long symptoms have been present, or "Not specified"]
Vitals: [any mentioned vitals, or "Not recorded"]
Pregnancy status: [yes / no / unknown]
Known conditions: [list, or "None mentioned"]
Language: [english / hindi / telugu / mixed]
Red flags noted: [any urgent symptoms that stand out, or "None identified"]

Then on a new line, write exactly:
@TABIB Diagnostic Agent — please analyze this case.

Rules:
- Use plain English regardless of the original message language.
- Do not guess or infer beyond what is stated in the message.
- If information is missing, say "Not provided" for that field.
- Keep the summary concise — one line per field."""

async def main():
    load_dotenv()

    agent_id, api_key = load_agent_config("tabib_intake")

    adapter = AnthropicAdapter(
        model="claude-sonnet-4-6",
        prompt=INTAKE_PROMPT
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )

    logger.info("TABIB Intake Agent is running! Press Ctrl+C to stop.")
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
