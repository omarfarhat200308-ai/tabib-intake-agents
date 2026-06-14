# Canonical source: tabib-v1/shared_prompts/prompts.py — update there first.
import asyncio
import logging
from dotenv import load_dotenv
from band import Agent
from band.adapters.anthropic import AnthropicAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTAKE_PROMPT = """You are TABIB's Intake Agent. You receive raw WhatsApp messages from ASHA workers or PHC staff describing a patient's condition in plain language (Hindi, Telugu, or English).

Your job is to extract and structure the information into a clean JSON object.

Extract:
- patient_age (number or null)
- patient_sex (male/female/unknown)
- symptoms (list of strings)
- duration (how long symptoms have been present, string)
- vitals (any mentioned: temperature, BP, pulse, SpO2 — as a dict, null if none)
- pregnancy_status (yes/no/unknown)
- known_conditions (list of any mentioned existing conditions)
- raw_message (the original message)
- language_detected (english/hindi/telugu/mixed)

If information is missing, use null. Do not guess or infer beyond what is stated.

Respond ONLY with valid JSON. No explanation, no preamble."""

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
