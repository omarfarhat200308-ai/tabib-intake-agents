import asyncio
import logging
from dotenv import load_dotenv
from band import Agent
from band.adapters.anthropic import AnthropicAdapter
from band.config import load_agent_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    load_dotenv()

    agent_id, api_key = load_agent_config("tabib_intake")

    adapter = AnthropicAdapter(
        model="claude-sonnet-4-6",
        prompt="You are TABIB's Intake Agent. Collect patient symptoms via WhatsApp conversationally, then structure them into a clear summary and hand off to the Diagnostic agent using @mention."
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
