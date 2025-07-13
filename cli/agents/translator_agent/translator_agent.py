# import asyncio
# from typing import Annotated
# from genai_session.session import GenAISession
# from genai_session.utils.context import GenAIContext
#
# AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhNzE4ODUxYS04MmE0LTQxY2QtOTI3OC1hOTU4YWNhNTgwYmYiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6IjdmOTgwYTg0LWE4N2ItNDVmMy05ODBkLTYxN2E0ZWY0NjI1OSJ9.28vFdyyrvX5nh5GZIyb_JSpcGT0sBbIisv4AhUzkV9M" # noqa: E501
# session = GenAISession(jwt_token=AGENT_JWT)
#
#
# @session.bind(
#     name="translator_agent",
#     description="Agent that translate in different language"
# )
# async def translator_agent(
#     agent_context: GenAIContext,
#     test_arg: Annotated[
#         str,
#         "This is a test argument. Your agent can have as many parameters as you want. Feel free to rename or adjust it to your needs.",  # noqa: E501
#     ],
# ):
#     """Agent that translate in different language"""
#     return "Hello, World!"
#
#
# async def main():
#     print(f"Agent with token '{AGENT_JWT}' started")
#     await session.process_events()
#
# if __name__ == "__main__":
#     asyncio.run(main())
#
# ===============================================================================

import asyncio
import os
from typing import Annotated
from datetime import datetime
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from dotenv import load_dotenv
from openai import AsyncOpenAI
import logging

# === Load environment variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_JWT = os.getenv("AGENT_JWT")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")
if not AGENT_JWT:
    raise RuntimeError("AGENT_JWT is missing from .env")

# === Configure logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("translator_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("translator_agent")

# === Initialize GPT client and GenAI session ===
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
session = GenAISession(jwt_token=AGENT_JWT)


@session.bind(
    name="translator_agent",
    description="Translate text from source_language to destination_language using GPT-4o."
)
async def translator_agent(
    agent_context: GenAIContext,
    source_language: Annotated[str, "Language of the original transcript (e.g., English)"],
    destination_language: Annotated[str, "Target language for translation (e.g., French)"],
    original_transcript: Annotated[str, "The original text to translate"],
    gender: Annotated[str, "Preferred pronouns: neutral, male, female"] = "neutral"
):
    """
    Translate text using GPT-4o and return structured response.
    """
    logger.info("Received translation request:")
    logger.info(f"Source: {source_language}, Destination: {destination_language}")
    logger.info(f"Transcript: {original_transcript}, Gender: {gender}")

    if not original_transcript.strip():
        logger.warning("⚠ Empty original_transcript received.")
        return {
            "source_language": source_language,
            "destination_language": destination_language,
            "translated_transcript": None,
            "error": "Original transcript is empty."
        }

    system_prompt = (
        f"You are a professional translator. Translate from {source_language} "
        f"to {destination_language}. Preserve meaning, tone, and context. "
        f"Use {gender} pronouns if applicable."
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_transcript}
            ],
            temperature=0.2,
            max_tokens=4096
        )

        translated_text = response.choices[0].message.content.strip()
        logger.info("Translation successful:")
        logger.info(translated_text)

        result = {
            "source_language": source_language,
            "destination_language": destination_language,
            "translated_transcript": translated_text
        }

        logger.info("Returning response to event bus.")
        return result

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return {
            "source_language": source_language,
            "destination_language": destination_language,
            "translated_transcript": None,
            "error": str(e)
        }


async def self_test():
    """
    Standalone test for translator_agent without GenAI event system.
    """
    logger.info("🧪 Running self-test for translator_agent...")
    test_context = GenAIContext()
    response = await translator_agent(
        agent_context=test_context,
        source_language="English",
        destination_language="Spanish",
        original_transcript="Hello! How are you today?",
        gender="neutral"
    )
    logger.info("🧪 Self-Test Result:")
    logger.info(response)
    print("Self-Test Output:", response)


async def main():
    logger.info("translator_agent is starting and waiting for events...")
    try:
        await session.process_events()
    except KeyboardInterrupt:
        logger.info("translator_agent stopped by user.")
    except Exception as e:
        logger.error(f"translator_agent crashed: {e}", exc_info=True)
    finally:
        logger.info("translator_agent shutdown complete.")


if __name__ == "__main__":
    # Toggle between self-test and event listener
    USE_SELF_TEST = False  # Set True to test locally without event system
    if USE_SELF_TEST:
        asyncio.run(self_test())
    else:
        asyncio.run(main())
