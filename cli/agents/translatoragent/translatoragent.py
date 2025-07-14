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
AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3Mzc2MDY5OC1hMWJkLTRhODYtOGMwYi03YjgxN2M2NmQ3YWQiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.iATSdVdtkCHB5Df3TWAfLTGbDJvz6pqyJLz9IMStrLY" # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)


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

# AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3Mzc2MDY5OC1hMWJkLTRhODYtOGMwYi03YjgxN2M2NmQ3YWQiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.iATSdVdtkCHB5Df3TWAfLTGbDJvz6pqyJLz9IMStrLY" # noqa: E501
# session = GenAISession(jwt_token=AGENT_JWT)

# if not AGENT_JWT:
#     raise RuntimeError("AGENT_JWT is missing")

# # === Configure logging ===
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     handlers=[
#         logging.FileHandler("translator_agent.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("translator_agent")


# @session.bind(
#     name="translator_agent",
#     description="Translate text from source_language to destination_language using configurable LLM models."
# )
# async def translator_agent(
#     agent_context: GenAIContext,
#     source_language: Annotated[str, "Language of the original transcript (e.g., English)"],
#     destination_language: Annotated[str, "Target language for translation (e.g., French)"],
#     original_transcript: Annotated[str, "The original text to translate"],
#     gender: Annotated[str, "Preferred pronouns: neutral, male, female"] = "neutral",
#     llm_config: Annotated[dict, "LLM configuration containing provider, model, api_key, etc."] = None
# ):
#     """
#     Translate text using configurable LLM models and return structured response.
#     """
#     logger.info("Received translation request:")
#     logger.info(f"Source: {source_language}, Destination: {destination_language}")
#     logger.info(f"Transcript: {original_transcript}, Gender: {gender}")
#     logger.info(f"LLM Config: {llm_config}")

#     if not original_transcript.strip():
#         logger.warning("⚠ Empty original_transcript received.")
#         return {
#             "source_language": source_language,
#             "destination_language": destination_language,
#             "translated_transcript": None,
#             "error": "Original transcript is empty."
#         }

#     # Set default LLM configuration if none provided
#     if not llm_config:
#         llm_config = {
#             "provider": "genai",  # Default to genai provider
#             "model": "gpt-4o",
#             "temperature": 0.2,
#             "base_url": "http://localhost:8080"  # Default base URL
#         }
#         logger.info("Using default LLM configuration")

#     try:
#         # Create LLM model using the factory pattern
#         model = LLMFactory.create(llm_config)
#         logger.info(f"Created {llm_config.get('provider')} model: {llm_config.get('model')}")

#         # Create system prompt
#         system_prompt = (
#             f"You are a professional translator. Translate from {source_language} "
#             f"to {destination_language}. Preserve meaning, tone, and context. "
#             f"Use {gender} pronouns if applicable. Only respond with the translation."
#         )

#         # For LangChain models, we need to use their message format
#         from langchain_core.messages import HumanMessage, SystemMessage
        
#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=original_transcript)
#         ]

#         # Invoke the model
#         response = await model.ainvoke(messages)
        
#         # Extract the translated text
#         if hasattr(response, 'content'):
#             translated_text = response.content.strip()
#         else:
#             translated_text = str(response).strip()

#         logger.info("Translation successful:")
#         logger.info(translated_text)

#         result = {
#             "source_language": source_language,
#             "destination_language": destination_language,
#             "translated_transcript": translated_text,
#             "llm_provider": llm_config.get("provider"),
#             "llm_model": llm_config.get("model")
#         }

#         logger.info("Returning response to event bus.")
#         return result

#     except Exception as e:
#         logger.error(f"Translation failed: {e}", exc_info=True)
#         return {
#             "source_language": source_language,
#             "destination_language": destination_language,
#             "translated_transcript": None,
#             "llm_provider": llm_config.get("provider") if llm_config else "unknown",
#             "llm_model": llm_config.get("model") if llm_config else "unknown",
#             "error": str(e)
#         }


# async def self_test():
#     """
#     Standalone test for translator_agent without GenAI event system.
#     """
#     logger.info("🧪 Running self-test for translator_agent...")
    
#     # Test with different LLM configurations
#     test_configs = [
#         {
#             "provider": "genai",
#             "model": "gpt-4o",
#             "temperature": 0.2,
#             "base_url": "http://localhost:8080"
#         },
#         # You can add more configurations here for testing
#         # {
#         #     "provider": "openai",
#         #     "model": "gpt-4",
#         #     "api_key": "your-openai-key",
#         #     "temperature": 0.2
#         # }
#     ]
    
#     for config in test_configs:
#         logger.info(f"Testing with config: {config}")
#         test_context = GenAIContext()
#         response = await translator_agent(
#             agent_context=test_context,
#             source_language="English",
#             destination_language="Spanish",
#             original_transcript="Hello! How are you today?",
#             gender="neutral",
#             llm_config=config
#         )
#         logger.info("🧪 Self-Test Result:")
#         logger.info(response)
#         print("Self-Test Output:", response)


# async def main():
#     logger.info("translator_agent is starting and waiting for events...")
#     try:
#         await session.process_events()
#     except KeyboardInterrupt:
#         logger.info("translator_agent stopped by user.")
#     except Exception as e:
#         logger.error(f"translator_agent crashed: {e}", exc_info=True)
#     finally:
#         logger.info("translator_agent shutdown complete.")


# if __name__ == "__main__":
#     # Toggle between self-test and event listener
#     USE_SELF_TEST = False  # Set True to test locally without event system
#     if USE_SELF_TEST:
#         asyncio.run(self_test())
#     else:
#         asyncio.run(main())