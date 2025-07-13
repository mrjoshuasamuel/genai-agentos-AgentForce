


import asyncio
import os
import logging
from typing import Annotated, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from openai import OpenAI
import openai
import json


# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = ""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZmYxMmViYi05MDRlLTQyNTktOWQzMi00MWMyZTIwM2E0ZDEiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6IjdmOTgwYTg0LWE4N2ItNDVmMy05ODBkLTYxN2E0ZWY0NjI1OSJ9.9AeF-3X7HJlGFl0wugzEx15YfmnEDme_KkFpKpJrN6o" # noqa: E501


# AGENT_JWT = os.getenv("AGENT_JWT") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # Add your token here
# AGENT_JWT = os.getenv("AGENT_JWT") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # Add your token here

session = GenAISession(jwt_token=AGENT_JWT)


class ResponseBuilder:
    """Standardize all agent responses"""

    def __init__(self):
        self.start_time = datetime.now()

    def success(self, data: Any, message: str = "Translation completed", metadata: Dict = None) -> Dict[str, Any]:
        return {
            "success": True,
            "message": message,
            "data": data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_ms": int((datetime.now() - self.start_time).total_seconds() * 1000),
                "agent": "translator_agent",
                "category": "translation",
                "complexity": "intermediate",
                **(metadata or {})
            }
        }

    def error(self, error_type: str, message: str, details: Dict = None) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error_type,
            "message": message,
            "details": details or {},
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_ms": int((datetime.now() - self.start_time).total_seconds() * 1000),
                "agent": "translator_agent",
                "category": "translation"
            }
        }


def translate_with_gpt(source_lang: str, dest_lang: str, text: str) -> str:
    """Use OpenAI GPT model for translation"""
    logger.info(f"Translating using GPT-4 from {source_lang} to {dest_lang}")

    prompt = (
        f"Translate the following text from {source_lang} to {dest_lang}. "
        f"Preserve meaning, tone, and context. Only return the translated text:\n\n{text}"
    )

    try:
        client = OpenAI(  # This is the default and can be omitted
        api_key=OPENAI_API_KEY),
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a highly accurate multilingual translator."},
            {"role": "user", "content": prompt}
        ],
        )
        translated_text = response.choices[0].message.content
        logger.info(f"Translation result: {translated_text}")
        return translated_text
    except Exception as e:
        logger.error(f"GPT translation failed: {e}")
        raise ValueError("Failed to translate text with GPT-4.")


@session.bind(
    name="translator_agent",
    description="Translate text from source language to destination language using GPT-4"
)
async def translator_agent(
        agent_context: GenAIContext,
        source_language: Annotated[str, "Language of the original transcript"],
        destination_language: Annotated[str, "Target language for translation"],
        original_transcript: Annotated[str, "The original transcript text"],
        gender: Annotated[str, "Speaker gender (optional)"] = None,
):
    """Translate text using OpenAI GPT-4"""
    response_builder = ResponseBuilder()
    try:
        if not original_transcript.strip():
            raise ValueError("Original transcript cannot be empty")

        translated_text = translate_with_gpt(source_language, destination_language, original_transcript)
        logger.info("After Translation Test")

        result = {
            "source_language": source_language,
            "destination_language": destination_language,
            "translated_transcript": translated_text
        }

        logger.info("Translation completed successfully")
        return response_builder.success(result)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return response_builder.error("validation_error", str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return response_builder.error("processing_error", str(e))


async def main():
    logger.info("Starting translator_agent...")
    try:
        await session.process_events()
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Agent crashed: {e}", exc_info=True)
    finally:
        logger.info("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
