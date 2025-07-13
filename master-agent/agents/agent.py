import asyncio
import os
import logging
from typing import Annotated, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

import aiohttp
import json
from openai import OpenAI
from transformers import pipeline
import heapq
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AGENT_JWT = "{{agent_token}}"
session = GenAISession(jwt_token=AGENT_JWT)

class ResponseBuilder:
    def __init__(self):
        self.start_time = datetime.now()

    def success(self, data: Any, message: str = "Operation completed successfully", metadata: Dict = None) -> Dict[str, Any]:
        return {
            "success": True,
            "message": message,
            "data": data,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_ms": int((datetime.now() - self.start_time).total_seconds() * 1000),
                "agent": "text_summarizer",
                "category": "ai-ml",
                "complexity": "expert",
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
                "agent": "text_summarizer",
                "category": "ai-ml"
            }
        }

async def post_to_notion_audio_async(summary: str):
    notion_api_url = os.getenv("NOTION_AUDIO_ENDPOINT")
    notion_api_key = os.getenv("NOTION_API_KEY")

    if not notion_api_url or not notion_api_key:
        raise ValueError("NOTION_AUDIO_ENDPOINT and NOTION_API_KEY must be set in .env")

    payload = {"summary": summary, "timestamp": datetime.now().isoformat()}
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as http_session:
        async with http_session.post(notion_api_url, headers=headers, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Failed to post to Notion. Status: {response.status}")

@session.bind(
    name="text_summarizer",
    description="Intelligent text summarization with extractive and abstractive methods"
)
async def text_summarizer(
    agent_context: GenAIContext,
    text: Annotated[str, "Text to summarize"],
    summary_type: Annotated[str, "Type: extractive, abstractive, or hybrid"],
    max_length: Annotated[int, "Maximum summary length in words"],
    key_points: Annotated[int, "Number of key points to extract"],
):
    response_builder = ResponseBuilder()

    try:
        openai_api_key_value = os.getenv("OPENAI_API_KEY")
        if not openai_api_key_value:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not text or not text.strip():
            raise ValueError("Text to summarize is required and cannot be empty")

        logger.info("All validations passed, starting summarization logic...")

        summary_text = ""

        if summary_type.lower() == "extractive":
            sentences = re.split(r'(?<=[.!?]) +', text)
            word_frequencies = {}
            for word in re.findall(r'\w+', text.lower()):
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

            max_freq = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] /= max_freq

            sentence_scores = {}
            for sent in sentences:
                for word in re.findall(r'\w+', sent.lower()):
                    if word in word_frequencies:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

            summary_sentences = heapq.nlargest(key_points, sentence_scores, key=sentence_scores.get)
            summary_text = " ".join(summary_sentences)

        elif summary_type.lower() == "abstractive":
            try:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summary_output = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
                summary_text = summary_output[0]["summary_text"]
            except Exception as e:
                logger.warning("Abstractive summarization failed, falling back to extractive.")
                # Fallback to extractive
                sentences = re.split(r'(?<=[.!?]) +', text)
                word_frequencies = {}
                for word in re.findall(r'\w+', text.lower()):
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

                max_freq = max(word_frequencies.values())
                for word in word_frequencies:
                    word_frequencies[word] /= max_freq

                sentence_scores = {}
                for sent in sentences:
                    for word in re.findall(r'\w+', sent.lower()):
                        if word in word_frequencies:
                            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

                summary_sentences = heapq.nlargest(key_points, sentence_scores, key=sentence_scores.get)
                summary_text = " ".join(summary_sentences)

        elif summary_type.lower() == "hybrid":
            sentences = re.split(r'(?<=[.!?]) +', text)
            word_frequencies = {}
            for word in re.findall(r'\w+', text.lower()):
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

            max_freq = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] /= max_freq

            sentence_scores = {}
            for sent in sentences:
                for word in re.findall(r'\w+', sent.lower()):
                    if word in word_frequencies:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

            extracted = " ".join(heapq.nlargest(key_points, sentence_scores, key=sentence_scores.get))
            abstracter = pipeline("summarization", model="facebook/bart-large-cnn")
            summary_output = abstracter(extracted, max_length=max_length, min_length=30, do_sample=False)
            summary_text = summary_output[0]["summary_text"]

        else:
            raise ValueError("Invalid summary_type. Choose from 'extractive', 'abstractive', or 'hybrid'.")

        logger.info("Summary generation complete. Posting to Notion Audio...")

        result = {
            "summary_type": summary_type,
            "summary": summary_text,
            "length": len(summary_text.split()),
            "status": "processed"
        }

        if os.getenv("DISABLE_NOTION_POST") != "true":
            await post_to_notion_audio_async(summary_text)

        return response_builder.success(
            data=result,
            message="Summarization and Notion posting complete",
            metadata={"processing_time": "optimized", "scalability": "medium", "security_level": "standard"}
        )

    except ValueError as e:
        logger.error(f"Validation error in text_summarizer: {e}")
        return response_builder.error("validation_error", str(e))

    except ConnectionError as e:
        logger.error(f"Connection error in text_summarizer: {e}")
        return response_builder.error("connection_error", str(e))

    except TimeoutError as e:
        logger.error(f"Timeout error in text_summarizer: {e}")
        return response_builder.error("timeout_error", str(e))

    except Exception as e:
        logger.error(f"Unexpected error in text_summarizer: {e}", exc_info=True)
        return response_builder.error("processing_error", str(e))

async def main():
    logger.info(f"Starting text_summarizer agent with token '{{agent_token}}'")
    logger.info(f"Agent category: ai-ml")
    logger.info(f"Complexity level: expert")
    logger.info(f"Scalability: medium")

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
