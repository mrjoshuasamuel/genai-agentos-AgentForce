import asyncio
import os
from typing import Annotated
from datetime import datetime, timezone
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from dotenv import load_dotenv
from openai import AsyncOpenAI
import logging
import requests
import json

# === Load environment variables ===
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZDFjOTNjZi1hZGQ5LTQyZjEtYTBiMy0yMzYyMTRjNjJhZjAiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.L2nW6g8VI2AJsbuyUpDEeLcpkxMM8sw29OTo_NHB-_8" # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

headers = {
    "Authorization": "Bearer " + NOTION_API_KEY,
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# === Initialize GPT client and GenAI session ===
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from .env")
if not AGENT_JWT:
    raise RuntimeError("AGENT_JWT is missing from .env")


def get_database_schema():
    """Get the database schema to understand the property structure"""
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
    
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        
        schema = res.json()
        logger.info("Database schema retrieved successfully")
        logger.info(f"Properties: {list(schema.get('properties', {}).keys())}")
        
        return schema
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get database schema: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        return None


def create_page(title: str, summarized_text: str, translated_transcript: str = ""):
    """Create a new page in Notion database with summary and transcript"""
    create_url = "https://api.notion.com/v1/pages"
    
    # Get current datetime in UTC
    now_utc = datetime.now(timezone.utc)
    published_date = now_utc.isoformat()
    date_only = now_utc.date().isoformat()

    # First, let's get the database schema to understand the structure
    schema = get_database_schema()
    
    # Simple payload that should work with most Notion databases
    # Adjust property names based on your actual database schema
    data = {
        "Name": {  # Try "Name" instead of "Title"
            "title": [
                {
                    "text": {
                        "content": title[:100]  # Limit title length
                    }
                }
            ]
        }
    }
    
    # Try different property names for summary
    summary_property_names = ["Summary", "Description", "Content", "Notes"]
    for prop_name in summary_property_names:
        if schema and prop_name in schema.get('properties', {}):
            data[prop_name] = {
                "rich_text": [
                    {
                        "text": {
                            "content": summarized_text[:2000]  # Notion has character limits
                        }
                    }
                ]
            }
            break
    else:
        # Fallback if no matching property found
        data["Summary"] = {
            "rich_text": [
                {
                    "text": {
                        "content": summarized_text[:2000]
                    }
                }
            ]
        }

    # Try different property names for date
    date_property_names = ["Published", "Created", "Date", "Created Date"]
    for prop_name in date_property_names:
        if schema and prop_name in schema.get('properties', {}):
            data[prop_name] = {
                "date": {
                    "start": date_only
                }
            }
            break

    # Try different property names for status
    status_property_names = ["Status", "State", "Progress"]
    for prop_name in status_property_names:
        if schema and prop_name in schema.get('properties', {}):
            data[prop_name] = {
                "select": {
                    "name": "Processed"
                }
            }
            break

    # Prepare the minimal payload
    payload = {
        "parent": {
            "database_id": NOTION_DATABASE_ID
        },
        "properties": data
    }

    logger.info("Sending payload to Notion:")
    logger.info(json.dumps(payload, indent=2))

    try:
        res = requests.post(create_url, headers=headers, json=payload)
        
        # Log the response for debugging
        logger.info(f"Notion response status: {res.status_code}")
        logger.info(f"Notion response: {res.text}")
        
        res.raise_for_status()  # Raise an exception for bad status codes
        
        response_data = res.json()
        logger.info(f"Successfully created Notion page: {response_data.get('url', 'No URL')}")
        
        return {
            "success": True,
            "page_id": response_data.get("id"),
            "url": response_data.get("url"),
            "status_code": res.status_code
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create Notion page: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response text: {e.response.text}")  # Fixed: removed ()
            logger.error(f"Response headers: {dict(e.response.headers)}")
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
            "response_text": e.response.text if hasattr(e, 'response') and e.response is not None else None
        }


def create_simple_page(title: str, content: str):
    """Create a very simple page with minimal properties"""
    create_url = "https://api.notion.com/v1/pages"
    
    # Minimal payload that should work
    payload = {
        "parent": {
            "database_id": NOTION_DATABASE_ID
        },
        "properties": {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title[:100]
                        }
                    }
                ]
            }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content[:2000]
                            }
                        }
                    ]
                }
            }
        ]
    }

    logger.info("Sending simple payload to Notion:")
    logger.info(json.dumps(payload, indent=2))

    try:
        res = requests.post(create_url, headers=headers, json=payload)
        logger.info(f"Simple page response status: {res.status_code}")
        logger.info(f"Simple page response: {res.text}")
        
        res.raise_for_status()
        
        response_data = res.json()
        return {
            "success": True,
            "page_id": response_data.get("id"),
            "url": response_data.get("url"),
            "status_code": res.status_code
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create simple page: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response text: {e.response.text}")
        return {
            "success": False,
            "error": str(e),
            "response_text": e.response.text if hasattr(e, 'response') and e.response is not None else None
        }


def update_page(page_id: str, data: dict):
    """Update an existing Notion page"""
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {"properties": data}
    
    try:
        res = requests.patch(url, json=payload, headers=headers)
        res.raise_for_status()
        return {
            "success": True,
            "status_code": res.status_code
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update Notion page: {e}")
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }


@session.bind(
    name="text_summarizer_notion",
    description="Summarizes the translated text and posts it to Notion using API"
)
async def text_summarizer(
    agent_context: GenAIContext,
    translated_transcript: Annotated[str, "Transcript that is translated and needs to be summarized"],
    video_title: Annotated[str, "Title of the video being summarized"] = "Untitled Video",
    language: Annotated[str, "Language of the translated transcript"] = "Unknown",
    post_to_notion: Annotated[bool, "Whether to post the summary to Notion"] = True,
):
    """
    Summarizes the translated text and posts it to Notion using API.
    """
    logger.info("Received summarization request:")
    logger.info(f"Video Title: {video_title}")
    logger.info(f"Language: {language}")
    logger.info(f"Transcript length: {len(translated_transcript)} characters")
    logger.info(f"Post to Notion: {post_to_notion}")

    if not translated_transcript.strip():
        logger.warning("⚠ Empty translated_transcript received.")
        return {
            "video_title": video_title,
            "translated_transcript": translated_transcript,
            "error": "Transcript is empty."
        }

    # Create a more detailed system prompt
    system_prompt = (
        f"You are a professional summarizer. You will receive a translated text in {language} "
        f"from a video titled '{video_title}'. "
        f"Create a comprehensive but concise summary that captures:\n"
        f"- Main topics and key points\n"
        f"- Important insights or conclusions\n"
        f"- Key information that would be useful for someone who hasn't watched the video\n\n"
        f"Preserve the meaning, tone, and context. Write the summary in the same language as the input text."
    )

    try:
        # Generate summary using OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please summarize this transcript:\n\n{translated_transcript}"}
            ],
            temperature=0.2,
            max_tokens=1000  # Limit summary length
        )

        summarized_text = response.choices[0].message.content.strip()
        logger.info("Summarization successful")
        logger.info(f"Summary length: {len(summarized_text)} characters")

        # Prepare result with fixed datetime
        timestamp = datetime.now(timezone.utc).isoformat()  # Fixed deprecation warning
        result = {
            "video_title": video_title,
            "language": language,
            "translated_transcript": translated_transcript,
            "summarized_text": summarized_text,
            "timestamp": timestamp,
            "notion_posted": False,
            "notion_url": None
        }

        # Post to Notion if enabled
        if post_to_notion:
            logger.info("Posting to Notion...")
            
            # Try the regular create_page first
            notion_result = create_page(
                title=video_title,
                summarized_text=summarized_text,
                translated_transcript=translated_transcript
            )
            
            # If that fails, try the simple version
            if not notion_result["success"]:
                logger.info("Regular page creation failed, trying simple page...")
                notion_result = create_simple_page(
                    title=video_title,
                    content=f"Summary:\n{summarized_text}\n\nFull Text:\n{translated_transcript[:1000]}"
                )
            
            if notion_result["success"]:
                logger.info(f"Successfully posted to Notion: {notion_result.get('url', 'No URL')}")
                result["notion_posted"] = True
                result["notion_url"] = notion_result.get("url")
                result["notion_page_id"] = notion_result.get("page_id")
            else:
                logger.error(f"Failed to post to Notion: {notion_result.get('error', 'Unknown error')}")
                result["notion_error"] = notion_result.get("error")
                result["notion_response"] = notion_result.get("response_text")
        else:
            logger.info("Notion posting disabled")

        logger.info("Returning response to event bus.")
        return result

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        return {
            "video_title": video_title,
            "language": language,
            "translated_transcript": translated_transcript,
            "summarized_text": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),  # Fixed deprecation warning
            "notion_posted": False,
            "error": str(e)
        }

async def main():
    logger.info("text_summarizer is starting and waiting for events...")
    logger.info(f"Notion integration: {'Enabled' if NOTION_API_KEY and NOTION_DATABASE_ID else 'Disabled'}")
    
    try:
        await session.process_events()
    except KeyboardInterrupt:
        logger.info("text_summarizer stopped by user.")
    except Exception as e:
        logger.error(f"text_summarizer crashed: {e}", exc_info=True)
    finally:
        logger.info("text_summarizer shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())