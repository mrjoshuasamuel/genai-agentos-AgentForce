import asyncio
import os
from typing import Annotated, Dict, Tuple
from datetime import datetime, timezone
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from dotenv import load_dotenv
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

# === Configuration ===
AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4NzM4MzY1OS1iM2ZiLTQ2OGItYTNhNS0zZjI1MTZhMzA5MDgiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.vADBMcsi3xIjxV3aGO5AlNzXxdvWiQboyYmFJUpqESw" # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)

# === Murf Configuration ===
MURF_API_KEY = os.getenv("MURF_API_KEY")

# === Notion Configuration ===
NOTION_API_KEY = "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"
NOTION_DATABASE_ID = "22ff169cb25a8099af37de9fc2d7f8ec"
if not MURF_API_KEY:
    logger.warning("MURF_API_KEY is missing - TTS generation will be disabled")
if not NOTION_API_KEY:
    logger.warning("NOTION_API_KEY is missing - Notion updates will be disabled")

# === Notion headers ===
notion_headers = {
    "Authorization": "Bearer " + NOTION_API_KEY if NOTION_API_KEY else "",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# === Voice and Locale Mapping ===
class VoiceMapper:
    """Maps destination languages to appropriate Murf voice IDs and locales"""
    
    # Comprehensive language to voice/locale mapping
    LANGUAGE_MAPPING = {
        # German
        "German": ("en-UK-ruby", "de-DE"),
        "Deutsch": ("en-UK-ruby", "de-DE"),
        
        # Greek
        "Greek": ("en-UK-ruby", "el-GR"),
        "Ελληνικά": ("en-UK-ruby", "el-GR"),
        
        # English variants
        "English": ("en-UK-ruby", "en-US"),  # Default to US English
        "English (UK)": ("en-UK-ruby", "en-UK"),
        "English (US)": ("en-UK-ruby", "en-US"),
        "British English": ("en-UK-ruby", "en-UK"),
        "American English": ("en-UK-ruby", "en-US"),
        
        # Spanish variants
        "Spanish": ("en-UK-ruby", "es-ES"),  # Default to Spain Spanish
        "Spanish (Spain)": ("en-UK-ruby", "es-ES"),
        "Spanish (Mexico)": ("en-UK-ruby", "es-MX"),
        "Español": ("en-UK-ruby", "es-ES"),
        "Castilian": ("en-UK-ruby", "es-ES"),
        
        # French
        "French": ("en-UK-ruby", "fr-FR"),
        "Français": ("en-UK-ruby", "fr-FR"),
        
        # Hindi
        "Hindi": ("en-UK-ruby", "hi-IN"),
        "हिन्दी": ("en-UK-ruby", "hi-IN"),
        
        # Croatian
        "Croatian": ("en-UK-ruby", "hr-HR"),
        "Hrvatski": ("en-UK-ruby", "hr-HR"),
        
        # Indonesian
        "Indonesian": ("en-UK-ruby", "id-ID"),
        "Bahasa Indonesia": ("en-UK-ruby", "id-ID"),
        
        # Turkish
        "Turkish": ("en-UK-ruby", "tr-TR"),
        "Türkçe": ("en-UK-ruby", "tr-TR"),
        
        # Italian
        "Italian": ("it-IT-lorenzo", "it-IT"),
        "Italiano": ("it-IT-lorenzo", "it-IT"),
        
        # Chinese
        "Chinese": ("zh-CN-tao", "zh-CN"),
        "Chinese (Simplified)": ("zh-CN-tao", "zh-CN"),
        "Mandarin": ("zh-CN-tao", "zh-CN"),
        "中文": ("zh-CN-tao", "zh-CN"),
        
        # Korean
        "Korean": ("ko-KR-gyeong", "ko-KR"),
        "한국어": ("ko-KR-gyeong", "ko-KR"),
        
        # Japanese
        "Japanese": ("ja-JP-kenji", "ja-JP"),
        "日本語": ("ja-JP-kenji", "ja-JP"),
        
        # Slovak
        "Slovak": ("sk-SK-nina", "sk-SK"),
        "Slovenčina": ("sk-SK-nina", "sk-SK"),
    }
    
    @classmethod
    def get_voice_config(cls, destination_language: str) -> Tuple[str, str]:
        """
        Get the appropriate voice ID and locale for a destination language
        
        Args:
            destination_language: The target language name
            
        Returns:
            Tuple of (voice_id, locale)
        """
        # Normalize the language name (strip whitespace, title case)
        normalized_lang = destination_language.strip()
        
        # Try exact match first
        if normalized_lang in cls.LANGUAGE_MAPPING:
            voice_id, locale = cls.LANGUAGE_MAPPING[normalized_lang]
            logger.info(f"Exact match found for '{normalized_lang}': {voice_id} ({locale})")
            return voice_id, locale
        
        # Try case-insensitive match
        for lang_key, (voice_id, locale) in cls.LANGUAGE_MAPPING.items():
            if normalized_lang.lower() == lang_key.lower():
                logger.info(f"Case-insensitive match found for '{normalized_lang}': {voice_id} ({locale})")
                return voice_id, locale
        
        # Try partial matching (if destination language contains a known language)
        for lang_key, (voice_id, locale) in cls.LANGUAGE_MAPPING.items():
            if lang_key.lower() in normalized_lang.lower() or normalized_lang.lower() in lang_key.lower():
                logger.info(f"Partial match found for '{normalized_lang}' -> '{lang_key}': {voice_id} ({locale})")
                return voice_id, locale
        
        # Default fallback to English (US)
        logger.warning(f"No match found for '{normalized_lang}', defaulting to English (US)")
        return "en-UK-ruby", "en-US"
    
    @classmethod
    def get_supported_languages(cls) -> list:
        """Get list of all supported languages"""
        return list(cls.LANGUAGE_MAPPING.keys())
    
    @classmethod
    def get_voice_capabilities(cls) -> dict:
        """Get detailed voice capabilities"""
        return {
            "en-UK-ruby": {
                "locales": ["de-DE", "el-GR", "en-UK", "en-US", "es-ES", "es-MX", "fr-FR", "hi-IN", "hr-HR", "id-ID", "tr-TR"],
                "description": "Multi-language capable voice"
            },
            "it-IT-lorenzo": {
                "locales": ["it-IT"],
                "description": "Italian native voice"
            },
            "zh-CN-tao": {
                "locales": ["zh-CN"],
                "description": "Chinese (Mandarin) native voice"
            },
            "ko-KR-gyeong": {
                "locales": ["ko-KR"],
                "description": "Korean native voice"
            },
            "ja-JP-kenji": {
                "locales": ["ja-JP"],
                "description": "Japanese native voice"
            },
            "sk-SK-nina": {
                "locales": ["sk-SK"],
                "description": "Slovak native voice"
            }
        }


class MurfTTSClient:
    """Murf TTS client wrapper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from murf import Murf
            self.client = Murf(api_key=api_key)
        except ImportError:
            logger.error("Murf package not installed. Run: pip install murf")
            self.client = None
    
    def generate_audio(self, text: str, voice_id: str, locale: str) -> dict:
        """Generate TTS audio using Murf API"""
        if not self.client:
            raise RuntimeError("Murf client not initialized")
        
        logger.info(f"Generating TTS audio with voice_id={voice_id}, locale={locale}")
        
        try:
            res = self.client.text_to_speech.generate(
                text=text,
                voice_id=voice_id,
                multi_native_locale=locale
            )
            
            audio_url = res.audio_file
            logger.info(f"TTS audio generated successfully: {audio_url}")
            
            return {
                "success": True,
                "audio_url": audio_url,
                "voice_id": voice_id,
                "locale": locale,
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Murf TTS generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "voice_id": voice_id,
                "locale": locale
            }


def find_notion_page_by_title(title: str) -> dict:
    """Find a Notion page by title in the database"""
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        return {"success": False, "error": "Notion not configured"}
    
    search_url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    
    # Search for pages with matching title
    payload = {
        "filter": {
            "property": "Name",
            "title": {
                "equals": title
            }
        }
    }
    
    try:
        res = requests.post(search_url, headers=notion_headers, json=payload)
        res.raise_for_status()
        
        data = res.json()
        results = data.get("results", [])
        
        if results:
            page = results[0]
            logger.info(f"Found Notion page: {page.get('id')}")
            return {
                "success": True,
                "page_id": page.get("id"),
                "url": page.get("url"),
                "page": page
            }
        else:
            logger.warning(f"No Notion page found with title: {title}")
            return {
                "success": False,
                "error": f"No page found with title: {title}"
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to search Notion: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def update_notion_page_with_audio(page_id: str, audio_url: str, voice_info: dict) -> dict:
    """Update an existing Notion page with audio URL"""
    if not NOTION_API_KEY:
        return {"success": False, "error": "Notion not configured"}
    
    # Add audio URL as a block to the page content
    blocks_to_add = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": "🎵 Generated Audio"
                        }
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"Voice: {voice_info.get('voice_id', 'Unknown')} | "
                        }
                    },
                    {
                        "type": "text",
                        "text": {
                            "content": f"Language: {voice_info.get('locale', 'Unknown')} | "
                        }
                    },
                    {
                        "type": "text",
                        "text": {
                            "content": "🔗 Download Audio",
                            "link": {
                                "url": audio_url
                            }
                        }
                    }
                ]
            }
        },
        {
            "object": "block",
            "type": "embed",
            "embed": {
                "url": audio_url
            }
        }
    ]
    
    try:
        # Add audio blocks to page content
        blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        blocks_payload = {"children": blocks_to_add}
        
        blocks_res = requests.patch(blocks_url, headers=notion_headers, json=blocks_payload)
        logger.info(f"Blocks update status: {blocks_res.status_code}")
        
        if blocks_res.status_code == 200:
            logger.info(f"Successfully updated Notion page with audio URL")
            return {
                "success": True,
                "page_id": page_id,
                "audio_url": audio_url
            }
        else:
            logger.error(f"Failed to update Notion page: {blocks_res.status_code}")
            logger.error(f"Response: {blocks_res.text}")
            return {
                "success": False,
                "error": f"HTTP {blocks_res.status_code}: {blocks_res.text}"
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update Notion page: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Initialize Murf client
murf_client = MurfTTSClient(MURF_API_KEY) if MURF_API_KEY else None


@session.bind(
    name="dubbing_agent_auto_voice",
    description="Converts text to speech using automatic voice/locale mapping based on destination language"
)
async def dubbing_agent(
    agent_context: GenAIContext,
    text_to_synthesize: Annotated[str, "Text to convert to speech (summary or transcript)"],
    destination_language: Annotated[str, "Target language (e.g., 'English', 'French', 'German', 'Spanish', etc.)"],
    video_title: Annotated[str, "Title of the video (to find corresponding Notion page)"] = "",
    notion_page_id: Annotated[str, "Direct Notion page ID (if known, skips title search)"] = "",
    update_notion: Annotated[bool, "Whether to update the Notion page with audio URL"] = True,
    override_voice_id: Annotated[str, "Override automatic voice selection with specific voice ID"] = "",
    override_locale: Annotated[str, "Override automatic locale selection with specific locale"] = "",
):
    """
    Converts text to speech using automatic voice/locale mapping or manual overrides
    """
    logger.info("Received dubbing request:")
    logger.info(f"Text length: {len(text_to_synthesize)} characters")
    logger.info(f"Destination language: {destination_language}")
    logger.info(f"Video title: {video_title}")
    logger.info(f"Update Notion: {update_notion}")

    if not text_to_synthesize.strip():
        logger.warning("⚠ Empty text_to_synthesize received.")
        return {
            "text_to_synthesize": text_to_synthesize,
            "error": "Text to synthesize is empty."
        }

    if not murf_client:
        logger.error("Murf client not available")
        return {
            "text_to_synthesize": text_to_synthesize,
            "error": "Murf TTS not configured - missing API key"
        }

    # Determine voice and locale
    if override_voice_id and override_locale:
        voice_id = override_voice_id
        locale = override_locale
        logger.info(f"Using manual override: {voice_id} ({locale})")
    else:
        voice_id, locale = VoiceMapper.get_voice_config(destination_language)
        logger.info(f"Auto-mapped '{destination_language}' to: {voice_id} ({locale})")

    # Prepare result
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "text_to_synthesize": text_to_synthesize,
        "destination_language": destination_language,
        "voice_id": voice_id,
        "locale": locale,
        "video_title": video_title,
        "timestamp": timestamp,
        "audio_generated": False,
        "audio_url": None,
        "notion_updated": False,
        "notion_page_id": notion_page_id,
        "voice_mapping_used": not (override_voice_id and override_locale)
    }

    try:
        # Generate TTS audio
        logger.info("Generating TTS audio...")
        tts_result = murf_client.generate_audio(
            text=text_to_synthesize,
            voice_id=voice_id,
            locale=locale
        )
        
        if tts_result["success"]:
            logger.info("TTS audio generated successfully")
            result["audio_generated"] = True
            result["audio_url"] = tts_result["audio_url"]
            result["tts_info"] = tts_result
        else:
            logger.error(f"TTS generation failed: {tts_result.get('error')}")
            result["tts_error"] = tts_result.get("error")
            return result

        # Update Notion page if requested
        if update_notion and result["audio_url"]:
            logger.info("Updating Notion page...")
            
            # Find the page if we don't have a direct page ID
            if not notion_page_id and video_title:
                logger.info(f"Searching for Notion page with title: {video_title}")
                page_search = find_notion_page_by_title(video_title)
                
                if page_search["success"]:
                    notion_page_id = page_search["page_id"]
                    result["notion_page_id"] = notion_page_id
                    logger.info(f"Found page ID: {notion_page_id}")
                else:
                    logger.warning(f"Could not find Notion page: {page_search.get('error')}")
                    result["notion_error"] = page_search.get("error")
            
            # Update the page with audio URL
            if notion_page_id:
                notion_result = update_notion_page_with_audio(
                    page_id=notion_page_id,
                    audio_url=result["audio_url"],
                    voice_info={
                        "voice_id": voice_id,
                        "locale": locale,
                        "destination_language": destination_language
                    }
                )
                
                if notion_result["success"]:
                    logger.info("Successfully updated Notion page with audio")
                    result["notion_updated"] = True
                else:
                    logger.error(f"Failed to update Notion: {notion_result.get('error')}")
                    result["notion_error"] = notion_result.get("error")
            else:
                logger.warning("No Notion page ID available for update")
                result["notion_error"] = "No page ID provided and title search failed"

        logger.info("Returning response to event bus.")
        return result

    except Exception as e:
        logger.error(f"Dubbing agent failed: {e}", exc_info=True)
        return {
            "text_to_synthesize": text_to_synthesize,
            "destination_language": destination_language,
            "voice_id": voice_id,
            "locale": locale,
            "timestamp": timestamp,
            "audio_generated": False,
            "notion_updated": False,
            "error": str(e)
        }


# @session.bind(
#     name="get_supported_languages",
#     description="Get list of all supported languages and voice capabilities"
# )
# async def get_supported_languages(
#     agent_context: GenAIContext,
# ):
#     """Return list of supported languages and voice mapping information"""
    
#     result = {
#         "supported_languages": VoiceMapper.get_supported_languages(),
#         "voice_capabilities": VoiceMapper.get_voice_capabilities(),
#         "total_languages": len(VoiceMapper.get_supported_languages()),
#         "total_voices": len(VoiceMapper.get_voice_capabilities())
#     }
    
#     logger.info(f"Returning {result['total_languages']} supported languages")
#     return result

async def main():
    logger.info("dubbing_agent with auto voice mapping is starting and waiting for events...")
    logger.info(f"Murf TTS: {'Enabled' if murf_client else 'Disabled'}")
    logger.info(f"Notion integration: {'Enabled' if NOTION_API_KEY else 'Disabled'}")
    logger.info(f"Supported languages: {len(VoiceMapper.get_supported_languages())}")
    
    try:
        await session.process_events()
    except KeyboardInterrupt:
        logger.info("dubbing_agent stopped by user.")
    except Exception as e:
        logger.error(f"dubbing_agent crashed: {e}", exc_info=True)
    finally:
        logger.info("dubbing_agent shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())