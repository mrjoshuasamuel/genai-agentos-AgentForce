import asyncio
import re
from typing import Any, Optional, Tuple, Dict, List
from dataclasses import dataclass
from urllib.parse import urlparse

from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from langchain_core.messages import SystemMessage
from loguru import logger

from agents.react_master_agent import ReActMasterAgent
from config.settings import Settings
from llms import LLMFactory
from prompts import FILE_RELATED_SYSTEM_PROMPT
from utils.agents import get_agents
from utils.chat_history import get_chat_history
from utils.common import attach_files_to_message

app_settings = Settings()

session = GenAISession(
    api_key=app_settings.MASTER_AGENT_API_KEY,
    ws_url=app_settings.ROUTER_WS_URL
)


@dataclass
class VideoAnalysis:
    """Data class for video request analysis results."""
    original_message: str
    video_url: Optional[str] = None
    has_video_url: bool = False
    has_translation_intent: bool = False
    target_language: Optional[str] = None
    has_target_language: bool = False
    is_complete_request: bool = False
    missing_components: List[str] = None
    confidence_score: float = 0.0
    platform_detected: Optional[str] = None
    
    def __post_init__(self):
        if self.missing_components is None:
            self.missing_components = []


class VideoURLExtractor:
    """Enhanced video URL extraction with support for multiple platforms."""
    
    # Common video hosting platforms that yt_dlp supports
    SUPPORTED_PLATFORMS = [
        # YouTube
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:\S*)?',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})(?:\S*)?',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})(?:\S*)?',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})(?:\S*)?',
        r'(?:https?://)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:\S*)?',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})(?:\S*)?',
        
        # Vimeo
        r'(?:https?://)?(?:www\.)?vimeo\.com/(\d+)(?:\S*)?',
        r'(?:https?://)?player\.vimeo\.com/video/(\d+)(?:\S*)?',
        
        # Dailymotion
        r'(?:https?://)?(?:www\.)?dailymotion\.com/video/([a-zA-Z0-9]+)(?:\S*)?',
        r'(?:https?://)?dai\.ly/([a-zA-Z0-9]+)(?:\S*)?',
        
        # Twitch
        r'(?:https?://)?(?:www\.)?twitch\.tv/videos/(\d+)(?:\S*)?',
        r'(?:https?://)?clips\.twitch\.tv/([a-zA-Z0-9_-]+)(?:\S*)?',
        
        # Facebook
        r'(?:https?://)?(?:www\.)?facebook\.com/.*/videos/(\d+)(?:\S*)?',
        r'(?:https?://)?(?:www\.)?facebook\.com/watch/?\?v=(\d+)(?:\S*)?',
        
        # Instagram
        r'(?:https?://)?(?:www\.)?instagram\.com/p/([a-zA-Z0-9_-]+)(?:\S*)?',
        r'(?:https?://)?(?:www\.)?instagram\.com/reel/([a-zA-Z0-9_-]+)(?:\S*)?',
        
        # TikTok
        r'(?:https?://)?(?:www\.)?tiktok\.com/@[^/]+/video/(\d+)(?:\S*)?',
        r'(?:https?://)?vm\.tiktok\.com/([a-zA-Z0-9]+)(?:\S*)?',
        
        # Twitter/X
        r'(?:https?://)?(?:www\.)?twitter\.com/\w+/status/(\d+)(?:\S*)?',
        r'(?:https?://)?(?:www\.)?x\.com/\w+/status/(\d+)(?:\S*)?',
        
        # Reddit
        r'(?:https?://)?(?:www\.)?reddit\.com/r/\w+/comments/[a-zA-Z0-9]+/[^/]+/?(?:\S*)?',
        
        # Streamable
        r'(?:https?://)?streamable\.com/([a-zA-Z0-9]+)(?:\S*)?',
        
        # Bitchute
        r'(?:https?://)?(?:www\.)?bitchute\.com/video/([a-zA-Z0-9]+)(?:\S*)?',
        
        # Generic video file URLs (direct links)
        r'(?:https?://\S+\.(?:mp4|avi|mov|wmv|flv|webm|mkv|m4v)(?:\?\S*)?)',
    ]
    
    # Platform detection patterns
    PLATFORM_PATTERNS = {
        'youtube': [r'youtube\.com', r'youtu\.be'],
        'vimeo': [r'vimeo\.com'],
        'dailymotion': [r'dailymotion\.com', r'dai\.ly'],
        'twitch': [r'twitch\.tv'],
        'facebook': [r'facebook\.com'],
        'instagram': [r'instagram\.com'],
        'tiktok': [r'tiktok\.com'],
        'twitter': [r'twitter\.com', r'x\.com'],
        'reddit': [r'reddit\.com'],
        'streamable': [r'streamable\.com'],
        'bitchute': [r'bitchute\.com'],
        'direct': [r'\.(mp4|avi|mov|wmv|flv|webm|mkv|m4v)']
    }
    
    @classmethod
    def extract(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract video URL from text and detect platform."""
        # First try specific platform patterns
        for pattern in cls.SUPPORTED_PLATFORMS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the full URL from the original text
                url_match = re.search(r'https?://\S+', text, re.IGNORECASE)
                if url_match:
                    full_url = url_match.group(0)
                    platform = cls._detect_platform(full_url)
                    return full_url, platform
        
        # Fallback: look for any URL-like pattern and validate
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        
        for url in urls:
            if cls._is_potential_video_url(url):
                platform = cls._detect_platform(url)
                return url, platform
        
        return None, None
    
    @classmethod
    def _detect_platform(cls, url: str) -> str:
        """Detect the platform from URL."""
        url_lower = url.lower()
        
        for platform, patterns in cls.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return platform
        
        return 'unknown'
    
    @classmethod
    def _is_potential_video_url(cls, url: str) -> bool:
        """Check if URL could potentially be a video URL."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            
            # Check for common video-related domains or file extensions
            video_indicators = [
                'video', 'watch', 'play', 'stream', 'tv', 'tube', 'clip',
                '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'
            ]
            
            url_lower = url.lower()
            return any(indicator in url_lower for indicator in video_indicators)
        
        except Exception:
            return False


class LanguageDetector:
    """Enhanced language detection with confidence scoring."""
    
    # Enhanced language mapping with more variations and regional codes
    LANGUAGE_MAPPING = {
        'spanish': {
            'variants': ['spanish', 'español', 'castellano', 'es', 'spa', 'es-es', 'es-mx', 'es-ar'],
            'weight': 1.0
        },
        'french': {
            'variants': ['french', 'français', 'francais', 'fr', 'fra', 'fr-fr', 'fr-ca'],
            'weight': 1.0
        },
        'german': {
            'variants': ['german', 'deutsch', 'de', 'ger', 'deu', 'de-de', 'de-at'],
            'weight': 1.0
        },
        'italian': {
            'variants': ['italian', 'italiano', 'it', 'ita', 'it-it'],
            'weight': 1.0
        },
        'portuguese': {
            'variants': ['portuguese', 'português', 'portugues', 'pt', 'por', 'pt-br', 'pt-pt'],
            'weight': 1.0
        },
        'russian': {
            'variants': ['russian', 'русский', 'ru', 'rus', 'ru-ru'],
            'weight': 1.0
        },
        'chinese': {
            'variants': ['chinese', 'mandarin', '中文', 'zh', 'chi', 'cmn', 'zh-cn', 'zh-tw'],
            'weight': 1.0
        },
        'japanese': {
            'variants': ['japanese', 'nihongo', '日本語', 'ja', 'jpn', 'ja-jp'],
            'weight': 1.0
        },
        'korean': {
            'variants': ['korean', 'hangul', '한국어', 'ko', 'kor', 'ko-kr'],
            'weight': 1.0
        },
        'arabic': {
            'variants': ['arabic', 'عربي', 'ar', 'ara', 'ar-sa', 'ar-eg'],
            'weight': 1.0
        },
        'hindi': {
            'variants': ['hindi', 'हिंदी', 'hi', 'hin', 'hi-in'],
            'weight': 1.0
        },
        'dutch': {
            'variants': ['dutch', 'nederlands', 'nl', 'nld', 'nl-nl', 'nl-be'],
            'weight': 1.0
        },
        'english': {
            'variants': ['english', 'en', 'eng', 'en-us', 'en-gb', 'en-au', 'en-ca'],
            'weight': 1.0
        }
    }
    
    TRANSLATION_KEYWORDS = [
        'translate', 'translation', 'convert', 'transcribe', 'transcript',
        'subtitle', 'subtitles', 'caption', 'captions', 'audio', 'speech', 
        'voice', 'sound', 'spoken', 'change to', 'into', 'language',
        'dub', 'dubbing', 'voice over', 'voiceover'
    ]
    
    LANGUAGE_PATTERNS = [
        r'(?:translate|transcribe|convert|change)\s+(?:to|into|in)\s+(\w+)',
        r'(?:in|as|to)\s+(\w+)\s+(?:language|subtitles|captions)',
        r'(\w+)\s+(?:language|subtitles|captions|translation)',
        r'speak(?:ing)?\s+in\s+(\w+)',
        r'audio\s+in\s+(\w+)',
        r'make\s+it\s+(\w+)',
    ]
    
    @classmethod
    def detect_translation_intent(cls, text: str) -> Tuple[bool, float]:
        """Detect translation intent with confidence scoring."""
        text_lower = text.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in cls.TRANSLATION_KEYWORDS if keyword in text_lower)
        
        # Calculate confidence based on keyword density
        word_count = len(text_lower.split())
        confidence = min(keyword_matches / max(word_count * 0.1, 1), 1.0)
        
        has_intent = keyword_matches > 0
        return has_intent, confidence
    
    @classmethod
    def detect_target_language(cls, text: str) -> Tuple[Optional[str], float]:
        """Detect target language with confidence scoring."""
        text_lower = text.lower()
        best_match = None
        best_confidence = 0.0
        
        # Try pattern matching first (higher confidence)
        for pattern in cls.LANGUAGE_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                potential_lang = match.lower().strip()
                for lang_key, lang_data in cls.LANGUAGE_MAPPING.items():
                    if potential_lang in lang_data['variants']:
                        confidence = 0.9 * lang_data['weight']  # High confidence for pattern matches
                        if confidence > best_confidence:
                            best_match = lang_key
                            best_confidence = confidence
        
        # Fallback to word boundary matching (lower confidence)
        if not best_match:
            for lang_key, lang_data in cls.LANGUAGE_MAPPING.items():
                for variant in lang_data['variants']:
                    if len(variant) <= 2:
                        # For short codes, use exact word matching
                        pattern = r'\b' + re.escape(variant) + r'\b'
                    else:
                        # For longer words, use word boundaries
                        pattern = r'\b' + re.escape(variant) + r'\b'
                    
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        confidence = 0.7 * lang_data['weight']  # Lower confidence for word boundary
                        if confidence > best_confidence:
                            best_match = lang_key
                            best_confidence = confidence
        
        return best_match, best_confidence


class VideoRequestAnalyzer:
    """Main analyzer for video translation requests."""
    
    @classmethod
    def analyze(cls, message_content: str) -> VideoAnalysis:
        """Comprehensive analysis of user request."""
        # Extract video URL and platform
        video_url, platform = VideoURLExtractor.extract(message_content)
        
        # Detect translation intent and language
        has_translation_intent, intent_confidence = LanguageDetector.detect_translation_intent(message_content)
        target_language, language_confidence = LanguageDetector.detect_target_language(message_content)
        
        # Calculate overall confidence
        confidence_score = (intent_confidence + language_confidence) / 2 if has_translation_intent else 0.0
        
        # Create analysis object
        analysis = VideoAnalysis(
            original_message=message_content,
            video_url=video_url,
            has_video_url=video_url is not None,
            has_translation_intent=has_translation_intent,
            target_language=target_language,
            has_target_language=target_language is not None,
            confidence_score=confidence_score,
            platform_detected=platform
        )
        
        # Determine completeness
        analysis.is_complete_request = analysis.has_video_url and analysis.has_target_language
        
        # Identify missing components
        if not analysis.has_video_url:
            analysis.missing_components.append('video_url')
        if not analysis.has_target_language:
            analysis.missing_components.append('target_language')
        
        return analysis


class ResponseGenerator:
    """Generate contextual responses for missing information."""
    
    SUPPORTED_LANGUAGES_TEXT = """
**Supported languages include:**
🇪🇸 Spanish • 🇫🇷 French • 🇩🇪 German • 🇮🇹 Italian • 🇵🇹 Portuguese • 🇷🇺 Russian 
🇨🇳 Chinese • 🇯🇵 Japanese • 🇰🇷 Korean • 🇸🇦 Arabic • 🇮🇳 Hindi • 🇳🇱 Dutch • 🇺🇸 English
"""
    
    SUPPORTED_PLATFORMS_TEXT = """
**Supported video platforms:**
🎥 YouTube • 📺 Vimeo • 🎬 Dailymotion • 🎮 Twitch • 📱 Facebook • 📸 Instagram 
🎵 TikTok • 🐦 Twitter/X • 📰 Reddit • 🎪 Streamable • 📹 BitChute • 🔗 Direct video files
"""
    
    @classmethod
    def generate_missing_info_response(cls, analysis: VideoAnalysis) -> str:
        """Generate response asking for missing information with context."""
        missing = analysis.missing_components
        
        if 'video_url' in missing and 'target_language' in missing:
            return f"""🎥 **Universal Video Translation Assistant**

I'd be happy to help you translate any video's audio! To get started, I need:

**1. Video URL** 📎
Please provide the video link you want to translate from any supported platform.

**2. Target Language** 🌍  
Specify which language you want the audio translated to.

{cls.SUPPORTED_PLATFORMS_TEXT}

{cls.SUPPORTED_LANGUAGES_TEXT}

**Example:** "Please translate this video to Spanish: https://www.youtube.com/watch?v=VIDEO_ID"

What video would you like to translate and to which language?"""
        
        elif 'video_url' in missing:
            target_lang = analysis.target_language.title()
            return f"""🎥 **Ready to translate to {target_lang}!** 

I just need the **video URL** to proceed. Please provide the video link from any supported platform.

{cls.SUPPORTED_PLATFORMS_TEXT}

**Supported formats:**
• YouTube: https://www.youtube.com/watch?v=VIDEO_ID
• Vimeo: https://vimeo.com/VIDEO_ID  
• TikTok: https://www.tiktok.com/@user/video/ID
• And many more platforms!

What video would you like me to translate to {target_lang}?"""
        
        elif 'target_language' in missing:
            platform_info = f" ({analysis.platform_detected.title()})" if analysis.platform_detected != 'unknown' else ""
            return f"""🎥 **Video detected{platform_info}:** {analysis.video_url}

I can translate this video's audio! Just tell me the **target language**.

{cls.SUPPORTED_LANGUAGES_TEXT}

**Examples:** 
• "Translate it to Spanish"
• "I want French subtitles" 
• "Convert to German"

Which language would you prefer?"""
        
        return "I need more information to help you with video audio translation."


class SystemPromptEnhancer:
    """Enhanced system prompt generation with modular approach."""
    
    @classmethod
    def enhance_prompt(cls, base_prompt: str, analysis: VideoAnalysis) -> str:
        """Enhance system prompt based on request analysis."""
        if not analysis.is_complete_request:
            return base_prompt
        
        platform_info = f" (Platform: {analysis.platform_detected.title()})" if analysis.platform_detected != 'unknown' else ""
        
        enhancement = f"""

🎥 **VIDEO AUDIO TRANSLATION CONTEXT DETECTED**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Request Details:**
• Video URL: {analysis.video_url}{platform_info}
• Target Language: {analysis.target_language.title()}
• Confidence Score: {analysis.confidence_score:.2f}
• Task: Translate video audio to {analysis.target_language.title()}

**Priority Instructions:**
1. **Agent Selection**: Prioritize agents with video audio translation capabilities (yt_dlp supported)
2. **Required Parameters**: 
   - video_url: {analysis.video_url}
   - target_language: {analysis.target_language}
   - task_type: video_audio_translation

3. **Workflow Sequence**:
   - Extract audio from video using yt_dlp
   - Transcribe audio to original language text  
   - Translate text to {analysis.target_language.title()}
   - Generate translated transcripts/subtitles

4. **Quality Standards**: Maintain context, meaning, and timing accuracy

**Note**: This is a confirmed video audio translation request with high confidence ({analysis.confidence_score:.1%}).
The video platform is supported by yt_dlp for audio extraction.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return base_prompt + enhancement


@session.bind(name="VideoTranslationMasterAgent", description="Advanced master agent for universal video audio translation")
async def receive_message(
        agent_context: GenAIContext,
        session_id: str,
        user_id: str,
        configs: dict[str, Any],
        files: Optional[list[dict[str, Any]]],
        timestamp: str
):
    """Enhanced message handler with improved analysis and error handling."""
    try:
        # Get chat history
        chat_history = await get_chat_history(
            f"{app_settings.BACKEND_API_URL}/chat",
            session_id=session_id,
            user_id=user_id,
            api_key=app_settings.MASTER_BE_API_KEY,
            max_last_messages=configs.get("max_last_messages", 5)
        )

        # Attach files if present
        if files and chat_history:
            chat_history[-1] = attach_files_to_message(message=chat_history[-1], files=files)

        # Analyze the request
        latest_message_content = chat_history[-1].content if chat_history else ""
        analysis = VideoRequestAnalyzer.analyze(latest_message_content)
        
        # Enhanced logging
        logger.info("🔍 VIDEO TRANSLATION REQUEST ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"📝 Message: {latest_message_content[:100]}...")
        logger.info(f"🎥 Video URL: {'✅ ' + analysis.video_url if analysis.has_video_url else '❌ Missing'}")
        if analysis.platform_detected:
            logger.info(f"🏷️ Platform: {analysis.platform_detected.title()}")
        logger.info(f"🌍 Translation Intent: {'✅' if analysis.has_translation_intent else '❌'}")
        logger.info(f"🎯 Target Language: {'✅ ' + analysis.target_language.title() if analysis.target_language else '❌ Not specified'}")
        logger.info(f"📊 Confidence Score: {analysis.confidence_score:.2%}")
        logger.info(f"✅ Complete Request: {'Yes' if analysis.is_complete_request else 'No'}")
        if analysis.missing_components:
            logger.warning(f"⚠️ Missing: {', '.join(analysis.missing_components)}")
        logger.info("=" * 60)
        
        # Handle incomplete requests
        if not analysis.is_complete_request:
            response = ResponseGenerator.generate_missing_info_response(analysis)
            
            logger.info("🤔 Incomplete request - providing guidance")
            
            trace = {
                "name": "VideoTranslationMasterAgent",
                "input": latest_message_content,
                "output": response,
                "analysis": analysis.__dict__,
                "is_success": True,
                "action": "requested_missing_information"
            }
            
            return {
                "agents_trace": [trace], 
                "response": response, 
                "is_success": True
            }
        
        # Proceed with agent orchestration
        logger.success("✅ Complete video translation request - starting orchestration")
        
        # Setup configuration
        graph_config = {"configurable": {"session": session}, "recursion_limit": 100}
        
        # Enhance system prompt
        base_system_prompt = configs.get("system_prompt", "You are a helpful AI assistant specialized in video audio translation.")
        user_system_prompt = configs.get("user_prompt")
        system_prompt = user_system_prompt or base_system_prompt
        system_prompt = f"{system_prompt}\n\n{FILE_RELATED_SYSTEM_PROMPT}"
        
        enhanced_system_prompt = SystemPromptEnhancer.enhance_prompt(system_prompt, analysis)
        
        init_messages = [
            SystemMessage(content=enhanced_system_prompt),
            *chat_history
        ]
        
        # Get agents and create master agent
        agents = await get_agents(
            url=f"{app_settings.BACKEND_API_URL}/agents/active",
            agent_type="all",
            api_key=app_settings.MASTER_BE_API_KEY,
            user_id=user_id
        )
        
        llm = LLMFactory.create(configs=configs)
        master_agent = ReActMasterAgent(model=llm, agents=agents)
        
        logger.info("🚀 Executing Video Translation Master Agent")
        
        # Execute workflow
        final_state = await master_agent.graph.ainvoke(
            input={"messages": init_messages},
            config=graph_config
        )
        
        response = final_state["messages"][-1].content
        
        logger.success("✅ Video Translation completed successfully")
        
        # Add analysis to trace
        if "trace" in final_state and final_state["trace"]:
            final_state["trace"][0]["video_analysis"] = analysis.__dict__
        
        return {
            "agents_trace": final_state["trace"], 
            "response": response, 
            "is_success": True
        }

    except Exception as e:
        error_message = f"❌ Video Translation Master Agent error: {e}"
        logger.exception(error_message)

        trace = {
            "name": "VideoTranslationMasterAgent",
            "output": error_message,
            "is_success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        return {
            "agents_trace": [trace], 
            "response": "I encountered an error while processing your video translation request. Please try again or contact support.", 
            "is_success": False
        }


async def main():
    """Main entry point."""
    logger.info("🎥 Video Translation Master Agent started")
    await session.process_events()


if __name__ == "__main__":
    asyncio.run(main())


# import asyncio
# import re
# from typing import Any, Optional, Tuple, Dict, List
# from dataclasses import dataclass

# from genai_session.session import GenAISession
# from genai_session.utils.context import GenAIContext
# from langchain_core.messages import SystemMessage
# from loguru import logger

# from agents.react_master_agent import ReActMasterAgent
# from config.settings import Settings
# from llms import LLMFactory
# from prompts import FILE_RELATED_SYSTEM_PROMPT
# from utils.agents import get_agents
# from utils.chat_history import get_chat_history
# from utils.common import attach_files_to_message

# app_settings = Settings()

# session = GenAISession(
#     api_key=app_settings.MASTER_AGENT_API_KEY,
#     ws_url=app_settings.ROUTER_WS_URL
# )


# @dataclass
# class YouTubeAnalysis:
#     """Data class for YouTube request analysis results."""
#     original_message: str
#     youtube_url: Optional[str] = None
#     has_youtube_url: bool = False
#     has_translation_intent: bool = False
#     target_language: Optional[str] = None
#     has_target_language: bool = False
#     is_complete_request: bool = False
#     missing_components: List[str] = None
#     confidence_score: float = 0.0
    
#     def __post_init__(self):
#         if self.missing_components is None:
#             self.missing_components = []


# class YouTubeURLExtractor:
#     """Enhanced YouTube URL extraction with better regex patterns."""
    
#     # More comprehensive YouTube URL patterns
#     YOUTUBE_PATTERNS = [
#         # Standard youtube.com URLs
#         r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:\S*)?',
#         # Short youtu.be URLs  
#         r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})(?:\S*)?',
#         # Embed URLs
#         r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})(?:\S*)?',
#         # Legacy /v/ URLs
#         r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})(?:\S*)?',
#         # Mobile URLs
#         r'(?:https?://)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:\S*)?',
#         # YouTube Shorts
#         r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})(?:\S*)?',
#     ]
    
#     @classmethod
#     def extract(cls, text: str) -> Optional[str]:
#         """Extract YouTube URL from text with enhanced patterns."""
#         for pattern in cls.YOUTUBE_PATTERNS:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 video_id = match.group(1)
#                 # Validate video ID length (YouTube video IDs are exactly 11 characters)
#                 if len(video_id) == 11:
#                     return f"https://www.youtube.com/watch?v={video_id}"
#         return None


# class LanguageDetector:
#     """Enhanced language detection with confidence scoring."""
    
#     # Enhanced language mapping with more variations and regional codes
#     LANGUAGE_MAPPING = {
#         'spanish': {
#             'variants': ['spanish', 'español', 'castellano', 'es', 'spa', 'es-es', 'es-mx', 'es-ar'],
#             'weight': 1.0
#         },
#         'french': {
#             'variants': ['french', 'français', 'francais', 'fr', 'fra', 'fr-fr', 'fr-ca'],
#             'weight': 1.0
#         },
#         'german': {
#             'variants': ['german', 'deutsch', 'de', 'ger', 'deu', 'de-de', 'de-at'],
#             'weight': 1.0
#         },
#         'italian': {
#             'variants': ['italian', 'italiano', 'it', 'ita', 'it-it'],
#             'weight': 1.0
#         },
#         'portuguese': {
#             'variants': ['portuguese', 'português', 'portugues', 'pt', 'por', 'pt-br', 'pt-pt'],
#             'weight': 1.0
#         },
#         'russian': {
#             'variants': ['russian', 'русский', 'ru', 'rus', 'ru-ru'],
#             'weight': 1.0
#         },
#         'chinese': {
#             'variants': ['chinese', 'mandarin', '中文', 'zh', 'chi', 'cmn', 'zh-cn', 'zh-tw'],
#             'weight': 1.0
#         },
#         'japanese': {
#             'variants': ['japanese', 'nihongo', '日本語', 'ja', 'jpn', 'ja-jp'],
#             'weight': 1.0
#         },
#         'korean': {
#             'variants': ['korean', 'hangul', '한국어', 'ko', 'kor', 'ko-kr'],
#             'weight': 1.0
#         },
#         'arabic': {
#             'variants': ['arabic', 'عربي', 'ar', 'ara', 'ar-sa', 'ar-eg'],
#             'weight': 1.0
#         },
#         'hindi': {
#             'variants': ['hindi', 'हिंदी', 'hi', 'hin', 'hi-in'],
#             'weight': 1.0
#         },
#         'dutch': {
#             'variants': ['dutch', 'nederlands', 'nl', 'nld', 'nl-nl', 'nl-be'],
#             'weight': 1.0
#         },
#         'english': {
#             'variants': ['english', 'en', 'eng', 'en-us', 'en-gb', 'en-au', 'en-ca'],
#             'weight': 1.0
#         }
#     }
    
#     TRANSLATION_KEYWORDS = [
#         'translate', 'translation', 'convert', 'transcribe', 'transcript',
#         'subtitle', 'subtitles', 'caption', 'captions', 'audio', 'speech', 
#         'voice', 'sound', 'spoken', 'change to', 'into', 'language',
#         'dub', 'dubbing', 'voice over', 'voiceover'
#     ]
    
#     LANGUAGE_PATTERNS = [
#         r'(?:translate|transcribe|convert|change)\s+(?:to|into|in)\s+(\w+)',
#         r'(?:in|as|to)\s+(\w+)\s+(?:language|subtitles|captions)',
#         r'(\w+)\s+(?:language|subtitles|captions|translation)',
#         r'speak(?:ing)?\s+in\s+(\w+)',
#         r'audio\s+in\s+(\w+)',
#         r'make\s+it\s+(\w+)',
#     ]
    
#     @classmethod
#     def detect_translation_intent(cls, text: str) -> Tuple[bool, float]:
#         """Detect translation intent with confidence scoring."""
#         text_lower = text.lower()
        
#         # Count keyword matches
#         keyword_matches = sum(1 for keyword in cls.TRANSLATION_KEYWORDS if keyword in text_lower)
        
#         # Calculate confidence based on keyword density
#         word_count = len(text_lower.split())
#         confidence = min(keyword_matches / max(word_count * 0.1, 1), 1.0)
        
#         has_intent = keyword_matches > 0
#         return has_intent, confidence
    
#     @classmethod
#     def detect_target_language(cls, text: str) -> Tuple[Optional[str], float]:
#         """Detect target language with confidence scoring."""
#         text_lower = text.lower()
#         best_match = None
#         best_confidence = 0.0
        
#         # Try pattern matching first (higher confidence)
#         for pattern in cls.LANGUAGE_PATTERNS:
#             matches = re.findall(pattern, text_lower)
#             for match in matches:
#                 potential_lang = match.lower().strip()
#                 for lang_key, lang_data in cls.LANGUAGE_MAPPING.items():
#                     if potential_lang in lang_data['variants']:
#                         confidence = 0.9 * lang_data['weight']  # High confidence for pattern matches
#                         if confidence > best_confidence:
#                             best_match = lang_key
#                             best_confidence = confidence
        
#         # Fallback to word boundary matching (lower confidence)
#         if not best_match:
#             for lang_key, lang_data in cls.LANGUAGE_MAPPING.items():
#                 for variant in lang_data['variants']:
#                     if len(variant) <= 2:
#                         # For short codes, use exact word matching
#                         pattern = r'\b' + re.escape(variant) + r'\b'
#                     else:
#                         # For longer words, use word boundaries
#                         pattern = r'\b' + re.escape(variant) + r'\b'
                    
#                     if re.search(pattern, text_lower, re.IGNORECASE):
#                         confidence = 0.7 * lang_data['weight']  # Lower confidence for word boundary
#                         if confidence > best_confidence:
#                             best_match = lang_key
#                             best_confidence = confidence
        
#         return best_match, best_confidence


# class YouTubeRequestAnalyzer:
#     """Main analyzer for YouTube translation requests."""
    
#     @classmethod
#     def analyze(cls, message_content: str) -> YouTubeAnalysis:
#         """Comprehensive analysis of user request."""
#         # Extract YouTube URL
#         youtube_url = YouTubeURLExtractor.extract(message_content)
        
#         # Detect translation intent and language
#         has_translation_intent, intent_confidence = LanguageDetector.detect_translation_intent(message_content)
#         target_language, language_confidence = LanguageDetector.detect_target_language(message_content)
        
#         # Calculate overall confidence
#         confidence_score = (intent_confidence + language_confidence) / 2 if has_translation_intent else 0.0
        
#         # Create analysis object
#         analysis = YouTubeAnalysis(
#             original_message=message_content,
#             youtube_url=youtube_url,
#             has_youtube_url=youtube_url is not None,
#             has_translation_intent=has_translation_intent,
#             target_language=target_language,
#             has_target_language=target_language is not None,
#             confidence_score=confidence_score
#         )
        
#         # Determine completeness
#         analysis.is_complete_request = analysis.has_youtube_url and analysis.has_target_language
        
#         # Identify missing components
#         if not analysis.has_youtube_url:
#             analysis.missing_components.append('youtube_url')
#         if not analysis.has_target_language:
#             analysis.missing_components.append('target_language')
        
#         return analysis


# class ResponseGenerator:
#     """Generate contextual responses for missing information."""
    
#     SUPPORTED_LANGUAGES_TEXT = """
# **Supported languages include:**
# 🇪🇸 Spanish • 🇫🇷 French • 🇩🇪 German • 🇮🇹 Italian • 🇵🇹 Portuguese • 🇷🇺 Russian 
# 🇨🇳 Chinese • 🇯🇵 Japanese • 🇰🇷 Korean • 🇸🇦 Arabic • 🇮🇳 Hindi • 🇳🇱 Dutch • 🇺🇸 English
# """
    
#     @classmethod
#     def generate_missing_info_response(cls, analysis: YouTubeAnalysis) -> str:
#         """Generate response asking for missing information with context."""
#         missing = analysis.missing_components
        
#         if 'youtube_url' in missing and 'target_language' in missing:
#             return f"""🎥 **YouTube Audio Translation Assistant**

# I'd be happy to help you translate a YouTube video's audio! To get started, I need:

# **1. YouTube URL** 📎
# Please provide the YouTube video link you want to translate.

# **2. Target Language** 🌍  
# Specify which language you want the audio translated to.

# {cls.SUPPORTED_LANGUAGES_TEXT}

# **Example:** "Please translate this video to Spanish: https://www.youtube.com/watch?v=VIDEO_ID"

# What YouTube video would you like to translate and to which language?"""
        
#         elif 'youtube_url' in missing:
#             target_lang = analysis.target_language.title()
#             return f"""🎥 **Ready to translate to {target_lang}!** 

# I just need the **YouTube video URL** to proceed. Please provide the YouTube link you want to translate.

# **Supported formats:**
# • https://www.youtube.com/watch?v=VIDEO_ID
# • https://youtu.be/VIDEO_ID  
# • YouTube Shorts links

# What YouTube video would you like me to translate to {target_lang}?"""
        
#         elif 'target_language' in missing:
#             return f"""🎥 **Video detected:** {analysis.youtube_url}

# I can translate this YouTube video's audio! Just tell me the **target language**.

# {cls.SUPPORTED_LANGUAGES_TEXT}

# **Examples:** 
# • "Translate it to Spanish"
# • "I want French subtitles" 
# • "Convert to German"

# Which language would you prefer?"""
        
#         return "I need more information to help you with YouTube audio translation."


# class SystemPromptEnhancer:
#     """Enhanced system prompt generation with modular approach."""
    
#     @classmethod
#     def enhance_prompt(cls, base_prompt: str, analysis: YouTubeAnalysis) -> str:
#         """Enhance system prompt based on request analysis."""
#         if not analysis.is_complete_request:
#             return base_prompt
        
#         enhancement = f"""

# 🎥 **YOUTUBE AUDIO TRANSLATION CONTEXT DETECTED**
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# **Request Details:**
# • YouTube URL: {analysis.youtube_url}
# • Target Language: {analysis.target_language.title()}
# • Confidence Score: {analysis.confidence_score:.2f}
# • Task: Translate YouTube video audio to {analysis.target_language.title()}

# **Priority Instructions:**
# 1. **Agent Selection**: Prioritize agents with YouTube audio translation capabilities
# 2. **Required Parameters**: 
#    - youtube_url: {analysis.youtube_url}
#    - target_language: {analysis.target_language}
#    - task_type: youtube_audio_translation

# 3. **Workflow Sequence**:
#    - Extract audio from YouTube video
#    - Transcribe audio to original language text  
#    - Translate text to {analysis.target_language.title()}
#    - Generate translated transcripts/subtitles

# 4. **Quality Standards**: Maintain context, meaning, and timing accuracy

# **Note**: This is a confirmed YouTube audio translation request with high confidence ({analysis.confidence_score:.1%}).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# """
        
#         return base_prompt + enhancement


# @session.bind(name="YouTubeTranslationMasterAgent", description="Advanced master agent for YouTube audio translation")
# async def receive_message(
#         agent_context: GenAIContext,
#         session_id: str,
#         user_id: str,
#         configs: dict[str, Any],
#         files: Optional[list[dict[str, Any]]],
#         timestamp: str
# ):
#     """Enhanced message handler with improved analysis and error handling."""
#     try:
#         # Get chat history
#         chat_history = await get_chat_history(
#             f"{app_settings.BACKEND_API_URL}/chat",
#             session_id=session_id,
#             user_id=user_id,
#             api_key=app_settings.MASTER_BE_API_KEY,
#             max_last_messages=configs.get("max_last_messages", 5)
#         )

#         # Attach files if present
#         if files and chat_history:
#             chat_history[-1] = attach_files_to_message(message=chat_history[-1], files=files)

#         # Analyze the request
#         latest_message_content = chat_history[-1].content if chat_history else ""
#         analysis = YouTubeRequestAnalyzer.analyze(latest_message_content)
        
#         # Enhanced logging
#         logger.info("🔍 YOUTUBE TRANSLATION REQUEST ANALYSIS")
#         logger.info("=" * 60)
#         logger.info(f"📝 Message: {latest_message_content[:100]}...")
#         logger.info(f"🎥 YouTube URL: {'✅ ' + analysis.youtube_url if analysis.has_youtube_url else '❌ Missing'}")
#         logger.info(f"🌍 Translation Intent: {'✅' if analysis.has_translation_intent else '❌'}")
#         logger.info(f"🎯 Target Language: {'✅ ' + analysis.target_language.title() if analysis.target_language else '❌ Not specified'}")
#         logger.info(f"📊 Confidence Score: {analysis.confidence_score:.2%}")
#         logger.info(f"✅ Complete Request: {'Yes' if analysis.is_complete_request else 'No'}")
#         if analysis.missing_components:
#             logger.warning(f"⚠️ Missing: {', '.join(analysis.missing_components)}")
#         logger.info("=" * 60)
        
#         # Handle incomplete requests
#         if not analysis.is_complete_request:
#             response = ResponseGenerator.generate_missing_info_response(analysis)
            
#             logger.info("🤔 Incomplete request - providing guidance")
            
#             trace = {
#                 "name": "YouTubeTranslationMasterAgent",
#                 "input": latest_message_content,
#                 "output": response,
#                 "analysis": analysis.__dict__,
#                 "is_success": True,
#                 "action": "requested_missing_information"
#             }
            
#             return {
#                 "agents_trace": [trace], 
#                 "response": response, 
#                 "is_success": True
#             }
        
#         # Proceed with agent orchestration
#         logger.success("✅ Complete YouTube translation request - starting orchestration")
        
#         # Setup configuration
#         graph_config = {"configurable": {"session": session}, "recursion_limit": 100}
        
#         # Enhance system prompt
#         base_system_prompt = configs.get("system_prompt", "You are a helpful AI assistant specialized in YouTube audio translation.")
#         user_system_prompt = configs.get("user_prompt")
#         system_prompt = user_system_prompt or base_system_prompt
#         system_prompt = f"{system_prompt}\n\n{FILE_RELATED_SYSTEM_PROMPT}"
        
#         enhanced_system_prompt = SystemPromptEnhancer.enhance_prompt(system_prompt, analysis)
        
#         init_messages = [
#             SystemMessage(content=enhanced_system_prompt),
#             *chat_history
#         ]
        
#         # Get agents and create master agent
#         agents = await get_agents(
#             url=f"{app_settings.BACKEND_API_URL}/agents/active",
#             agent_type="all",
#             api_key=app_settings.MASTER_BE_API_KEY,
#             user_id=user_id
#         )
        
#         llm = LLMFactory.create(configs=configs)
#         master_agent = ReActMasterAgent(model=llm, agents=agents)
        
#         logger.info("🚀 Executing YouTube Translation Master Agent")
        
#         # Execute workflow
#         final_state = await master_agent.graph.ainvoke(
#             input={"messages": init_messages},
#             config=graph_config
#         )
        
#         response = final_state["messages"][-1].content
        
#         logger.success("✅ YouTube Translation completed successfully")
        
#         # Add analysis to trace
#         if "trace" in final_state and final_state["trace"]:
#             final_state["trace"][0]["youtube_analysis"] = analysis.__dict__
        
#         return {
#             "agents_trace": final_state["trace"], 
#             "response": response, 
#             "is_success": True
#         }

#     except Exception as e:
#         error_message = f"❌ YouTube Translation Master Agent error: {e}"
#         logger.exception(error_message)

#         trace = {
#             "name": "YouTubeTranslationMasterAgent",
#             "output": error_message,
#             "is_success": False,
#             "error": str(e),
#             "error_type": type(e).__name__
#         }
        
#         return {
#             "agents_trace": [trace], 
#             "response": "I encountered an error while processing your YouTube translation request. Please try again or contact support.", 
#             "is_success": False
#         }


# async def main():
#     """Main entry point."""
#     logger.info("🎥 YouTube Translation Master Agent started")
#     await session.process_events()


# if __name__ == "__main__":
#     asyncio.run(main())








































# import asyncio
# import re
# from typing import Any, Optional, Tuple

# from genai_session.session import GenAISession
# from genai_session.utils.context import GenAIContext
# from langchain_core.messages import SystemMessage
# from loguru import logger

# from agents.react_master_agent import ReActMasterAgent
# from config.settings import Settings
# from llms import LLMFactory
# from prompts import FILE_RELATED_SYSTEM_PROMPT
# from utils.agents import get_agents
# from utils.chat_history import get_chat_history
# from utils.common import attach_files_to_message

# app_settings = Settings()

# session = GenAISession(
#     api_key=app_settings.MASTER_AGENT_API_KEY,
#     ws_url=app_settings.ROUTER_WS_URL
# )


# def extract_youtube_url(text: str) -> Optional[str]:
#     """
#     Extract YouTube URL from text message with enhanced patterns.
    
#     Args:
#         text: Input text that may contain YouTube URL
        
#     Returns:
#         YouTube URL if found, None otherwise
#     """
#     youtube_patterns = [
#         r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
#         r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
#         r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
#         r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
#     ]
    
#     for pattern in youtube_patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             video_id = match.group(1)
#             # Return normalized YouTube URL
#             return f"https://www.youtube.com/watch?v={video_id}"
    
#     return None


# def detect_translation_intent_and_language(text: str) -> Tuple[bool, Optional[str]]:
#     """
#     Enhanced detection of translation intent and target language.
    
#     Args:
#         text: User message text
        
#     Returns:
#         Tuple of (has_translation_intent, target_language)
#     """
#     import re
    
#     text_lower = text.lower()
    
#     # Enhanced translation keywords
#     translation_keywords = [
#         'translate', 'translation', 'convert', 'transcribe', 'transcript',
#         'subtitle', 'subtitles', 'caption', 'captions', 'audio', 'speech', 
#         'voice', 'sound', 'spoken', 'change to', 'into', 'language'
#     ]
    
#     # Check for translation intent
#     has_translation_intent = any(keyword in text_lower for keyword in translation_keywords)
    
#     if not has_translation_intent:
#         return False, None
    
#     # Enhanced language detection with more variations
#     language_mapping = {
#         'spanish': ['spanish', 'español', 'castellano', 'es', 'spa'],
#         'french': ['french', 'français', 'francais', 'fr', 'fra'],
#         'german': ['german', 'deutsch', 'de', 'ger', 'deu'],
#         'italian': ['italian', 'italiano', 'it', 'ita'],
#         'portuguese': ['portuguese', 'português', 'portugues', 'pt', 'por'],
#         'russian': ['russian', 'русский', 'ru', 'rus'],
#         'chinese': ['chinese', 'mandarin', '中文', 'zh', 'chi', 'cmn'],
#         'japanese': ['japanese', 'nihongo', '日本語', 'ja', 'jpn'],
#         'korean': ['korean', 'hangul', '한국어', 'ko', 'kor'],
#         'arabic': ['arabic', 'عربي', 'ar', 'ara'],
#         'hindi': ['hindi', 'हिंदी', 'hi', 'hin'],
#         'dutch': ['dutch', 'nederlands', 'nl', 'nld'],
#         'english': ['english', 'en', 'eng']
#     }
    
#     target_language = None
    
#     # Enhanced pattern matching for target language
#     patterns = [
#         r'to\s+(\w+)',          # "translate to Spanish"
#         r'in\s+(\w+)',          # "transcribe in French" 
#         r'into\s+(\w+)',        # "convert into German"
#         r'as\s+(\w+)',          # "transcribe as English"
#         r'language\s+(\w+)',    # "language Spanish"
#         r'(\w+)\s+language',    # "Spanish language"
#         r'(\w+)\s+subtitles',   # "Spanish subtitles"
#         r'(\w+)\s+captions'     # "French captions"
#     ]
    
#     # First try pattern matching
#     for pattern in patterns:
#         matches = re.findall(pattern, text_lower)
#         for match in matches:
#             potential_lang = match.lower()
#             # Check if the matched word is a known language
#             for lang_key, lang_variants in language_mapping.items():
#                 if potential_lang in lang_variants:
#                     target_language = lang_key
#                     break
#             if target_language:
#                 break
#         if target_language:
#             break
    
#     # If no pattern match, check for direct language mentions with WORD BOUNDARIES
#     if not target_language:
#         for lang_key, lang_variants in language_mapping.items():
#             for variant in lang_variants:
#                 # Use word boundaries to avoid partial matches
#                 # Skip very short codes (2 chars or less) for word boundary matching
#                 if len(variant) <= 2:
#                     # For short codes, use exact word matching
#                     pattern = r'\b' + re.escape(variant) + r'\b'
#                 else:
#                     # For longer words, use word boundaries
#                     pattern = r'\b' + re.escape(variant) + r'\b'
                
#                 if re.search(pattern, text_lower, re.IGNORECASE):
#                     target_language = lang_key
#                     break
#             if target_language:
#                 break
    
#     return has_translation_intent, target_language


# def analyze_user_request(message_content: str) -> dict[str, Any]:
#     """
#     Enhanced analysis of user request for YouTube URL and translation intent.
    
#     Args:
#         message_content: User's message content
        
#     Returns:
#         Dictionary with detailed analysis results
#     """
#     youtube_url = extract_youtube_url(message_content)
#     has_translation_intent, target_language = detect_translation_intent_and_language(message_content)
    
#     analysis = {
#         'original_message': message_content,
#         'youtube_url': youtube_url,
#         'has_youtube_url': youtube_url is not None,
#         'has_translation_intent': has_translation_intent,
#         'target_language': target_language,
#         'has_target_language': target_language is not None,
#         'is_complete_request': youtube_url is not None and target_language is not None,
#         'missing_components': []
#     }
    
#     # Identify what's missing
#     if not analysis['has_youtube_url']:
#         analysis['missing_components'].append('youtube_url')
#     if not analysis['has_target_language']:
#         analysis['missing_components'].append('target_language')
    
#     return analysis


# def generate_missing_info_response(analysis: dict[str, Any]) -> str:
#     """
#     Generate a response asking for missing information.
    
#     Args:
#         analysis: Request analysis results
        
#     Returns:
#         Response string asking for missing information
#     """
#     missing = analysis['missing_components']
    
#     if 'youtube_url' in missing and 'target_language' in missing:
#         return """🎥 I'd be happy to help you translate a YouTube video's audio! 

# To get started, I need two things:
# 1. **YouTube URL** - Please provide the YouTube video link you want to translate
# 2. **Target Language** - Please specify which language you want the audio translated to

# Example: "Please translate this video to Spanish: https://www.youtube.com/watch?v=VIDEO_ID"

# What YouTube video would you like to translate and to which language?"""
    
#     elif 'youtube_url' in missing:
#         target_lang = analysis['target_language'].title()
#         return f"""🎥 I understand you want to translate audio to {target_lang}! 

# I just need the **YouTube video URL** to proceed. Please provide the YouTube link you want to translate.

# Example: "https://www.youtube.com/watch?v=VIDEO_ID"

# What YouTube video would you like me to translate to {target_lang}?"""
    
#     elif 'target_language' in missing:
#         return f"""🎥 I can see you want to translate this YouTube video: {analysis['youtube_url']}

# I just need to know the **target language**. Which language would you like the audio translated to?

# Supported languages include: Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Dutch, English, and more.

# Example: "Please translate it to Spanish" or "I want French subtitles"

# Which language would you prefer?"""
    
#     return "I need more information to help you with YouTube audio translation."


# def enhance_system_prompt_with_context(base_prompt: str, analysis: dict[str, Any]) -> str:
#     """
#     Enhance system prompt based on request analysis.
    
#     Args:
#         base_prompt: Original system prompt
#         analysis: Request analysis results
        
#     Returns:
#         Enhanced system prompt with context
#     """
#     if not analysis['is_complete_request']:
#         # Don't enhance if the request is incomplete
#         return base_prompt
    
#     enhanced_prompt = base_prompt + f"""

# 🎥 YOUTUBE AUDIO TRANSLATION CONTEXT DETECTED:
# - YouTube URL: {analysis['youtube_url']}
# - Target Language: {analysis['target_language'].title()}
# - Task: Translate YouTube video audio to {analysis['target_language'].title()}

# SPECIALIZED INSTRUCTIONS:
# You have access to agents that can handle YouTube audio translation workflows. For this request:

# 1. **Prioritize YouTube Audio Translation Agents** - Look for agents with capabilities to:
#    - Download audio from YouTube videos
#    - Transcribe audio to text using speech-to-text
#    - Translate transcribed text to target languages
#    - Generate subtitles/captions in target language

# 2. **Required Parameters** - When calling agents, ensure you pass:
#    - youtube_url: {analysis['youtube_url']}
#    - target_language: {analysis['target_language']}
#    - task_type: youtube_audio_translation

# 3. **Workflow Priority** - The ideal workflow should be:
#    - Extract audio from YouTube video
#    - Transcribe audio to original language text
#    - Translate text to {analysis['target_language'].title()}
#    - Provide translated text and/or subtitle files

# 4. **Quality Focus** - Ensure translations maintain context and meaning, especially for audio content.

# IMPORTANT: This is a confirmed YouTube audio translation request. Focus on agents specialized in this workflow.
# """
    
#     return enhanced_prompt


# @session.bind(name="MasterNewAgent", description="Master agent specialized in YouTube audio translation")
# async def receive_message(
#         agent_context: GenAIContext,
#         session_id: str,
#         user_id: str,
#         configs: dict[str, Any],
#         files: Optional[list[dict[str, Any]]],
#         timestamp: str
# ):
#     try:
#         # Get the latest user message
#         chat_history = await get_chat_history(
#             f"{app_settings.BACKEND_API_URL}/chat",
#             session_id=session_id,
#             user_id=user_id,
#             api_key=app_settings.MASTER_BE_API_KEY,
#             max_last_messages=configs.get("max_last_messages", 5)
#         )

#         # Attach files to the latest message if present
#         if files and chat_history:
#             chat_history[-1] = attach_files_to_message(message=chat_history[-1], files=files)

#         # Analyze the latest user message for YouTube URL and translation intent
#         latest_message_content = chat_history[-1].content if chat_history else ""
#         request_analysis = analyze_user_request(latest_message_content)
        
#         # Enhanced logging for debugging
#         logger.info("🔍 YOUTUBE TRANSLATION REQUEST ANALYSIS")
#         logger.info("=" * 50)
#         logger.info(f"📝 Message: {latest_message_content[:100]}...")
#         logger.info(f"🎥 YouTube URL Found: {'✅' if request_analysis['has_youtube_url'] else '❌'}")
#         if request_analysis['youtube_url']:
#             logger.info(f"🔗 URL: {request_analysis['youtube_url']}")
#         logger.info(f"🌍 Translation Intent: {'✅' if request_analysis['has_translation_intent'] else '❌'}")
#         logger.info(f"🎯 Target Language: {'✅ ' + request_analysis['target_language'].title() if request_analysis['target_language'] else '❌ Not specified'}")
#         logger.info(f"✅ Complete Request: {'Yes' if request_analysis['is_complete_request'] else 'No'}")
#         if request_analysis['missing_components']:
#             logger.warning(f"⚠️ Missing: {', '.join(request_analysis['missing_components'])}")
#         logger.info("=" * 50)
        
#         # Check if we have both YouTube URL and target language
#         if not request_analysis['is_complete_request']:
#             # Generate response asking for missing information
#             missing_info_response = generate_missing_info_response(request_analysis)
            
#             logger.info("🤔 Incomplete request - asking for missing information")
            
#             trace = {
#                 "name": "YouTubeTranslationMasterAgent",
#                 "input": latest_message_content,
#                 "output": missing_info_response,
#                 "analysis": request_analysis,
#                 "is_success": True,
#                 "action": "requested_missing_information"
#             }
            
#             return {
#                 "agents_trace": [trace], 
#                 "response": missing_info_response, 
#                 "is_success": True
#             }
        
#         # If we have both YouTube URL and target language, proceed with agent orchestration
#         logger.success("✅ Complete YouTube translation request detected - proceeding with agent orchestration")
        
#         graph_config = {"configurable": {"session": session}, "recursion_limit": 100}

#         base_system_prompt = configs.get("system_prompt", "You are a helpful AI assistant specialized in YouTube audio translation.")
#         user_system_prompt = configs.get("user_prompt")

#         system_prompt = user_system_prompt or base_system_prompt
#         system_prompt = f"{system_prompt}\n\n{FILE_RELATED_SYSTEM_PROMPT}"

#         # Enhance system prompt with YouTube translation context
#         enhanced_system_prompt = enhance_system_prompt_with_context(system_prompt, request_analysis)

#         init_messages = [
#             SystemMessage(content=enhanced_system_prompt),
#             *chat_history
#         ]

#         # Get available agents
#         agents = await get_agents(
#             url=f"{app_settings.BACKEND_API_URL}/agents/active",
#             agent_type="all",
#             api_key=app_settings.MASTER_BE_API_KEY,
#             user_id=user_id
#         )

#         # Create LLM and master agent
#         llm = LLMFactory.create(configs=configs)
#         master_agent = ReActMasterAgent(model=llm, agents=agents)

#         logger.info("🚀 Running YouTube Translation Master Agent with enhanced context")

#         # Execute the master agent workflow
#         final_state = await master_agent.graph.ainvoke(
#             input={"messages": init_messages},
#             config=graph_config
#         )

#         response = final_state["messages"][-1].content

#         logger.success("✅ YouTube Translation Master Agent completed successfully")

#         # Add analysis to the trace
#         if "trace" in final_state and final_state["trace"]:
#             final_state["trace"][0]["youtube_analysis"] = request_analysis

#         return {
#             "agents_trace": final_state["trace"], 
#             "response": response, 
#             "is_success": True
#         }

#     except Exception as e:
#         error_message = f"❌ Unexpected error in YouTube Translation Master Agent: {e}"
#         logger.exception(error_message)

#         trace = {
#             "name": "YouTubeTranslationMasterAgent",
#             "output": error_message,
#             "is_success": False,
#             "error": str(e)
#         }
#         return {
#             "agents_trace": [trace], 
#             "response": "I encountered an error while processing your YouTube translation request. Please try again.", 
#             "is_success": False
#         }


# async def main():
#     logger.info("🎥 YouTube Translation Master Agent started")
#     await session.process_events()


# if __name__ == "__main__":
#     asyncio.run(main())

# import asyncio
# from typing import Any, Optional

# from genai_session.session import GenAISession
# from genai_session.utils.context import GenAIContext
# from langchain_core.messages import SystemMessage
# from loguru import logger

# from agents.react_master_agent import ReActMasterAgent
# from config.settings import Settings
# from llms import LLMFactory
# from prompts import FILE_RELATED_SYSTEM_PROMPT
# from utils.agents import get_agents
# from utils.chat_history import get_chat_history
# from utils.common import attach_files_to_message

# app_settings = Settings()

# session = GenAISession(
#     api_key=app_settings.MASTER_AGENT_API_KEY,
#     ws_url=app_settings.ROUTER_WS_URL
# )


# @session.bind(name="MasterAgent", description="Master agent that orchestrates other agents")
# async def receive_message(
#         agent_context: GenAIContext,
#         session_id: str,
#         user_id: str,
#         configs: dict[str, Any],
#         files: Optional[list[dict[str, Any]]],
#         timestamp: str
# ):
#     try:
#         graph_config = {"configurable": {"session": session}, "recursion_limit": 100}  # recursion_limit can be adjusted

#         base_system_prompt = configs.get("system_prompt")
#         user_system_prompt = configs.get("user_prompt")

#         system_prompt = user_system_prompt or base_system_prompt
#         system_prompt = f"{system_prompt}\n\n{FILE_RELATED_SYSTEM_PROMPT}"

#         chat_history = await get_chat_history(
#             f"{app_settings.BACKEND_API_URL}/chat",
#             session_id=session_id,
#             user_id=user_id,
#             api_key=app_settings.MASTER_BE_API_KEY,
#             max_last_messages=configs.get("max_last_messages", 5)
#         )

#         chat_history[-1] = attach_files_to_message(message=chat_history[-1], files=files) if files else chat_history[-1]
#         init_messages = [
#             SystemMessage(content=system_prompt),
#             *chat_history
#         ]

#         agents = await get_agents(
#             url=f"{app_settings.BACKEND_API_URL}/agents/active",
#             agent_type="all",
#             api_key=app_settings.MASTER_BE_API_KEY,
#             user_id=user_id
#         )

#         llm = LLMFactory.create(configs=configs)
#         master_agent = ReActMasterAgent(model=llm, agents=agents)

#         logger.info("Running Master Agent")

#         final_state = await master_agent.graph.ainvoke(
#             input={"messages": init_messages},
#             config=graph_config
#         )

#         response = final_state["messages"][-1].content

#         logger.success("Master Agent run successfully")

#         return {"agents_trace": final_state["trace"], "response": response, "is_success": True}

#     except Exception as e:
#         error_message = f"Unexpected error while running Master Agent: {e}"
#         logger.exception(error_message)

#         trace = {
#             "name": "MasterAgent",
#             "output": error_message,
#             "is_success": False
#         }
#         return {"agents_trace": [trace], "response": error_message, "is_success": False}


# async def main():
#     logger.info("Master Agent started")
#     await session.process_events()


# if __name__ == "__main__":
#     asyncio.run(main())
