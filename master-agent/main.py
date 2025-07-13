import asyncio
import re
from typing import Any, Optional, Tuple

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


def extract_youtube_url(text: str) -> Optional[str]:
    """
    Extract YouTube URL from text message with enhanced patterns.
    
    Args:
        text: Input text that may contain YouTube URL
        
    Returns:
        YouTube URL if found, None otherwise
    """
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            video_id = match.group(1)
            # Return normalized YouTube URL
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return None


def detect_translation_intent_and_language(text: str) -> Tuple[bool, Optional[str]]:
    """
    Enhanced detection of translation intent and target language.
    
    Args:
        text: User message text
        
    Returns:
        Tuple of (has_translation_intent, target_language)
    """
    text_lower = text.lower()
    
    # Enhanced translation keywords
    translation_keywords = [
        'translate', 'translation', 'convert', 'transcribe', 'transcript',
        'subtitle', 'subtitles', 'caption', 'captions', 'audio', 'speech', 
        'voice', 'sound', 'spoken', 'change to', 'into', 'language'
    ]
    
    # Check for translation intent
    has_translation_intent = any(keyword in text_lower for keyword in translation_keywords)
    
    if not has_translation_intent:
        return False, None
    
    # Enhanced language detection with more variations
    language_mapping = {
        'spanish': ['spanish', 'español', 'castellano', 'es', 'spa'],
        'french': ['french', 'français', 'francais', 'fr', 'fra'],
        'german': ['german', 'deutsch', 'de', 'ger', 'deu'],
        'italian': ['italian', 'italiano', 'it', 'ita'],
        'portuguese': ['portuguese', 'português', 'portugues', 'pt', 'por'],
        'russian': ['russian', 'русский', 'ru', 'rus'],
        'chinese': ['chinese', 'mandarin', '中文', 'zh', 'chi', 'cmn'],
        'japanese': ['japanese', 'nihongo', '日本語', 'ja', 'jpn'],
        'korean': ['korean', 'hangul', '한국어', 'ko', 'kor'],
        'arabic': ['arabic', 'عربي', 'ar', 'ara'],
        'hindi': ['hindi', 'हिंदी', 'hi', 'hin'],
        'dutch': ['dutch', 'nederlands', 'nl', 'nld'],
        'english': ['english', 'en', 'eng']
    }
    
    target_language = None
    
    # Enhanced pattern matching for target language
    patterns = [
        r'to\s+(\w+)',          # "translate to Spanish"
        r'in\s+(\w+)',          # "transcribe in French" 
        r'into\s+(\w+)',        # "convert into German"
        r'as\s+(\w+)',          # "transcribe as English"
        r'language\s+(\w+)',    # "language Spanish"
        r'(\w+)\s+language',    # "Spanish language"
        r'(\w+)\s+subtitles',   # "Spanish subtitles"
        r'(\w+)\s+captions'     # "French captions"
    ]
    
    # First try pattern matching
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            potential_lang = match.lower()
            # Check if the matched word is a known language
            for lang_key, lang_variants in language_mapping.items():
                if potential_lang in lang_variants:
                    target_language = lang_key
                    break
            if target_language:
                break
        if target_language:
            break
    
    # If no pattern match, check for direct language mentions
    if not target_language:
        for lang_key, lang_variants in language_mapping.items():
            for variant in lang_variants:
                if variant in text_lower:
                    target_language = lang_key
                    break
            if target_language:
                break
    
    return has_translation_intent, target_language


def analyze_user_request(message_content: str) -> dict[str, Any]:
    """
    Enhanced analysis of user request for YouTube URL and translation intent.
    
    Args:
        message_content: User's message content
        
    Returns:
        Dictionary with detailed analysis results
    """
    youtube_url = extract_youtube_url(message_content)
    has_translation_intent, target_language = detect_translation_intent_and_language(message_content)
    
    analysis = {
        'original_message': message_content,
        'youtube_url': youtube_url,
        'has_youtube_url': youtube_url is not None,
        'has_translation_intent': has_translation_intent,
        'target_language': target_language,
        'has_target_language': target_language is not None,
        'is_complete_request': youtube_url is not None and target_language is not None,
        'missing_components': []
    }
    
    # Identify what's missing
    if not analysis['has_youtube_url']:
        analysis['missing_components'].append('youtube_url')
    if not analysis['has_target_language']:
        analysis['missing_components'].append('target_language')
    
    return analysis


def generate_missing_info_response(analysis: dict[str, Any]) -> str:
    """
    Generate a response asking for missing information.
    
    Args:
        analysis: Request analysis results
        
    Returns:
        Response string asking for missing information
    """
    missing = analysis['missing_components']
    
    if 'youtube_url' in missing and 'target_language' in missing:
        return """🎥 I'd be happy to help you translate a YouTube video's audio! 

To get started, I need two things:
1. **YouTube URL** - Please provide the YouTube video link you want to translate
2. **Target Language** - Please specify which language you want the audio translated to

Example: "Please translate this video to Spanish: https://www.youtube.com/watch?v=VIDEO_ID"

What YouTube video would you like to translate and to which language?"""
    
    elif 'youtube_url' in missing:
        target_lang = analysis['target_language'].title()
        return f"""🎥 I understand you want to translate audio to {target_lang}! 

I just need the **YouTube video URL** to proceed. Please provide the YouTube link you want to translate.

Example: "https://www.youtube.com/watch?v=VIDEO_ID"

What YouTube video would you like me to translate to {target_lang}?"""
    
    elif 'target_language' in missing:
        return f"""🎥 I can see you want to translate this YouTube video: {analysis['youtube_url']}

I just need to know the **target language**. Which language would you like the audio translated to?

Supported languages include: Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Dutch, English, and more.

Example: "Please translate it to Spanish" or "I want French subtitles"

Which language would you prefer?"""
    
    return "I need more information to help you with YouTube audio translation."


def enhance_system_prompt_with_context(base_prompt: str, analysis: dict[str, Any]) -> str:
    """
    Enhance system prompt based on request analysis.
    
    Args:
        base_prompt: Original system prompt
        analysis: Request analysis results
        
    Returns:
        Enhanced system prompt with context
    """
    if not analysis['is_complete_request']:
        # Don't enhance if the request is incomplete
        return base_prompt
    
    enhanced_prompt = base_prompt + f"""

🎥 YOUTUBE AUDIO TRANSLATION CONTEXT DETECTED:
- YouTube URL: {analysis['youtube_url']}
- Target Language: {analysis['target_language'].title()}
- Task: Translate YouTube video audio to {analysis['target_language'].title()}

SPECIALIZED INSTRUCTIONS:
You have access to agents that can handle YouTube audio translation workflows. For this request:

1. **Prioritize YouTube Audio Translation Agents** - Look for agents with capabilities to:
   - Download audio from YouTube videos
   - Transcribe audio to text using speech-to-text
   - Translate transcribed text to target languages
   - Generate subtitles/captions in target language

2. **Required Parameters** - When calling agents, ensure you pass:
   - youtube_url: {analysis['youtube_url']}
   - target_language: {analysis['target_language']}
   - task_type: youtube_audio_translation

3. **Workflow Priority** - The ideal workflow should be:
   - Extract audio from YouTube video
   - Transcribe audio to original language text
   - Translate text to {analysis['target_language'].title()}
   - Provide translated text and/or subtitle files

4. **Quality Focus** - Ensure translations maintain context and meaning, especially for audio content.

IMPORTANT: This is a confirmed YouTube audio translation request. Focus on agents specialized in this workflow.
"""
    
    return enhanced_prompt


@session.bind(name="MasterNewAgent", description="Master agent specialized in YouTube audio translation")
async def receive_message(
        agent_context: GenAIContext,
        session_id: str,
        user_id: str,
        configs: dict[str, Any],
        files: Optional[list[dict[str, Any]]],
        timestamp: str
):
    try:
        # Get the latest user message
        chat_history = await get_chat_history(
            f"{app_settings.BACKEND_API_URL}/chat",
            session_id=session_id,
            user_id=user_id,
            api_key=app_settings.MASTER_BE_API_KEY,
            max_last_messages=configs.get("max_last_messages", 5)
        )

        # Attach files to the latest message if present
        if files and chat_history:
            chat_history[-1] = attach_files_to_message(message=chat_history[-1], files=files)

        # Analyze the latest user message for YouTube URL and translation intent
        latest_message_content = chat_history[-1].content if chat_history else ""
        request_analysis = analyze_user_request(latest_message_content)
        
        # Enhanced logging for debugging
        logger.info("🔍 YOUTUBE TRANSLATION REQUEST ANALYSIS")
        logger.info("=" * 50)
        logger.info(f"📝 Message: {latest_message_content[:100]}...")
        logger.info(f"🎥 YouTube URL Found: {'✅' if request_analysis['has_youtube_url'] else '❌'}")
        if request_analysis['youtube_url']:
            logger.info(f"🔗 URL: {request_analysis['youtube_url']}")
        logger.info(f"🌍 Translation Intent: {'✅' if request_analysis['has_translation_intent'] else '❌'}")
        logger.info(f"🎯 Target Language: {'✅ ' + request_analysis['target_language'].title() if request_analysis['target_language'] else '❌ Not specified'}")
        logger.info(f"✅ Complete Request: {'Yes' if request_analysis['is_complete_request'] else 'No'}")
        if request_analysis['missing_components']:
            logger.warning(f"⚠️ Missing: {', '.join(request_analysis['missing_components'])}")
        logger.info("=" * 50)
        
        # Check if we have both YouTube URL and target language
        if not request_analysis['is_complete_request']:
            # Generate response asking for missing information
            missing_info_response = generate_missing_info_response(request_analysis)
            
            logger.info("🤔 Incomplete request - asking for missing information")
            
            trace = {
                "name": "YouTubeTranslationMasterAgent",
                "input": latest_message_content,
                "output": missing_info_response,
                "analysis": request_analysis,
                "is_success": True,
                "action": "requested_missing_information"
            }
            
            return {
                "agents_trace": [trace], 
                "response": missing_info_response, 
                "is_success": True
            }
        
        # If we have both YouTube URL and target language, proceed with agent orchestration
        logger.success("✅ Complete YouTube translation request detected - proceeding with agent orchestration")
        
        graph_config = {"configurable": {"session": session}, "recursion_limit": 100}

        base_system_prompt = configs.get("system_prompt", "You are a helpful AI assistant specialized in YouTube audio translation.")
        user_system_prompt = configs.get("user_prompt")

        system_prompt = user_system_prompt or base_system_prompt
        system_prompt = f"{system_prompt}\n\n{FILE_RELATED_SYSTEM_PROMPT}"

        # Enhance system prompt with YouTube translation context
        enhanced_system_prompt = enhance_system_prompt_with_context(system_prompt, request_analysis)

        init_messages = [
            SystemMessage(content=enhanced_system_prompt),
            *chat_history
        ]

        # Get available agents
        agents = await get_agents(
            url=f"{app_settings.BACKEND_API_URL}/agents/active",
            agent_type="all",
            api_key=app_settings.MASTER_BE_API_KEY,
            user_id=user_id
        )

        # Create LLM and master agent
        llm = LLMFactory.create(configs=configs)
        master_agent = ReActMasterAgent(model=llm, agents=agents)

        logger.info("🚀 Running YouTube Translation Master Agent with enhanced context")

        # Execute the master agent workflow
        final_state = await master_agent.graph.ainvoke(
            input={"messages": init_messages},
            config=graph_config
        )

        response = final_state["messages"][-1].content

        logger.success("✅ YouTube Translation Master Agent completed successfully")

        # Add analysis to the trace
        if "trace" in final_state and final_state["trace"]:
            final_state["trace"][0]["youtube_analysis"] = request_analysis

        return {
            "agents_trace": final_state["trace"], 
            "response": response, 
            "is_success": True
        }

    except Exception as e:
        error_message = f"❌ Unexpected error in YouTube Translation Master Agent: {e}"
        logger.exception(error_message)

        trace = {
            "name": "YouTubeTranslationMasterAgent",
            "output": error_message,
            "is_success": False,
            "error": str(e)
        }
        return {
            "agents_trace": [trace], 
            "response": "I encountered an error while processing your YouTube translation request. Please try again.", 
            "is_success": False
        }


async def main():
    logger.info("🎥 YouTube Translation Master Agent started")
    await session.process_events()


if __name__ == "__main__":
    asyncio.run(main())

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
