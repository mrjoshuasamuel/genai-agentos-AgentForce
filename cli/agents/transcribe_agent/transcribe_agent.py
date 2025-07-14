import asyncio
import json
import os
import time
from typing import Annotated, Dict, Any, Optional
import whisper
import language_tool_python
from loguru import logger
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjODJmMGUzNS0wNWI4LTRkNzktOTI5ZS0xYTEwNGFjYzVkYWMiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.Sh3xkM4yn3mEtqeylJMVy0t9ZUp-Dw_tK0EL_X9BMqQ" # noqa: E501

session = GenAISession(jwt_token=AGENT_JWT)


class AudioTranscriptionProcessor:
    """Enhanced audio transcription with Whisper and grammar correction."""
    
    def __init__(self):
        self.whisper_model = None
        self.language_tool = None
        self.supported_languages = {
            'en': 'en-US',
            'es': 'es',
            'fr': 'fr',
            'de': 'de-DE',
            'it': 'it',
            'pt': 'pt-BR',
            'ru': 'ru-RU',
            'zh': 'en-US',  # Fallback to English for Chinese
            'ja': 'en-US',  # Fallback to English for Japanese
            'ko': 'en-US',  # Fallback to English for Korean
            'ar': 'en-US',  # Fallback to English for Arabic
            'hi': 'en-US',  # Fallback to English for Hindi
            'nl': 'nl'
        }
    
    def load_whisper_model(self, model_size: str = "base") -> bool:
        """Load Whisper model with error handling."""
        try:
            if self.whisper_model is None:
                logger.info(f"🤖 Loading Whisper model: {model_size}")
                self.whisper_model = whisper.load_model(model_size)
                logger.success("✅ Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def load_language_tool(self, language: str = "en") -> bool:
        """Load LanguageTool with language detection."""
        try:
            # Map language codes to LanguageTool format
            lang_code = self.supported_languages.get(language, "en-US")
            
            if self.language_tool is None or getattr(self.language_tool, '_lang', None) != lang_code:
                logger.info(f"📝 Loading LanguageTool for: {lang_code}")
                if self.language_tool:
                    self.language_tool.close()
                self.language_tool = language_tool_python.LanguageTool(lang_code)
                self.language_tool._lang = lang_code
                logger.success(f"✅ LanguageTool loaded for {lang_code}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load LanguageTool for {language}: {e}")
            # Fallback to English
            try:
                if self.language_tool:
                    self.language_tool.close()
                self.language_tool = language_tool_python.LanguageTool('en-US')
                logger.info("📝 Fallback to English LanguageTool")
                return True
            except Exception as fallback_e:
                logger.error(f"❌ Failed to load fallback LanguageTool: {fallback_e}")
                return False
    
    async def transcribe_audio_file(
        self,
        audio_file_path: str,
        video_metadata: Dict[str, Any],
        model_size: str = "base",
        enable_grammar_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file and return structured results.
        
        Args:
            audio_file_path: Path to the audio file
            video_metadata: Video metadata from extractor
            model_size: Whisper model size (tiny, base, small, medium, large)
            enable_grammar_correction: Whether to apply grammar correction
        """
        start_time = time.time()
        warnings = []
        
        try:
            # Validate audio file
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            logger.info(f"🎵 Starting transcription of: {os.path.basename(audio_file_path)}")
            logger.info(f"📊 File size: {file_size / (1024 * 1024):.2f} MB")
            
            # Load Whisper model
            if not self.load_whisper_model(model_size):
                raise RuntimeError("Failed to load Whisper model")
            
            # Transcribe audio with Whisper
            logger.info("🎤 Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(audio_file_path, word_timestamps=True)
            
            # Extract transcription data
            raw_transcript = result.get("text", "").strip()
            detected_language = result.get("language", "unknown")
            
            if not raw_transcript:
                raise ValueError("No speech detected in audio file")
            
            logger.info(f"🌍 Detected language: {detected_language}")
            logger.info(f"📝 Raw transcript length: {len(raw_transcript)} characters")
            
            # Extract timestamps
            timestamps = []
            for segment in result.get("segments", []):
                timestamps.append({
                    "start": round(segment.get("start", 0), 2),
                    "end": round(segment.get("end", 0), 2),
                    "text": segment.get("text", "").strip()
                })
            
            # Apply grammar correction if enabled
            corrected_transcript = raw_transcript
            grammar_corrections = 0
            
            if enable_grammar_correction and raw_transcript:
                try:
                    if self.load_language_tool(detected_language):
                        logger.info("✏️ Applying grammar correction...")
                        matches = self.language_tool.check(raw_transcript)
                        corrected_transcript = language_tool_python.utils.correct(raw_transcript, matches)
                        grammar_corrections = len(matches)
                        
                        if grammar_corrections > 0:
                            logger.info(f"📝 Applied {grammar_corrections} grammar corrections")
                        else:
                            logger.info("📝 No grammar corrections needed")
                    else:
                        warnings.append("Grammar correction tool failed to load")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Grammar correction failed: {e}")
                    warnings.append(f"Grammar correction error: {str(e)}")
            
            # Collect audio metadata
            audio_info = {
                "file_path": os.path.abspath(audio_file_path),
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "format": os.path.splitext(audio_file_path)[1].lower().replace('.', ''),
                "transcription_model": model_size,
                "grammar_correction_enabled": enable_grammar_correction,
                "grammar_corrections_applied": grammar_corrections
            }
            
            # Analyze speech characteristics
            total_duration = timestamps[-1]["end"] if timestamps else 0
            word_count = len(corrected_transcript.split())
            speech_rate = round(word_count / max(total_duration / 60, 1), 2) if total_duration > 0 else 0
            
            # Construct comprehensive output
            output = {
                "success": True,
                "data": {
                    "original_text": corrected_transcript,
                    "raw_transcript": raw_transcript,
                    "metadata": {
                        "video_id": video_metadata.get("video_id", "unknown"),
                        "video_title": video_metadata.get("title", "Unknown"),
                        "video_duration": video_metadata.get("duration", 0),
                        "video_platform": video_metadata.get("platform", "unknown"),
                        "detected_language": detected_language,
                        "speech_analysis": {
                            "word_count": word_count,
                            "speech_rate_wpm": speech_rate,
                            "total_speech_duration": round(total_duration, 2),
                            "segment_count": len(timestamps)
                        },
                        "speaker_info": {
                            "gender": "unknown",
                            "confidence": None,
                            "voice_characteristics": {
                                "tone": "unknown",
                                "speed": "normal" if 120 <= speech_rate <= 180 else ("fast" if speech_rate > 180 else "slow")
                            }
                        },
                        "audio_info": audio_info
                    },
                    "timestamps": timestamps,
                    "temp_audio_file": os.path.abspath(audio_file_path)
                },
                "processing_time": round(time.time() - start_time, 2),
                "warnings": warnings
            }
            
            logger.success(f"✅ Transcription completed in {output['processing_time']}s")
            logger.info(f"📊 Stats: {word_count} words, {len(timestamps)} segments, {speech_rate} WPM")
            
            return output
            
        except Exception as e:
            error_msg = f"❌ Transcription failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "audio_file_path": audio_file_path,
                "processing_time": round(time.time() - start_time, 2),
                "warnings": warnings
            }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.language_tool:
                self.language_tool.close()
                self.language_tool = None
                logger.info("🗑️ LanguageTool resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


# Global transcription processor
transcription_processor = AudioTranscriptionProcessor()


@session.bind(
    name="transcribe_agent",
    description="Transcribes audio files using Whisper and applies grammar correction. Accepts audio extractor output directly or individual parameters."
)
async def transcribe_agent(
    agent_context: GenAIContext,
    extractor_output: Annotated[str, "JSON output from audio extractor agent"] = "",
    audio_file_path: Annotated[str, "Path to the audio file to transcribe"] = "",
    video_title: Annotated[str, "Title of the original video"] = "",
    video_id: Annotated[str, "ID of the original video"] = "",
    video_duration: Annotated[int, "Duration of the video in seconds"] = 0,
    video_platform: Annotated[str, "Platform where the video is hosted"] = "",
    model_size: Annotated[str, "Whisper model size: tiny, base, small, medium, large"] = "base",
    enable_grammar_correction: Annotated[bool, "Whether to apply grammar correction"] = True
) -> str:
    """
    Transcribe audio file and return structured JSON with text, timestamps, and metadata.
    
    This agent uses OpenAI's Whisper for speech recognition and LanguageTool for grammar correction.
    It supports multiple languages and provides detailed metadata about the transcription process.
    
    Can accept either:
    1. Direct extractor_output (JSON string from audio extractor agent)
    2. Individual parameters for standalone use
    
    Returns:
        JSON string containing transcription results, timestamps, metadata, and processing info
    """
    logger.info("🎤 Audio Transcription Agent called")
    
    # Parse extractor output if provided
    final_audio_path = audio_file_path
    final_video_metadata = {}
    
    if extractor_output:
        try:
            # Parse extractor output JSON
            extractor_data = json.loads(extractor_output) if isinstance(extractor_output, str) else extractor_output
            
            # Validate extractor output
            if not extractor_data.get("success", False):
                error_result = {
                    "success": False,
                    "error": f"Audio extraction failed: {extractor_data.get('error', 'Unknown error')}",
                    "error_type": "ExtractionError",
                    "extractor_output": extractor_data
                }
                return json.dumps(error_result, indent=2, ensure_ascii=False)
            
            # Extract data from extractor output
            final_audio_path = extractor_data.get("audio_file_path", "")
            metadata = extractor_data.get("metadata", {})
            
            # Map extractor metadata to transcriber format
            final_video_metadata = {
                "title": metadata.get("title", video_title or "Unknown"),
                "video_id": metadata.get("video_id", video_id or "unknown"),
                "duration": metadata.get("duration", video_duration or 0),
                "platform": extractor_data.get("platform", video_platform or metadata.get("platform", "unknown")),
                "uploader": metadata.get("uploader", "Unknown"),
                "upload_date": metadata.get("upload_date", "Unknown"),
                "view_count": metadata.get("view_count", 0),
                "original_url": metadata.get("original_url", ""),
                "webpage_url": metadata.get("webpage_url", ""),
                "file_size_mb": extractor_data.get("file_size_mb", 0),
                "audio_format": extractor_data.get("audio_format", "unknown"),
                "extraction_id": extractor_data.get("extraction_id", "unknown")
            }
            
            logger.info("✅ Using audio extractor output")
            logger.info(f"📥 Extracted - Audio: {os.path.basename(final_audio_path)}")
            logger.info(f"🎬 Video: {final_video_metadata['title']} ({final_video_metadata['platform']})")
            
        except json.JSONDecodeError as e:
            error_result = {
                "success": False,
                "error": f"Failed to parse extractor output JSON: {str(e)}",
                "error_type": "JSONDecodeError",
                "provided_output": extractor_output[:500] if extractor_output else ""
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Failed to process extractor output: {str(e)}",
                "error_type": "ProcessingError"
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)
    
    else:
        # Use individual parameters (backward compatibility)
        if not final_audio_path:
            error_result = {
                "success": False,
                "error": "Either extractor_output or audio_file_path is required",
                "error_type": "ValidationError",
                "suggestion": "Provide the JSON output from audio extractor agent or specify audio_file_path directly"
            }
            return json.dumps(error_result, indent=2, ensure_ascii=False)
        
        final_video_metadata = {
            "title": video_title or "Unknown",
            "video_id": video_id or "unknown", 
            "duration": video_duration or 0,
            "platform": video_platform or "unknown"
        }
        
        logger.info("📥 Using individual parameters")
        logger.info(f"📥 Input - Audio: {os.path.basename(final_audio_path)}, Model: {model_size}")
    
    # Validate final audio path
    if not final_audio_path:
        error_result = {
            "success": False,
            "error": "No valid audio file path found",
            "error_type": "ValidationError"
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)
    
    # Validate model size
    supported_models = ["tiny", "base", "small", "medium", "large"]
    if model_size not in supported_models:
        logger.warning(f"⚠️ Unsupported model size '{model_size}', using 'base'")
        model_size = "base"
    
    try:
        # Perform transcription
        result = await transcription_processor.transcribe_audio_file(
            audio_file_path=final_audio_path,
            video_metadata=final_video_metadata,
            model_size=model_size,
            enable_grammar_correction=enable_grammar_correction
        )
        
        if result["success"]:
            logger.success("✅ Audio transcription completed successfully")
            
            # Add extractor integration info to result
            if extractor_output:
                result["data"]["metadata"]["extractor_integration"] = {
                    "used_extractor_output": True,
                    "audio_extraction_id": final_video_metadata.get("extraction_id", "unknown"),
                    "original_file_size_mb": final_video_metadata.get("file_size_mb", 0),
                    "original_audio_format": final_video_metadata.get("audio_format", "unknown")
                }
            else:
                result["data"]["metadata"]["extractor_integration"] = {
                    "used_extractor_output": False,
                    "input_method": "individual_parameters"
                }
        else:
            logger.error(f"❌ Audio transcription failed: {result.get('error')}")
        
        # Return JSON string
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Unexpected error in transcription agent: {str(e)}"
        logger.exception(error_msg)
        
        error_result = {
            "success": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "audio_file_path": final_audio_path,
            "video_metadata": final_video_metadata
        }
        
        return json.dumps(error_result, indent=2, ensure_ascii=False)


async def main():
    """Main entry point for the Audio Transcription Agent."""
    logger.info("🎤 Audio Transcription Agent starting...")
    
    try:
        await session.process_events()
    except KeyboardInterrupt:
        logger.info("🛑 Audio Transcription Agent stopped by user")
    except Exception as e:
        logger.exception(f"❌ Audio Transcription Agent error: {e}")
    finally:
        # Cleanup resources
        transcription_processor.cleanup()
        logger.info("🗑️ Transcription resources cleaned up")


# Integration example for reference
async def example_integration_workflow():
    """
    Example showing how audio extractor and transcriber work together.
    This is for reference only - not part of the agent execution.
    """
    # Step 1: Extract audio from video
    extractor_result = await extract_video_audio(
        video_url="https://www.youtube.com/watch?v=example",
        audio_format="mp3"
    )
    
    # Step 2: Use extractor output directly in transcriber
    if extractor_result["success"]:
        transcription_result = await transcribe_agent(
            agent_context=None,
            extractor_output=json.dumps(extractor_result),  # Pass full extractor output
            model_size="base",
            enable_grammar_correction=True
        )
        
        # The transcriber automatically extracts all needed info from extractor output:
        # - audio_file_path
        # - video title, ID, duration, platform
        # - file size, format, extraction ID
        # - original URL and metadata
        
        return transcription_result
    else:
        return json.dumps({
            "success": False,
            "error": "Audio extraction failed",
            "extractor_error": extractor_result.get("error")
        })


if __name__ == "__main__":
    asyncio.run(main())