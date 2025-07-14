import asyncio
import os
import tempfile
import uuid
from typing import Dict, Any, Optional
from pathlib import Path

import yt_dlp
from loguru import logger
from genai_session.session import GenAISession

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkMmFhZjYyYy01MTkzLTQ3MDctODM1Mi01MTk3MzNhNDYzZmUiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.6S5l4y5666Q-KeyqAw_V28Wm2MQyZqxN5oFPpaLgNXg" # noqa: E501

session = GenAISession(jwt_token=AGENT_JWT)


class YouTubeAudioExtractor:
    """Enhanced YouTube audio extraction with proper error handling."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="youtube_audio_")
        logger.info(f"🎵 Initialized YouTube Audio Extractor with temp dir: {self.temp_dir}")
    
    async def extract_audio(self, youtube_url: str, audio_format: str = "mp3") -> Dict[str, Any]:
        """
        Extract audio from YouTube video with enhanced options.
        
        Args:
            youtube_url: YouTube video URL
            audio_format: Output audio format (mp3, wav, m4a)
            
        Returns:
            Dictionary with audio file path, metadata, and status
        """
        try:
            # Generate unique filename
            audio_id = str(uuid.uuid4())
            output_filename = f"audio_{audio_id}.{audio_format}"
            output_path = os.path.join(self.temp_dir, output_filename)
            
            # Configure yt-dlp options for audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': audio_format,
                    'preferredquality': '192',
                }],
                'noplaylist': True,
                'extractflat': False,
                'writethumbnail': False,
                'writeinfojson': True,
                'ignoreerrors': False,
                'no_warnings': False,
                'quiet': False,
                'verbose': True
            }
            
            logger.info(f"🎬 Starting audio extraction from: {youtube_url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first
                info_dict = ydl.extract_info(youtube_url, download=False)
                
                # Validate video accessibility
                if info_dict.get('is_live'):
                    raise ValueError("Cannot extract audio from live streams")
                
                if info_dict.get('age_limit', 0) > 0:
                    logger.warning(f"⚠️ Age-restricted content detected (age limit: {info_dict.get('age_limit')})")
                
                # Get video metadata
                metadata = {
                    'title': info_dict.get('title', 'Unknown'),
                    'duration': info_dict.get('duration', 0),
                    'uploader': info_dict.get('uploader', 'Unknown'),
                    'upload_date': info_dict.get('upload_date', 'Unknown'),
                    'view_count': info_dict.get('view_count', 0),
                    'video_id': info_dict.get('id', 'Unknown'),
                    'description': info_dict.get('description', '')[:500],  # Truncate long descriptions
                }
                
                logger.info(f"📺 Video info - Title: {metadata['title']}, Duration: {metadata['duration']}s")
                
                # Download and extract audio
                ydl.download([youtube_url])
                
                # Verify output file exists
                expected_output = output_path.replace(f'.{audio_format}', f'.{audio_format}')
                audio_files = list(Path(self.temp_dir).glob(f"audio_{audio_id}.*"))
                
                if not audio_files:
                    raise FileNotFoundError(f"Audio extraction failed - no output file found")
                
                actual_output_path = str(audio_files[0])
                file_size = os.path.getsize(actual_output_path)
                
                if file_size == 0:
                    raise ValueError("Audio extraction failed - output file is empty")
                
                logger.success(f"✅ Audio extracted successfully: {actual_output_path} ({file_size / 1024 / 1024:.2f} MB)")
                
                return {
                    'success': True,
                    'audio_file_path': actual_output_path,
                    'audio_format': audio_format,
                    'file_size_mb': round(file_size / 1024 / 1024, 2),
                    'metadata': metadata,
                    'extraction_id': audio_id,
                    'temp_dir': self.temp_dir
                }
                
        except Exception as e:
            error_msg = f"❌ Audio extraction failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'error_type': type(e).__name__,
                'youtube_url': youtube_url
            }
    
    def cleanup(self, audio_file_path: Optional[str] = None):
        """Clean up temporary files."""
        try:
            if audio_file_path and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logger.info(f"🗑️ Cleaned up audio file: {audio_file_path}")
            
            # Clean up temp directory if empty
            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)
                logger.info(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
                
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


# Global extractor instance
extractor = YouTubeAudioExtractor()


@session.bind(
    name="audioextractor", 
    description="Extracts high-quality audio from YouTube videos in various formats (MP3, WAV, M4A)"
)
async def extract_youtube_audio(
    youtube_url: str,
    audio_format: str = "mp3",
    agent_context: str = ""
) -> Dict[str, Any]:
    """
    Extract audio from a YouTube video URL.
    
    Args:
        youtube_url (str): Valid YouTube video URL
        audio_format (str): Output audio format - mp3, wav, or m4a (default: mp3)
        agent_context (str): Additional context for the request
        
    Returns:
        Dict containing audio file path, metadata, and extraction status
        
    Example:
        result = await extract_youtube_audio(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            audio_format="mp3"
        )
    """
    logger.info("🎵 YouTube Audio Extractor Agent called")
    logger.info(f"📥 Input - URL: {youtube_url}, Format: {audio_format}")
    
    # Validate inputs
    if not youtube_url:
        return {
            'success': False,
            'error': 'YouTube URL is required',
            'error_type': 'ValidationError'
        }
    
    # Validate YouTube URL format
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
    ]
    
    import re
    is_valid_youtube_url = any(re.search(pattern, youtube_url, re.IGNORECASE) for pattern in youtube_patterns)
    
    if not is_valid_youtube_url:
        return {
            'success': False,
            'error': 'Invalid YouTube URL format',
            'error_type': 'ValidationError',
            'provided_url': youtube_url
        }
    
    # Validate audio format
    supported_formats = ['mp3', 'wav', 'm4a', 'aac', 'flac']
    if audio_format.lower() not in supported_formats:
        return {
            'success': False,
            'error': f'Unsupported audio format. Supported: {", ".join(supported_formats)}',
            'error_type': 'ValidationError',
            'provided_format': audio_format
        }
    
    try:
        # Extract audio
        result = await extractor.extract_audio(youtube_url, audio_format.lower())
        
        if result['success']:
            logger.success(f"✅ Audio extraction completed successfully")
            return result
        else:
            logger.error(f"❌ Audio extraction failed: {result.get('error')}")
            return result
            
    except Exception as e:
        error_msg = f"Unexpected error in audio extraction: {str(e)}"
        logger.exception(error_msg)
        
        return {
            'success': False,
            'error': error_msg,
            'error_type': type(e).__name__,
            'youtube_url': youtube_url
        }


async def main():
    """Main entry point for the YouTube Audio Extractor Agent."""
    logger.info("🎵 YouTube Audio Extractor Agent starting...")
    
    try:
        await session.process_events()
    except KeyboardInterrupt:
        logger.info("🛑 YouTube Audio Extractor Agent stopped by user")
    except Exception as e:
        logger.exception(f"❌ YouTube Audio Extractor Agent error: {e}")
    finally:
        # Cleanup any remaining temp files
        try:
            if hasattr(extractor, 'temp_dir') and os.path.exists(extractor.temp_dir):
                import shutil
                shutil.rmtree(extractor.temp_dir)
                logger.info("🗑️ Cleaned up all temporary files")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    asyncio.run(main())