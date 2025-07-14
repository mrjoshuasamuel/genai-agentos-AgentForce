# import asyncio
# import os
# import tempfile
# import uuid
# import re
# from typing import Dict, Any, Optional
# from pathlib import Path
# from urllib.parse import urlparse

# import yt_dlp
# from loguru import logger
# from genai_session.session import GenAISession

# AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkMmFhZjYyYy01MTkzLTQ3MDctODM1Mi01MTk3MzNhNDYzZmUiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.6S5l4y5666Q-KeyqAw_V28Wm2MQyZqxN5oFPpaLgNXg" # noqa: E501

# session = GenAISession(jwt_token=AGENT_JWT)


# class VideoAudioExtractor:
#     """Enhanced video audio extraction with yt_dlp supporting multiple platforms."""
    
#     # Platform detection patterns
#     PLATFORM_PATTERNS = {
#         'youtube': [r'youtube\.com', r'youtu\.be'],
#         'vimeo': [r'vimeo\.com'],
#         'dailymotion': [r'dailymotion\.com', r'dai\.ly'],
#         'twitch': [r'twitch\.tv'],
#         'facebook': [r'facebook\.com'],
#         'instagram': [r'instagram\.com'],
#         'tiktok': [r'tiktok\.com'],
#         'twitter': [r'twitter\.com', r'x\.com'],
#         'reddit': [r'reddit\.com'],
#         'streamable': [r'streamable\.com'],
#         'bitchute': [r'bitchute\.com'],
#     }
    
#     def __init__(self):
#         self.temp_dir = tempfile.mkdtemp(prefix="video_audio_")
#         logger.info(f"🎵 Initialized Video Audio Extractor with temp dir: {self.temp_dir}")
    
#     def detect_platform(self, url: str) -> str:
#         """Detect the platform from video URL."""
#         url_lower = url.lower()
        
#         for platform, patterns in self.PLATFORM_PATTERNS.items():
#             for pattern in patterns:
#                 if re.search(pattern, url_lower):
#                     return platform
        
#         # Check for direct video files
#         if re.search(r'\.(mp4|avi|mov|wmv|flv|webm|mkv|m4v)', url_lower):
#             return 'direct_file'
            
#         return 'unknown'
    
#     def validate_url(self, url: str) -> bool:
#         """Validate if URL is properly formatted and potentially contains video content."""
#         try:
#             parsed = urlparse(url)
            
#             # Must have scheme and netloc
#             if not parsed.scheme or not parsed.netloc:
#                 return False
            
#             # Check for common video indicators
#             video_indicators = [
#                 'video', 'watch', 'play', 'stream', 'tv', 'tube', 'clip',
#                 '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'
#             ]
            
#             url_lower = url.lower()
#             return any(indicator in url_lower for indicator in video_indicators)
            
#         except Exception:
#             return False

#     async def extract_audio(self, video_url: str, audio_format: str = "mp3") -> Dict[str, Any]:
#         """
#         Extract audio from video URL with enhanced options supporting multiple platforms.
        
#         Args:
#             video_url: Video URL from any supported platform
#             audio_format: Output audio format (mp3, wav, m4a, aac, flac)
            
#         Returns:
#             Dictionary with audio file path, metadata, and status
#         """
#         try:
#             # Detect platform
#             platform = self.detect_platform(video_url)
            
#             # Generate unique filename
#             audio_id = str(uuid.uuid4())
#             output_filename = f"audio_{audio_id}.{audio_format}"
#             output_path = os.path.join(self.temp_dir, output_filename)
            
#             # Configure yt-dlp options for audio extraction
#             ydl_opts = {
#                 'format': 'bestaudio/best',
#                 'outtmpl': output_path,
#                 'postprocessors': [{
#                     'key': 'FFmpegExtractAudio',
#                     'preferredcodec': audio_format,
#                     'preferredquality': '192',
#                 }],
#                 'noplaylist': True,
#                 'extractflat': False,
#                 'writethumbnail': False,
#                 'writeinfojson': True,
#                 'ignoreerrors': False,
#                 'no_warnings': False,
#                 'quiet': False,
#                 'verbose': True,
#                 # Add cookies and headers for better compatibility
#                 'cookiefile': None,
#                 'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
            
#             logger.info(f"🎬 Starting audio extraction from {platform.title()}: {video_url}")
            
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 # Extract video info first
#                 info_dict = ydl.extract_info(video_url, download=False)
                
#                 # Validate video accessibility
#                 if info_dict.get('is_live'):
#                     raise ValueError("Cannot extract audio from live streams")
                
#                 if info_dict.get('age_limit', 0) > 0:
#                     logger.warning(f"⚠️ Age-restricted content detected (age limit: {info_dict.get('age_limit')})")
                
#                 # Get video metadata with safe fallbacks
#                 metadata = {
#                     'title': info_dict.get('title', 'Unknown'),
#                     'duration': info_dict.get('duration', 0),
#                     'uploader': info_dict.get('uploader', 'Unknown'),
#                     'upload_date': info_dict.get('upload_date', 'Unknown'),
#                     'view_count': info_dict.get('view_count', 0),
#                     'video_id': info_dict.get('id', 'Unknown'),
#                     'description': (info_dict.get('description', '') or '')[:500],  # Truncate long descriptions
#                     'platform': platform,
#                     'original_url': video_url,
#                     'webpage_url': info_dict.get('webpage_url', video_url),
#                     'ext': info_dict.get('ext', 'unknown'),
#                     'format_id': info_dict.get('format_id', 'unknown')
#                 }
                
#                 logger.info(f"📺 Video info - Title: {metadata['title']}, Duration: {metadata['duration']}s, Platform: {platform.title()}")
                
#                 # Download and extract audio
#                 ydl.download([video_url])
                
#                 # Find the extracted audio file
#                 audio_files = list(Path(self.temp_dir).glob(f"audio_{audio_id}.*"))
                
#                 if not audio_files:
#                     # Try alternative search patterns
#                     audio_files = list(Path(self.temp_dir).glob(f"*{audio_id}*"))
                
#                 if not audio_files:
#                     raise FileNotFoundError(f"Audio extraction failed - no output file found in {self.temp_dir}")
                
#                 actual_output_path = str(audio_files[0])
#                 file_size = os.path.getsize(actual_output_path)
                
#                 if file_size == 0:
#                     raise ValueError("Audio extraction failed - output file is empty")
                
#                 logger.success(f"✅ Audio extracted successfully from {platform.title()}: {actual_output_path} ({file_size / 1024 / 1024:.2f} MB)")
                
#                 return {
#                     'success': True,
#                     'audio_file_path': actual_output_path,
#                     'audio_format': audio_format,
#                     'file_size_mb': round(file_size / 1024 / 1024, 2),
#                     'metadata': metadata,
#                     'extraction_id': audio_id,
#                     'temp_dir': self.temp_dir,
#                     'platform': platform
#                 }
                
#         except yt_dlp.utils.DownloadError as e:
#             error_msg = f"❌ Download error: {str(e)}"
#             logger.error(error_msg)
            
#             return {
#                 'success': False,
#                 'error': error_msg,
#                 'error_type': 'DownloadError',
#                 'video_url': video_url,
#                 'platform': platform,
#                 'suggestion': 'This video might be private, geo-blocked, or require authentication'
#             }
            
#         except yt_dlp.utils.ExtractorError as e:
#             error_msg = f"❌ Extractor error: {str(e)}"
#             logger.error(error_msg)
            
#             return {
#                 'success': False,
#                 'error': error_msg,
#                 'error_type': 'ExtractorError',
#                 'video_url': video_url,
#                 'platform': platform,
#                 'suggestion': 'This platform might not be supported or the URL format is incorrect'
#             }
            
#         except Exception as e:
#             error_msg = f"❌ Audio extraction failed: {str(e)}"
#             logger.error(error_msg)
            
#             return {
#                 'success': False,
#                 'error': error_msg,
#                 'error_type': type(e).__name__,
#                 'video_url': video_url,
#                 'platform': platform
#             }
    
#     def cleanup(self, audio_file_path: Optional[str] = None):
#         """Clean up temporary files."""
#         try:
#             if audio_file_path and os.path.exists(audio_file_path):
#                 os.remove(audio_file_path)
#                 logger.info(f"🗑️ Cleaned up audio file: {audio_file_path}")
            
#             # Clean up temp directory if empty
#             if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
#                 os.rmdir(self.temp_dir)
#                 logger.info(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
                
#         except Exception as e:
#             logger.warning(f"⚠️ Cleanup warning: {e}")


# # Global extractor instance
# extractor = VideoAudioExtractor()


# @session.bind(
#     name="audioextractor", 
#     description="Extracts high-quality audio from videos across multiple platforms (YouTube, Vimeo, TikTok, etc.) in various formats (MP3, WAV, M4A)"
# )
# async def extract_video_audio(
#     video_url: str,
#     audio_format: str = "mp3",
#     agent_context: str = ""
# ) -> Dict[str, Any]:
#     """
#     Extract audio from a video URL from any supported platform.
    
#     Args:
#         video_url (str): Valid video URL from supported platforms
#         audio_format (str): Output audio format - mp3, wav, m4a, aac, flac (default: mp3)
#         agent_context (str): Additional context for the request
        
#     Returns:
#         Dict containing audio file path, metadata, and extraction status
        
#     Supported Platforms:
#         - YouTube (youtube.com, youtu.be)
#         - Vimeo (vimeo.com)
#         - Dailymotion (dailymotion.com, dai.ly)
#         - Twitch (twitch.tv)
#         - Facebook (facebook.com)
#         - Instagram (instagram.com)
#         - TikTok (tiktok.com)
#         - Twitter/X (twitter.com, x.com)
#         - Reddit (reddit.com)
#         - Streamable (streamable.com)
#         - BitChute (bitchute.com)
#         - Direct video files (.mp4, .avi, .mov, etc.)
        
#     Example:
#         result = await extract_video_audio(
#             video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
#             audio_format="mp3"
#         )
#     """
#     logger.info("🎵 Video Audio Extractor Agent called")
#     logger.info(f"📥 Input - URL: {video_url}, Format: {audio_format}")
    
#     # Validate inputs
#     if not video_url:
#         return {
#             'success': False,
#             'error': 'Video URL is required',
#             'error_type': 'ValidationError'
#         }
    
#     # Validate URL format
#     if not extractor.validate_url(video_url):
#         return {
#             'success': False,
#             'error': 'Invalid or unsupported video URL format',
#             'error_type': 'ValidationError',
#             'provided_url': video_url,
#             'suggestion': 'Please provide a valid video URL from a supported platform'
#         }
    
#     # Validate audio format
#     supported_formats = ['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg']
#     if audio_format.lower() not in supported_formats:
#         return {
#             'success': False,
#             'error': f'Unsupported audio format. Supported: {", ".join(supported_formats)}',
#             'error_type': 'ValidationError',
#             'provided_format': audio_format
#         }
    
#     # Detect platform for logging
#     platform = extractor.detect_platform(video_url)
#     logger.info(f"🏷️ Detected platform: {platform.title()}")
    
#     try:
#         # Extract audio
#         result = await extractor.extract_audio(video_url, audio_format.lower())
        
#         if result['success']:
#             logger.success(f"✅ Audio extraction completed successfully from {platform.title()}")
#             return result
#         else:
#             logger.error(f"❌ Audio extraction failed from {platform.title()}: {result.get('error')}")
#             return result
            
#     except Exception as e:
#         error_msg = f"Unexpected error in audio extraction: {str(e)}"
#         logger.exception(error_msg)
        
#         return {
#             'success': False,
#             'error': error_msg,
#             'error_type': type(e).__name__,
#             'video_url': video_url,
#             'platform': platform
#         }


# # Backward compatibility alias
# extract_youtube_audio = extract_video_audio


# async def main():
#     """Main entry point for the Video Audio Extractor Agent."""
#     logger.info("🎵 Video Audio Extractor Agent starting...")
    
#     try:
#         await session.process_events()
#     except KeyboardInterrupt:
#         logger.info("🛑 Video Audio Extractor Agent stopped by user")
#     except Exception as e:
#         logger.exception(f"❌ Video Audio Extractor Agent error: {e}")
#     finally:
#         # Cleanup any remaining temp files
#         try:
#             if hasattr(extractor, 'temp_dir') and os.path.exists(extractor.temp_dir):
#                 import shutil
#                 shutil.rmtree(extractor.temp_dir)
#                 logger.info("🗑️ Cleaned up all temporary files")
#         except Exception as cleanup_error:
#             logger.warning(f"⚠️ Cleanup warning: {cleanup_error}")


# if __name__ == "__main__":
#     asyncio.run(main())








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