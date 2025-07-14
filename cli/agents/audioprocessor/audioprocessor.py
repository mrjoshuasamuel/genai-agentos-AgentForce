import asyncio
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Core dependencies
import torch
from loguru import logger
import yt_dlp

# Transcription model
import whisperx

# GenAI protocol
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxODZjNGVhYS0xNDQ4LTQ5MGYtOTVkYi00OGUxYzg5MGY4MTAiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.sjQbBYHr__Uo1yT37AEc_7BJJy5suC-sCbXgmLdlAC4" # noqa: E501
ROUTER_WS_URL = os.getenv("ROUTER_WS_URL", "ws://localhost:8001/ws")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

session = GenAISession(jwt_token=AGENT_JWT)

class SimpleTranscriptionProcessor:
    """Simple YouTube transcription processor with word-level timestamps"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing transcription processor on device: {self.device}")
        
        # Initialize model lazily
        self._whisper_model = None
        
    @property
    def whisper_model(self):
        """Lazy initialization of WhisperX model"""
        if self._whisper_model is None:
            logger.info("Loading WhisperX model...")
            self._whisper_model = whisperx.load_model(
                "large-v3", 
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
        return self._whisper_model

    def extract_youtube_audio(self, youtube_url: str, output_dir: str):
        """Extract high-quality audio from YouTube video"""
        logger.info(f"Extracting audio from: {youtube_url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'extractflat': False,
            'writethumbnail': False,
            'writeinfojson': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            uploader = info.get('uploader', 'Unknown')
            view_count = info.get('view_count', 0)
            upload_date = info.get('upload_date', 'Unknown')
            
            # Download audio
            ydl.download([youtube_url])
            
            # Find the downloaded file
            safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            audio_file_path = f"{output_dir}/{safe_title}.wav"
            
            return {
                "audio_file_path": audio_file_path,
                "video_title": video_title,
                "duration": duration,
                "uploader": uploader,
                "view_count": view_count,
                "upload_date": upload_date,
                "youtube_url": youtube_url
            }

    def transcribe_with_timestamps(self, audio_file_path: str, language: str = None):
        """Generate word-level transcript with precise timestamps"""
        logger.info("Generating word-level transcript with timestamps...")
        
        # Load audio
        audio = whisperx.load_audio(audio_file_path)
        
        # Transcribe with Whisper
        result = self.whisper_model.transcribe(audio, batch_size=16)
        
        # Detect language if not provided
        if language is None:
            language = result["language"]
        
        # Load alignment model for word-level timestamps
        model_a, metadata = whisperx.load_align_model(
            language_code=language, 
            device=self.device
        )
        
        # Perform alignment to get word-level timestamps
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        # Process segments to extract detailed timing
        detailed_segments = []
        all_words = []
        
        for segment in result["segments"]:
            words_with_timing = []
            
            if "words" in segment:
                for word in segment["words"]:
                    word_data = {
                        "word": word["word"].strip(),
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                        "duration": round(word["end"] - word["start"], 3),
                        "confidence": word.get("score", 1.0)
                    }
                    words_with_timing.append(word_data)
                    all_words.append(word_data)
            
            segment_data = {
                "segment_id": len(detailed_segments),
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "duration": round(segment["end"] - segment["start"], 3),
                "text": segment["text"].strip(),
                "words": words_with_timing,
                "word_count": len(words_with_timing)
            }
            detailed_segments.append(segment_data)
        
        # Create full transcript
        full_transcript = " ".join([seg["text"] for seg in detailed_segments])
        
        return {
            "language": language,
            "segments": detailed_segments,
            "words": all_words,
            "full_transcript": full_transcript,
            "total_duration": detailed_segments[-1]["end"] if detailed_segments else 0,
            "total_segments": len(detailed_segments),
            "total_words": len(all_words)
        }


# Initialize the processor
processor = SimpleTranscriptionProcessor()

@session.bind(
    name="YouTubeTranscriptionProcessor",
    description="Simple YouTube transcription with word-level timestamps"
)
async def transcribe_youtube_video(
    agent_context: GenAIContext,
    session_id: str = "",
    user_id: str = "",
    configs: dict = None,
    files: list = None,
    timestamp: str = ""
):
    """
    Transcribe YouTube video with word-level timestamps
    
    Parameters:
    - configs: Dictionary containing youtube_url and optional language
    """
    try:
        if configs is None:
            configs = {}
            
        # Extract parameters
        youtube_url = configs.get('youtube_url')
        language = configs.get('language')  # Optional: force specific language
        include_words = configs.get('include_words', True)  # Include individual words
        
        if not youtube_url:
            return {
                "is_success": False,
                "error": "YouTube URL is required",
                "results": None
            }
            
        logger.info(f"Transcribing YouTube URL: {youtube_url}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Extract audio
            audio_data = processor.extract_youtube_audio(youtube_url, temp_dir)
            audio_file_path = audio_data["audio_file_path"]
            
            # Step 2: Transcribe with timestamps
            transcript_data = processor.transcribe_with_timestamps(audio_file_path, language)
            
            # Compile results
            results = {
                "video_metadata": {
                    "title": audio_data["video_title"],
                    "duration": audio_data["duration"],
                    "uploader": audio_data["uploader"],
                    "view_count": audio_data["view_count"],
                    "upload_date": audio_data["upload_date"],
                    "youtube_url": audio_data["youtube_url"]
                },
                "transcript": {
                    "language": transcript_data["language"],
                    "full_text": transcript_data["full_transcript"],
                    "segments": transcript_data["segments"],
                    "words": transcript_data["words"] if include_words else [],
                    "total_duration": transcript_data["total_duration"],
                    "total_segments": transcript_data["total_segments"],
                    "total_words": transcript_data["total_words"]
                },
                "processing_info": {
                    "processed_at": datetime.now().isoformat(),
                    "device_used": processor.device,
                    "model_used": "whisperx-large-v3",
                    "language_detected": transcript_data["language"]
                }
            }
            
            logger.success(f"Successfully transcribed: {audio_data['video_title']}")
            logger.info(f"Language: {transcript_data['language']}, Duration: {transcript_data['total_duration']:.1f}s, Words: {transcript_data['total_words']}")
            
            return {
                "is_success": True,
                "results": results,
                "message": f"Transcription complete for '{audio_data['video_title']}' ({transcript_data['total_words']} words)"
            }
            
    except Exception as e:
        error_msg = f"Failed to transcribe YouTube video: {str(e)}"
        logger.error(error_msg)
        return {
            "is_success": False,
            "error": error_msg,
            "results": None
        }

@session.bind(
    name="GetTextAtTime",
    description="Get transcript text at specific timestamp"
)
async def get_text_at_time(
    agent_context: GenAIContext,
    session_id: str = "",
    user_id: str = "",
    configs: dict = None,
    files: list = None,
    timestamp: str = ""
):
    """
    Get transcript text at a specific timestamp
    
    Parameters:
    - configs: Dictionary containing processing_results and timestamp
    """
    try:
        if configs is None:
            configs = {}
            
        # Extract parameters
        results = configs.get('processing_results')
        ts = configs.get('timestamp', 0.0)
        
        # Parse timestamp
        if isinstance(ts, str):
            try:
                ts = float(ts)
            except (ValueError, TypeError):
                ts = 0.0
        elif ts is None:
            ts = 0.0
            
        if not results:
            return {
                "is_success": False,
                "error": "Processing results are required",
                "timestamp": ts
            }
        
        # Get transcript data
        transcript = results.get("results", {}).get("transcript", {})
        if not transcript:
            transcript = results.get("transcript", {})
            
        segments = transcript.get("segments", [])
        words = transcript.get("words", [])
        
        if not segments:
            return {
                "is_success": False,
                "error": "No transcript data found",
                "timestamp": ts
            }
        
        # Find segment containing the timestamp
        found_segment = None
        for segment in segments:
            if segment["start"] <= ts <= segment["end"]:
                found_segment = segment
                break
        
        # Find word at timestamp
        found_word = None
        for word in words:
            if word["start"] <= ts <= word["end"]:
                found_word = word
                break
        
        if found_segment:
            return {
                "is_success": True,
                "timestamp": ts,
                "segment": {
                    "text": found_segment["text"],
                    "start": found_segment["start"],
                    "end": found_segment["end"],
                    "duration": found_segment["duration"]
                },
                "word": found_word if found_word else None,
                "context": {
                    "segment_id": found_segment["segment_id"],
                    "word_count": found_segment["word_count"]
                }
            }
        else:
            return {
                "is_success": False,
                "error": f"No speech found at timestamp {ts} seconds",
                "timestamp": ts,
                "nearest_segments": [
                    {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
                    for seg in segments[:3]  # Show first few segments as reference
                ]
            }
        
    except Exception as e:
        return {
            "is_success": False,
            "error": f"Failed to get text at time: {str(e)}",
            "timestamp": configs.get('timestamp', 0.0) if configs else 0.0
        }

@session.bind(
    name="SearchTranscript",
    description="Search for specific text in transcript with timestamps"
)
async def search_transcript(
    agent_context: GenAIContext,
    session_id: str = "",
    user_id: str = "",
    configs: dict = None,
    files: list = None,
    timestamp: str = ""
):
    """
    Search for specific text in transcript and return timestamps
    
    Parameters:
    - configs: Dictionary containing processing_results and search_text
    """
    try:
        if configs is None:
            configs = {}
            
        # Extract parameters
        results = configs.get('processing_results')
        search_text = configs.get('search_text', '').lower().strip()
        
        if not results or not search_text:
            return {
                "is_success": False,
                "error": "Processing results and search_text are required",
                "search_text": search_text
            }
        
        # Get transcript data
        transcript = results.get("results", {}).get("transcript", {})
        if not transcript:
            transcript = results.get("transcript", {})
            
        segments = transcript.get("segments", [])
        
        if not segments:
            return {
                "is_success": False,
                "error": "No transcript data found",
                "search_text": search_text
            }
        
        # Search for text in segments
        matches = []
        for segment in segments:
            if search_text in segment["text"].lower():
                matches.append({
                    "segment_id": segment["segment_id"],
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"]
                })
        
        return {
            "is_success": True,
            "search_text": search_text,
            "matches_found": len(matches),
            "matches": matches,
            "message": f"Found {len(matches)} matches for '{search_text}'"
        }
        
    except Exception as e:
        return {
            "is_success": False,
            "error": f"Failed to search transcript: {str(e)}",
            "search_text": search_text
        }

async def main():
    logger.info("🎤 Simple YouTube Transcription Agent started")
    logger.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())

# import asyncio
# import os
# import tempfile
# import json
# from typing import Any, Dict, List, Optional, Tuple
# from pathlib import Path
# import logging
# from datetime import datetime
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=SyntaxWarning)

# # Core dependencies
# import torch
# import torchaudio
# import librosa
# import numpy as np
# import pandas as pd
# from loguru import logger
# import yt_dlp
# from pydub import AudioSegment
# import soundfile as sf

# # Advanced ML models
# import whisperx
# from pyannote.audio import Pipeline
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# import speechbrain as sb

# # Updated import for SpeechBrain v1.0+
# try:
#     from speechbrain.inference import EncoderClassifier
# except ImportError:
#     from speechbrain.pretrained import EncoderClassifier

# # GenAI protocol
# from genai_session.session import GenAISession
# from genai_session.utils.context import GenAIContext

# try:
#     from llms import LLMFactory
# except ImportError:
#     logger.warning("LLMFactory not found. Please ensure master-agent/llms is in PYTHONPATH")
#     LLMFactory = None

# AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxODZjNGVhYS0xNDQ4LTQ5MGYtOTVkYi00OGUxYzg5MGY4MTAiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImNiMDhmMzU4LWNkYTAtNDIyOC04ZTFlLWVlYjk3ZjFmYjFhZCJ9.sjQbBYHr__Uo1yT37AEc_7BJJy5suC-sCbXgmLdlAC4" # noqa: E501
# session = GenAISession(jwt_token=AGENT_JWT)
# ROUTER_WS_URL = os.getenv("ROUTER_WS_URL", "ws://localhost:8001/ws")
# HUGGINGFACE_TOKEN = "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"

# class YouTubeTranscriptionProcessor:
#     """Focused ML-powered audio transcription processor"""
    
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Initializing models on device: {self.device}")
        
#         # Initialize models lazily to avoid memory issues
#         self._whisper_model = None
#         self._diarization_pipeline = None
#         self._gender_classifier = None
#         self._speaker_embedding = None
        
#     @property
#     def whisper_model(self):
#         """Lazy initialization of WhisperX model"""
#         if self._whisper_model is None:
#             logger.info("Loading WhisperX model...")
#             self._whisper_model = whisperx.load_model(
#                 "large-v3", 
#                 device=self.device,
#                 compute_type="float16" if self.device == "cuda" else "int8"
#             )
#         return self._whisper_model
    
#     @property
#     def diarization_pipeline(self):
#         """Lazy initialization of speaker diarization pipeline"""
#         if self._diarization_pipeline is None:
#             logger.info("Loading speaker diarization pipeline...")
#             self._diarization_pipeline = Pipeline.from_pretrained(
#                 "pyannote/speaker-diarization-3.1",
#                 use_auth_token=HUGGINGFACE_TOKEN
#             ).to(torch.device(self.device))
#         return self._diarization_pipeline
    
#     @property
#     def gender_classifier(self):
#         """Lazy initialization of gender classification model"""
#         if self._gender_classifier is None:
#             logger.info("Loading gender classification model...")
#             self._gender_classifier = EncoderClassifier.from_hparams(
#                 source="speechbrain/spkrec-xvect-voxceleb",
#                 savedir="pretrained_models/spkrec-xvect-voxceleb"
#             )
#         return self._gender_classifier
    
#     @property
#     def speaker_embedding(self):
#         """Lazy initialization of speaker embedding model"""
#         if self._speaker_embedding is None:
#             logger.info("Loading speaker embedding model...")
#             self._speaker_embedding = PretrainedSpeakerEmbedding(
#                 "speechbrain/spkrec-ecapa-voxceleb",
#                 device=torch.device(self.device)
#             )
#         return self._speaker_embedding

#     def extract_youtube_audio(self, youtube_url: str, output_dir: str):
#         """Extract high-quality audio from YouTube video"""
#         logger.info(f"Extracting audio from: {youtube_url}")
        
#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
#             'postprocessors': [{
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'wav',
#                 'preferredquality': '192',
#             }],
#             'extractflat': False,
#             'writethumbnail': False,
#             'writeinfojson': True,
#         }
        
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             # Extract video info
#             info = ydl.extract_info(youtube_url, download=False)
#             video_title = info.get('title', 'Unknown')
#             duration = info.get('duration', 0)
#             uploader = info.get('uploader', 'Unknown')
            
#             # Download audio
#             ydl.download([youtube_url])
            
#             # Find the downloaded file
#             safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
#             audio_file_path = f"{output_dir}/{safe_title}.wav"
            
#             return {
#                 "audio_file_path": audio_file_path,
#                 "video_title": video_title,
#                 "duration": duration,
#                 "uploader": uploader,
#                 "sample_rate": 16000
#             }

#     def perform_speaker_diarization(self, audio_file_path: str):
#         """Perform speaker diarization with gender detection"""
#         logger.info("Performing speaker diarization...")
        
#         # Load audio
#         waveform, sample_rate = torchaudio.load(audio_file_path)
        
#         # Resample to 16kHz if needed
#         if sample_rate != 16000:
#             resampler = torchaudio.transforms.Resample(sample_rate, 16000)
#             waveform = resampler(waveform)
#             sample_rate = 16000
        
#         # Perform diarization
#         diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        
#         # Extract speaker segments with gender detection
#         speaker_segments = []
#         gender_predictions = {}
        
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             start_time = turn.start
#             end_time = turn.end
            
#             # Extract audio segment for gender detection
#             start_sample = int(start_time * sample_rate)
#             end_sample = int(end_time * sample_rate)
#             segment_audio = waveform[:, start_sample:end_sample]
            
#             # Predict gender
#             gender = self._predict_gender(segment_audio, sample_rate)
#             gender_predictions[speaker] = gender
            
#             speaker_segments.append({
#                 "speaker": speaker,
#                 "start_time": start_time,
#                 "end_time": end_time,
#                 "duration": end_time - start_time,
#                 "gender": gender
#             })
        
#         return {
#             "speaker_segments": speaker_segments,
#             "gender_mapping": gender_predictions,
#             "total_speakers": len(set(gender_predictions.keys()))
#         }

#     def _predict_gender(self, audio_segment, sample_rate: int):
#         """Predict gender from audio segment using acoustic features"""
#         audio_np = audio_segment.squeeze().numpy()
        
#         try:
#             # Fundamental frequency (pitch) analysis
#             f0, voiced_flag, voiced_probs = librosa.pyin(
#                 audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
#             )
            
#             # Calculate mean F0
#             mean_f0 = np.nanmean(f0[voiced_flag])
            
#             # Simple heuristic: Male < 165 Hz, Female > 165 Hz
#             if np.isnan(mean_f0):
#                 return "unknown"
#             elif mean_f0 < 165:
#                 return "male"
#             else:
#                 return "female"
                
#         except Exception as e:
#             logger.warning(f"Gender prediction failed: {e}")
#             return "unknown"

#     def transcribe_with_alignment(self, audio_file_path: str, language: str = None):
#         """Generate word-level transcript with precise timestamps using WhisperX"""
#         logger.info("Generating word-level transcript with alignment...")
        
#         # Load audio
#         audio = whisperx.load_audio(audio_file_path)
        
#         # Transcribe with Whisper
#         result = self.whisper_model.transcribe(audio, batch_size=16)
        
#         # Detect language if not provided
#         if language is None:
#             language = result["language"]
        
#         # Load alignment model
#         model_a, metadata = whisperx.load_align_model(
#             language_code=language, 
#             device=self.device
#         )
        
#         # Perform alignment to get word-level timestamps
#         result = whisperx.align(
#             result["segments"], 
#             model_a, 
#             metadata, 
#             audio, 
#             self.device, 
#             return_char_alignments=False
#         )
        
#         # Process segments to extract detailed timing
#         detailed_segments = []
#         for segment in result["segments"]:
#             words_with_timing = []
#             if "words" in segment:
#                 for word in segment["words"]:
#                     words_with_timing.append({
#                         "word": word["word"],
#                         "start": word["start"],
#                         "end": word["end"],
#                         "score": word.get("score", 1.0)
#                     })
            
#             detailed_segments.append({
#                 "start": segment["start"],
#                 "end": segment["end"],
#                 "text": segment["text"],
#                 "words": words_with_timing
#             })
        
#         return {
#             "language": language,
#             "segments": detailed_segments,
#             "full_transcript": " ".join([seg["text"] for seg in detailed_segments]),
#             "total_duration": detailed_segments[-1]["end"] if detailed_segments else 0
#         }

#     def extract_voice_characteristics(self, audio_file_path: str, speaker_segments):
#         """Extract detailed voice characteristics for each speaker"""
#         logger.info("Extracting voice characteristics...")
        
#         voice_profiles = {}
        
#         # Load full audio
#         waveform, sample_rate = torchaudio.load(audio_file_path)
        
#         for speaker_info in speaker_segments:
#             speaker = speaker_info["speaker"]
#             start_time = speaker_info["start_time"]
#             end_time = speaker_info["end_time"]
            
#             # Extract speaker audio segment
#             start_sample = int(start_time * sample_rate)
#             end_sample = int(end_time * sample_rate)
#             speaker_audio = waveform[:, start_sample:end_sample]
            
#             # Extract voice embedding
#             embedding = self.speaker_embedding(speaker_audio.squeeze())
            
#             # Extract acoustic features
#             audio_np = speaker_audio.squeeze().numpy()
            
#             # Fundamental frequency statistics
#             f0, voiced_flag, voiced_probs = librosa.pyin(
#                 audio_np, fmin=50, fmax=800
#             )
#             f0_mean = np.nanmean(f0[voiced_flag]) if np.any(voiced_flag) else 0
#             f0_std = np.nanstd(f0[voiced_flag]) if np.any(voiced_flag) else 0
            
#             # Spectral features
#             spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate))
#             spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_np, sr=sample_rate))
#             spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_np, sr=sample_rate))
            
#             # MFCC features
#             mfccs = librosa.feature.mfcc(y=audio_np, sr=sample_rate, n_mfcc=13)
#             mfcc_mean = np.mean(mfccs, axis=1)
            
#             voice_profiles[speaker] = {
#                 "embedding": embedding.cpu().numpy().tolist(),
#                 "f0_mean": float(f0_mean),
#                 "f0_std": float(f0_std),
#                 "spectral_centroid": float(spectral_centroid),
#                 "spectral_bandwidth": float(spectral_bandwidth),
#                 "spectral_rolloff": float(spectral_rolloff),
#                 "mfcc_features": mfcc_mean.tolist(),
#                 "gender": speaker_info["gender"]
#             }
        
#         return voice_profiles

#     def generate_timeline_data(self, transcript_data, speaker_data):
#         """Generate comprehensive timeline data"""
#         timeline = []
        
#         for segment in transcript_data["segments"]:
#             segment_start = segment["start"]
#             segment_end = segment["end"]
            
#             # Find corresponding speaker for this time segment
#             speaker_info = None
#             for speaker_seg in speaker_data["speaker_segments"]:
#                 if (speaker_seg["start_time"] <= segment_start <= speaker_seg["end_time"] or
#                     speaker_seg["start_time"] <= segment_end <= speaker_seg["end_time"]):
#                     speaker_info = speaker_seg
#                     break
            
#             timeline.append({
#                 "start_time": segment_start,
#                 "end_time": segment_end,
#                 "duration": segment_end - segment_start,
#                 "text": segment["text"],
#                 "words": segment.get("words", []),
#                 "speaker": speaker_info["speaker"] if speaker_info else "unknown",
#                 "gender": speaker_info["gender"] if speaker_info else "unknown",
#                 "word_count": len(segment["text"].split()),
#                 "speaking_rate": len(segment["text"].split()) / (segment_end - segment_start) if segment_end > segment_start else 0
#             })
        
#         return {
#             "timeline": timeline,
#             "total_segments": len(timeline),
#             "total_duration": timeline[-1]["end_time"] if timeline else 0,
#             "average_speaking_rate": np.mean([t["speaking_rate"] for t in timeline if t["speaking_rate"] > 0])
#         }


# # Initialize the processor
# processor = YouTubeTranscriptionProcessor()

# @session.bind(
#     name="YouTubeTranscriptionProcessor",
#     description="Advanced ML-powered YouTube audio transcription with speaker diarization and gender detection"
# )
# async def process_youtube_transcription(
#     agent_context: GenAIContext,
#     session_id: str = "",
#     user_id: str = "",
#     configs: dict = None,
#     files: list = None,
#     timestamp: str = ""
# ):
#     """
#     Complete YouTube audio transcription pipeline with advanced ML capabilities
    
#     Parameters:
#     - agent_context: GenAI context
#     - session_id: Session identifier
#     - user_id: User identifier  
#     - configs: Configuration dictionary containing youtube_url and options
#     - files: Optional file list
#     - timestamp: Request timestamp
#     """
#     try:
#         if configs is None:
#             configs = {}
            
#         # Extract processing parameters
#         youtube_url = configs.get('youtube_url')
#         include_voice_cloning = configs.get('include_voice_cloning', True)
#         output_format = configs.get('output_format', 'wav')
        
#         if not youtube_url:
#             return {
#                 "is_success": False,
#                 "error": "YouTube URL is required",
#                 "results": None
#             }
            
#         logger.info(f"Processing YouTube URL: {youtube_url}")
        
#         # Create temporary directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Step 1: Extract audio
#             audio_data = processor.extract_youtube_audio(youtube_url, temp_dir)
#             audio_file_path = audio_data["audio_file_path"]
            
#             # Step 2: Speaker diarization and gender detection
#             speaker_data = processor.perform_speaker_diarization(audio_file_path)
            
#             # Step 3: Transcription with word-level alignment
#             transcript_data = processor.transcribe_with_alignment(audio_file_path)
            
#             # Step 4: Extract voice characteristics (if requested)
#             voice_profiles = {}
#             if include_voice_cloning:
#                 voice_profiles = processor.extract_voice_characteristics(
#                     audio_file_path, speaker_data["speaker_segments"]
#                 )
            
#             # Step 5: Generate timeline data
#             timeline_data = processor.generate_timeline_data(transcript_data, speaker_data)
            
#             # Compile comprehensive results
#             results = {
#                 "video_metadata": {
#                     "title": audio_data["video_title"],
#                     "duration": audio_data["duration"],
#                     "uploader": audio_data["uploader"]
#                 },
#                 "audio_analysis": {
#                     "total_speakers": speaker_data["total_speakers"],
#                     "speaker_segments": speaker_data["speaker_segments"],
#                     "gender_distribution": {
#                         gender: len([s for s in speaker_data["speaker_segments"] if s["gender"] == gender])
#                         for gender in set(s["gender"] for s in speaker_data["speaker_segments"])
#                     }
#                 },
#                 "transcript": {
#                     "language": transcript_data["language"],
#                     "full_text": transcript_data["full_transcript"],
#                     "word_level_segments": transcript_data["segments"],
#                     "total_duration": transcript_data["total_duration"]
#                 },
#                 "timeline_data": timeline_data,
#                 "voice_profiles": voice_profiles if include_voice_cloning else {},
#                 "processing_info": {
#                     "processed_at": datetime.now().isoformat(),
#                     "device_used": processor.device,
#                     "models_used": [
#                         "whisperx-large-v3",
#                         "pyannote-speaker-diarization-3.1",
#                         "speechbrain-ecapa-voxceleb"
#                     ]
#                 }
#             }
            
#             logger.success(f"Successfully processed video: {audio_data['video_title']}")
#             return {
#                 "is_success": True,
#                 "results": results,
#                 "message": f"Transcription complete for '{audio_data['video_title']}'"
#             }
            
#     except Exception as e:
#         error_msg = f"Failed to process YouTube transcription: {str(e)}"
#         logger.error(error_msg)
#         return {
#             "is_success": False,
#             "error": error_msg,
#             "results": None
#         }

# @session.bind(
#     name="GetGenderAtTime", 
#     description="Get speaker gender at specific timestamp"
# )
# async def get_gender_at_time(
#     agent_context: GenAIContext,
#     session_id: str = "",
#     user_id: str = "", 
#     configs: dict = None,
#     files: list = None,
#     timestamp: str = "",
#     processing_results: dict = None,
#     timestamp_seconds: float = None
# ):
#     """
#     Get speaker gender and details at a specific timestamp
    
#     Parameters:
#     - agent_context: GenAI context
#     - session_id: Session identifier
#     - user_id: User identifier
#     - configs: Configuration dictionary
#     - files: Optional file list
#     - timestamp: Request timestamp
#     - processing_results: Results from previous transcription
#     - timestamp_seconds: Specific timestamp to check
#     """
#     try:
#         if configs is None:
#             configs = {}
            
#         # Extract parameters from configs or use direct parameters
#         results = configs.get('processing_results', processing_results)
#         ts = configs.get('timestamp', timestamp_seconds)
        
#         # Try to parse timestamp from string if needed
#         if isinstance(ts, str):
#             try:
#                 ts = float(ts)
#             except (ValueError, TypeError):
#                 ts = 0.0
#         elif ts is None:
#             ts = 0.0
            
#         if not results:
#             return {
#                 "is_success": False,
#                 "error": "Processing results are required",
#                 "timestamp": ts
#             }
        
#         # Handle both result structures
#         timeline_data = results.get("results", {}).get("timeline_data", {})
#         if not timeline_data:
#             timeline_data = results.get("timeline_data", {})
            
#         timeline = timeline_data.get("timeline", [])
        
#         if not timeline:
#             return {
#                 "is_success": False,
#                 "error": "No timeline data found in processing results",
#                 "timestamp": ts
#             }
        
#         # Find the segment containing the timestamp
#         for segment in timeline:
#             if segment["start_time"] <= ts <= segment["end_time"]:
#                 return {
#                     "is_success": True,
#                     "timestamp": ts,
#                     "speaker": segment["speaker"],
#                     "gender": segment["gender"],
#                     "text_at_time": segment["text"],
#                     "segment_duration": segment["duration"]
#                 }
        
#         return {
#             "is_success": False,
#             "error": f"No speech found at timestamp {ts} seconds",
#             "timestamp": ts
#         }
        
#     except Exception as e:
#         return {
#             "is_success": False,
#             "error": f"Failed to get gender at time: {str(e)}",
#             "timestamp": configs.get('timestamp', 0.0) if configs else 0.0
#         }

# async def main():
#     logger.info("🎥 YouTube Transcription Agent started")
#     logger.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
#     await session.process_events()

# if __name__ == "__main__":
#     asyncio.run(main())