import json
import os
import time
import whisper
import language_tool_python

# Load video info from JSON file
with open("video_info.json", "r", encoding="utf-8") as f:
    video_info = json.load(f)

# Start timer
start_time = time.time()

# Load Whisper model
model = whisper.load_model("base")
print("[INFO] Model loaded.")

# Transcribe audio
result = model.transcribe("audio1.wav", word_timestamps=True)

# Extract text and timestamps
raw_transcript = result["text"]

timestamps = []
for segment in result["segments"]:
    timestamps.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

# Detect language
detected_lang = result.get("language", "unknown")

# Initialize LanguageTool (default English)
tool = language_tool_python.LanguageTool('en-US')

# Correct transcript
matches = tool.check(raw_transcript)
corrected_transcript = language_tool_python.utils.correct(raw_transcript, matches)

# Collect audio metadata (dummy values for illustration)
audio_info = {
    "sample_rate": 44100,
    "format": "wav",
    "file_size_mb": round(os.path.getsize("audio1.wav") / (1024 * 1024), 2)
}

# Construct output dictionary
output = {
    "success": True,
    "data": {
        "original_text": corrected_transcript,
        "metadata": {
            "video_id": video_info["video_id"],
            "video_title": video_info["video_title"],
            "video_duration": video_info["video_duration"],
            "detected_language": detected_lang,
            "speaker_info": {
                "gender": "unknown",
                "confidence": None,
                "voice_characteristics": {
                    "tone": "unknown",
                    "speed": "unknown"
                }
            },
            "audio_info": audio_info
        },
        "timestamps": timestamps,
        "temp_audio_file": os.path.abspath("audio1.wav")
    },
    "processing_time": round(time.time() - start_time, 2),
    "warnings": []
}

# Print final output
print(json.dumps(output, indent=2, ensure_ascii=False))
