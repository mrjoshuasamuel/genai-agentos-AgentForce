import os
import logging
from dotenv import load_dotenv
from murf import Murf

# Load .env
load_dotenv()

# Load API key
MURF_API_KEY = os.getenv("MURF_API_KEY")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Murf client
murf_client = Murf(api_key=MURF_API_KEY)

def generate_tts_audio(text: str, voice_id: str) -> str:
    """
    Generate TTS audio using Murf API and return audio URL
    """
    logger.info(f"Generating TTS audio with Murf voice_id={voice_id}")
    try:
        res = murf_client.text_to_speech.generate(
            text=text,
            voice_id=voice_id,
            multi_native_locale = "fr-FR"
        )
        audio_url = res.audio_file
        logger.info(f"TTS audio generated: {audio_url}")
        return audio_url
    except Exception as e:
        logger.error(f"Murf TTS generation failed: {e}")
        raise RuntimeError("Failed to generate audio with Murf.")

def main():
    """
    Main function to parse text input and generate audio.
    """
    try:
        translated_transcript = input("Enter text to synthesize: ").strip()
        voice_id = input("Enter Murf voice ID: ").strip()

        if not translated_transcript:
            raise ValueError("Input text cannot be empty.")

        audio_url = generate_tts_audio(translated_transcript, voice_id)

        print("\n✅ TTS audio generated successfully.")
        print("Download URL:", audio_url)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
