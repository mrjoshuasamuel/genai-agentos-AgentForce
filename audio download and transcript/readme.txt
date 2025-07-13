# GenAI Audio Agent

This project downloads audio from a URL and transcribes it using OpenAI Whisper. It also corrects grammar and spelling using LanguageTool.

## Setup

1️⃣ **Create a virtual environment**

  *Mac/Linux

  python3 -m venv venv

  *Windows

  python -m venv venv

  
2️⃣ Activate the virtual environment

  *Mac/Linux

  source venv/bin/activate

  *windows
  venv\Scripts\activate

3️⃣ Install dependencies
   pip install -r requirements.txt


**Usage
**Download audio

  python download_audio.py
The downloaded Audio will be saved in the file audio1.mp3
**Transcribe audio

  python transcribe_audio.py


The transcribed text will be saved in transcript.txt and the corrected version in corrected_transcript1.txt