import whisper

# Load the Whisper model
model = whisper.load_model("base")
print("[INFO] Model loaded.")

# Transcribe the audio file
result = model.transcribe("audio1.mp3")
print("[INFO] Transcription result:")

# Extract raw text
raw_transcript = result["text"]
print("[INFO] Raw Transcript:")
print(raw_transcript)

# Detect language
detected_lang = result["language"]

# Map ISO codes to LanguageTool codes and readable names
lang_mapping = {
    "en": ("en-US", "English"),
    "de": ("de-DE", "German"),
    "fr": ("fr", "French"),
    "es": ("es", "Spanish"),
    "pt": ("pt", "Portuguese"),
    "it": ("it", "Italian"),
    "nl": ("nl", "Dutch"),
    "pl": ("pl", "Polish"),
    "ru": ("ru", "Russian"),
}

lt_entry = lang_mapping.get(detected_lang)
if lt_entry:
    lt_code, lang_name = lt_entry
    print(f"[INFO] Detected language: {lang_name} ({detected_lang})")
    do_correction = True
else:
    print(f"[INFO] Detected language '{detected_lang}' is not supported by LanguageTool. Skipping correction.")
    do_correction = False

# Save raw transcript
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(raw_transcript)
print("[INFO] Transcript saved to transcript.txt")

# If supported, run LanguageTool correction
if do_correction:
    import language_tool_python

    # Initialize the LanguageTool client
    tool = language_tool_python.LanguageTool(lt_code)
    print(f"[INFO] LanguageTool initialized for {lang_name}")

    # Correct grammar and spelling
    matches = tool.check(raw_transcript)
    corrected_transcript = language_tool_python.utils.correct(raw_transcript, matches)

    # Print and save corrected transcript
    print("[INFO] Corrected Transcript:")
    print(corrected_transcript)

    with open("corrected_transcript.txt", "w", encoding="utf-8") as f:
        f.write(corrected_transcript)
    print("[INFO] Corrected transcript saved to corrected_transcript.txt")
else:
    # If not supported, inform user and just save raw again
    print("[INFO] Skipped correction due to unsupported language.")
