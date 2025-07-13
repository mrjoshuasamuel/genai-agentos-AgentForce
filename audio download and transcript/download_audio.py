import subprocess

def download_audio(url, output_path="audio1.mp3"):
    cmd = [
        "yt-dlp",
        "-x",                     # Extract audio only
        "--audio-format", "mp3",  # Convert to mp3
        "-o", output_path,        # Output filename
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Download failed:")
        print(result.stderr)
    else:
        print("Download completed successfully.")
        print("Audio saved as:", output_path)

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    download_audio(video_url)
