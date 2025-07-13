import json
import yt_dlp

def download_audio():
    url = input("Enter video URL: ")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloaded_audio.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get('id')
        title = info.get('title')
        duration = info.get('duration')

    video_info = {
        "video_id": video_id,
        "video_title": title,
        "video_duration": duration
    }

    # Save metadata to JSON file
    with open("video_info.json", "w", encoding="utf-8") as f:
        json.dump(video_info, f, ensure_ascii=False, indent=2)

    print("Audio downloaded as 'audio1.wav'")
    print("Video info saved to 'video_info.json'")

if __name__ == "__main__":
    download_audio()
