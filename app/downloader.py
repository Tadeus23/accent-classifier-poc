import requests
from yt_dlp import YoutubeDL
from app.config import logging

def download_video(url: str) -> str:
    logging.info(f"Downloading video from URL: {url}")
    output_path = "video.mp4"
    
    if "youtube" in url or "youtu.be" in url:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'merge_output_format': 'mp4',
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    else:
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    logging.info(f"Saved video to {output_path}")
    return output_path