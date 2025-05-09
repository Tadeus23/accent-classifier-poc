from pydub import AudioSegment
from app.config import AUDIO_FILENAME, logging
import pydub
import os

def extract_audio(video_path: str, audio_path: str = AUDIO_FILENAME, target_sample_rate: int = 16000) -> str:
    """
    Extract audio from a video file using pydub, convert to mono, and set target sample rate.
    
    Args:
        video_path (str): Path to the input video file.
        audio_path (str, optional): Path to save the output audio file. Defaults to AUDIO_FILENAME.
        target_sample_rate (int, optional): Target sample rate in Hz. Defaults to 16000.
    
    Returns:
        str: Path to the saved audio file.
    """
    logging.info(f"Extracting audio from: {video_path}")
    audio_path = video_path.replace(".mp4", ".wav")
    sound = pydub.AudioSegment.from_file(video_path)
    sound.export(audio_path, format="wav")
    return audio_path