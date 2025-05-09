import logging
import logging.handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler("accent_classifier.log", maxBytes=10*1024*1024, backupCount=5)
    ]
)

ACCEPTED_FORMATS = ['.mp4', '.mov', '.webm', '.avi']
AUDIO_FILENAME = "audio.wav"