import streamlit as st
import os
import tempfile
from urllib.parse import urlparse
from app.downloader import download_video
from app.extractor import extract_audio
from app.classifier import AccentClassifier
from app.utils import cleanup_temp_files
from app.config import ACCEPTED_FORMATS

def classify_accent(file):
    """
    Classify the accent from a video file.
    
    Args:
        file: Video file uploaded by the user
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(file.read())
        video_path = tmp.name

    audio_path = None
    try:
        audio_path = extract_audio(video_path)
        if not os.path.exists(audio_path):
            return "Error: Failed to extract audio."

        classifier = AccentClassifier()
        label, confidence, metadata = classifier.predict(audio_path)
    
        probs_str = ', '.join(f'{label}: {prob:.2f}%' for label, prob in metadata.items())
        return f"Accent: {label}\nConfidence: {confidence * 100:.2f}%\nProbabilities: {probs_str}"

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        cleanup_temp_files(video_path, audio_path)

def classify_accent_from_url(url: str):
    """
    Classify the accent from a video URL.
    
    Args:
        url (str): URL to a public video (e.g., YouTube, Loom, or direct MP4 link).
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return "Error: Invalid URL provided."
    
    path = parsed_url.path.lower()
    if not any(path.endswith(fmt) for fmt in ACCEPTED_FORMATS) and "youtube" not in url and "youtu.be" not in url:
        return f"Error: URL does not point to a supported format {ACCEPTED_FORMATS} or YouTube video."

    video_path = None
    audio_path = None
    try:
        video_path = download_video(url)
        if not os.path.exists(video_path):
            return "Error: Failed to download video."

        audio_path = extract_audio(video_path)
        if not os.path.exists(audio_path):
            return "Error: Failed to extract audio."

        classifier = AccentClassifier()
        label, confidence, metadata = classifier.predict(audio_path)
    
        probs_str = ', '.join(f'{label}: {prob:.2f}%' for label, prob in metadata.items())
        return f"Accent: {label}\nConfidence: {confidence * 100:.2f}%\nProbabilities: {probs_str}"

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        cleanup_temp_files(video_path, audio_path)

st.title("Accent Classifier from video input")

upload_type = st.selectbox("Select input type", ["URL", "File"])

if upload_type == "URL":
    url_input = st.text_input("Enter Video URL:", "")
    if st.button("Classify Accent"):
        if url_input:
            result = classify_accent_from_url(url_input)
            st.text(result)
        else:
            st.error("Please enter a valid video URL.")
else:
    file = st.file_uploader("Upload a video file", type=["mp4"])
    if file is not None:
        if st.button("Classify Accent"):
            result = classify_accent(file)
            st.text(result)
    else:
        st.warning("Please upload a valid video file.")