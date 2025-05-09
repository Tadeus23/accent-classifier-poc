import numpy as np
import librosa
import joblib
from app.config import logging

class AccentClassifier:
    def __init__(self, model_path: str = "accent_classifier_model.pkl"):
        """
        Initialize the accent classifier by loading the trained model.
        
        Args:
            model_path (str): Path to the trained model file.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load("scaler.pkl")
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.max_length = 100
        self.labels = ["American", "Australian", "Canadian"]
        
        logging.info(f"Loaded accent classifier model from {model_path}")

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            np.ndarray: Flattened MFCC features.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            y = y / np.max(np.abs(y) + 1e-9)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            if mfcc.shape[1] > self.max_length:
                mfcc = mfcc[:, :self.max_length]
            else:
                mfcc = np.pad(mfcc, ((0, 0), (0, self.max_length - mfcc.shape[1])), mode='constant')
            features = mfcc.flatten()
            return self.scaler.transform([features])[0]
        except Exception as e:
            logging.error(f"Error extracting features from {audio_path}: {e}")
            raise

    def predict(self, audio_path: str) -> tuple[str, float, dict]:
        """
        Predict the accent from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            tuple: (predicted label, confidence score, additional metadata)
        """
        try:
            features = self.extract_features(audio_path)
            probs = self.model.predict_proba([features])[0]
            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx]
            predicted_label = self.labels[predicted_idx]
            metadata = {label: float(prob * 100) for label, prob in zip(self.labels, probs)}
            logging.info(f"Predicted accent: {predicted_label} with confidence {confidence:.4f}")
            return predicted_label, confidence, metadata
        except Exception as e:
            logging.error(f"Error predicting accent for {audio_path}: {e}")
            raise   
