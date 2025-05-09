import os
import numpy as np
import librosa
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib
from app.config import logging
import random

class AccentTrainer:
    def __init__(self, data_dir: str, model_path: str = "accent_classifier_model.pkl"):
        """
        Initialize the accent trainer.

        Args:
            data_dir (str): Directory containing VoxForge accent-labeled audio files.
            model_path (str): Path to save the trained model.
        """
        self.data_dir = data_dir
        self.model_path = model_path
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.max_length = 100
        self.labels = ["American", "Australian", "Canadian"]
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: Flattened MFCC features with fixed dimensions.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

            if mfcc.shape[1] > self.max_length:
                mfcc = mfcc[:, :self.max_length]
            else:
                mfcc = np.pad(mfcc, ((0, 0), (0, self.max_length - mfcc.shape[1])), mode='constant') 

            return mfcc.flatten()
        except Exception as e:
            logging.error(f"Error extracting features from {audio_path}: {e}")
            return None

    def augment_audio(self, audio_path: str) -> np.ndarray:
        """
        Augment the audio by adding noise and pitch shifting.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: Augmented audio features with fixed dimensions.
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        noise = np.random.randn(len(y)) * 0.005
        y_noisy = y + noise
        n_steps = random.randint(-3, 3)
        y_pitched = librosa.effects.pitch_shift(y_noisy, sr=sr, n_steps=n_steps)
        mfcc = librosa.feature.mfcc(y=y_pitched, sr=sr, n_mfcc=self.n_mfcc)

        if mfcc.shape[1] > self.max_length:
            mfcc = mfcc[:, :self.max_length]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.max_length - mfcc.shape[1])), mode='constant')

        return mfcc.flatten()



    def load_voxforge_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the VoxForge dataset, augment, and extract features.

        Returns:
            tuple: (features, labels)
        """
        features = []
        labels = []

        for label in self.labels:
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                logging.warning(f"Directory {label_dir} not found")
                continue

            for filename in os.listdir(label_dir):
                if filename.endswith((".wav", ".flac")):
                    audio_path = os.path.join(label_dir, filename)
                    feature = self.extract_features(audio_path)

                    if random.random() > 0.5:
                        feature = self.augment_audio(audio_path)

                    if feature is not None:
                        features.append(feature)
                        labels.append(self.labels.index(label))
                        logging.info(f"Processed {audio_path}")

        return np.array(features), np.array(labels)

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        param_grid = {
            'hidden_layer_sizes': [(100, 50), (200, 100), (300,)],
            'max_iter': [200, 500, 1000],
            'activation': ['relu', 'tanh'],
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train(self):
        """
        Train the accent classification model, perform cross-validation, and save it.
        """
        try:
            X, y = self.load_voxforge_dataset()
            if len(X) == 0:
                raise ValueError("No valid audio files found in the dataset")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            best_model = self.hyperparameter_tuning(X_train, y_train)

            cross_val_scores = cross_val_score(best_model, X, y, cv=5)
            logging.info(f"Cross-validation scores: {cross_val_scores}")
            logging.info(f"Mean cross-validation score: {np.mean(cross_val_scores):.4f}")

            best_model.fit(X_train, y_train)

            train_score = best_model.score(X_train, y_train)
            test_score = best_model.score(X_test, y_test)
            logging.info(f"Training accuracy: {train_score:.4f}")
            logging.info(f"Test accuracy: {test_score:.4f}")

            joblib.dump(best_model, self.model_path)
            joblib.dump(self.scaler, "scaler.pkl")
            logging.info(f"Saved model to {self.model_path}")

        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise


if __name__ == "__main__":
    trainer = AccentTrainer(data_dir="data")
    trainer.train()