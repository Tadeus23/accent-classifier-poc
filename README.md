# Accent Classifier

This project implements an AI-based accent classifier that predicts English accents (American, Australian, Canadian) from audio extracted from video URLs. It uses Mel-frequency cepstral coefficients (MFCC) features and a multilayer perceptron (MLP) classifier, with a CLI interface powered by typer. The tool is designed for evaluating spoken English in hiring processes, supporting public video URLs (e.g., YouTube, Loom, or direct MP4 links).

## Prerequisites
- Python 3.12.3
- Git
- FFmpeg (required for `pydub` audio processing)
- A dataset directory (e.g., `data/`) with subdirectories `American`, `Australian`, and `Canadian` containing `.wav` or `.flac` audio files.

## Setup Instructions

### 1. Clone the Repository
Clone the project to your local machine:

```bash
git clone https://github.com/your-username/accent-classifier.git
cd accent-classifier
```

### 2. Create a Virtual Environment
Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg
Ensure FFmpeg is installed for audio extraction:
- **Ubuntu/Debian**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Windows**: Download from [FFmpeg's official site](https://ffmpeg.org/download.html) and add to your PATH.

### 5. Update/Prepare the Dataset (Optional)
Place your prefered audio dataset in a `data/` directory with the following structure:

```
data/
├── American/
│   ├── file1.wav
│   ├── file2.flac
├── Australian/
│   ├── file3.wav
│   ├── file4.flac
├── Canadian/
│   ├── file5.wav
│   ├── file6.flac
```

Ensure audio files are in `.wav` or `.flac` format.

## Running the Application

### 1. Train the Model
Run the trainer to fine-tune the Wav2Vec2 model:

```bash
python trainer.py
```

This will:
- Load audio files from `data/`.
- Extract Wav2Vec2 features using `Wav2Vec2Processor`.
- Fine-tune the Wav2Vec2 model.
- Save the model and processor to `accent_classifier_model/`.

Logs are saved to `accent_classifier.log`.

### 2. Classify a Video URL
Use the CLI to classify an accent from a video URL (e.g., YouTube, Loom, or direct MP4 link):

```bash
python main.py classify https://www.youtube.com/watch?v=example
```

This will:
- Download the video.
- Extract audio (16 kHz, mono, normalized).
- Predict the accent using Wav2Vec2.
- Output the predicted accent, confidence score, and probability distribution.

Example output:
```
Accent: American
Confidence: 92.34%
Probabilities: {'American': 0.9234, 'Australian': 0.0456, 'Canadian': 0.0310}
```

Logs are saved to `accent_classifier.log`.

## Notes
- Ensure the model and processor (`accent_classifier_model/`) are present for inference.
- Supported video formats: `.mp4`, `.mov`, `.webm`, `.avi`, and YouTube videos.
- Temporary files are automatically cleaned up after processing.
- The Wav2Vec2 model requires significant memory; a GPU is recommended for training.

## Troubleshooting
- **FFmpeg errors**: Ensure FFmpeg is installed and accessible in your PATH.
- **Dataset issues**: Verify the `data/` directory structure and file formats.
- **URL errors**: Use valid public video URLs pointing to supported formats or YouTube videos.
- **Log inspection**: Check `accent_classifier.log` for detailed error messages.
- **CUDA errors**: If using a GPU, ensure PyTorch is installed with CUDA support (`pip install torch==2.0.1+cu118`).