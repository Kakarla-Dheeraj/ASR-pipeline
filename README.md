ASR Pipeline
Overview

This project implements a multilingual Automatic Speech Recognition (ASR) pipeline with:

Voice Activity Detection (VAD)
Language Identification using SpeechBrain
Confidence-based language detection retries
Routing to different ASR providers
Groq Whisper for English transcription
Sarvam Saaras for Indian language transcription
Supported Flow

Audio File
↓
Preprocessing
↓
Voice Activity Detection
↓
Speech Accumulation
↓
Language Detection
↓
Provider Selection
↓
Transcription

Project Structure
.
├── pipeline.py
├── preprocess.py
├── vad.py
├── speech_accumulator.py
├── language_detector.py
├── confidence_manager.py
├── router.py
├── whisper_transcriber.py
├── saaras_transcriber.py
├── requirements.txt
└── .env
Python Version

Recommended:

Python 3.11+
Create Virtual Environment
python -m venv .venv

source .venv/bin/activate
Install Dependencies
pip install torch torchaudio

pip install speechbrain

pip install silero-vad

pip install librosa

pip install numpy

pip install python-dotenv

pip install groq

pip install sarvamai
Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key

SARVAM_API_KEY=your_sarvam_api_key

HF_TOKEN=your_huggingface_token
Running

Update the audio path in:

AUDIO_FILE = "sample.mp3"

Then run:

python pipeline.py
Example Output
Window: 30s
{
    "language": "hi",
    "confidence": 0.99
}

Selected Provider:
saaras

Transcript:
...
Current Providers
English

Provider:

Groq Whisper
Indian Languages

Provider:

Sarvam Saaras

Supported language routing:

Hindi
Bengali
Tamil
Telugu
Marathi
Gujarati
Kannada
Malayalam
Assamese
Urdu
Sanskrit
Nepali
Punjabi
Odia
Sindhi