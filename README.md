# Multilingual ASR & Diarization Pipeline

This project implements a multilingual Automatic Speech Recognition (ASR) and speaker diarization pipeline. It automatically detects the spoken language, splits the audio by speaker turns, and routes transcription requests to the best-suited provider.

## Features

- **Voice Activity Detection (VAD)**: Extracts active speech regions using Silero VAD.
- **Language Detection (LID)**: Identifies the language using SpeechBrain, with confidence-based retry logic.
- **Speaker Diarization**: Identifies different speakers and their speech boundaries using PyAnnote.
- **Smart Turn Merging**: Groups consecutive segments from the same speaker to optimize API efficiency.
- **ASR Routing**: 
  - **Groq Whisper** for English transcription.
  - **Sarvam Saaras** for Indian regional languages (Hindi, Marathi, Tamil, Telugu, etc.).
- **FastAPI Wrapper**: Exposes the pipeline via a web service.

---

## Pipeline Flow

```text
  [Audio File Input]
          │
          ▼
   [Preprocessing] (Normalized mono audio at 16kHz)
          │
          ▼
   [VAD Processor] (Extract speech regions)
          │
          ▼
   [Speech Accumulator] (Collect sliding window of speech)
          │
          ▼
   [Language Identifier] ─── (Confidence < 0.70?) ───► [Increase Window & Retry]
          │
      (Accept)
          ▼
   [Speaker Diarizer] (PyAnnote turn boundary detection)
          │
          ▼
   [Audio Segmenter] (Merge consecutive turns & split chunks)
          │
          ▼
   [ASR Router] ───────────────┬────────────────
                               │
                       (Indian Language?)
                               ▼
                        [Sarvam Saaras]
                               or
                           (English?)
                               ▼
                         [Groq Whisper]
          │
          ▼
   [Formatted Dialogues / Clean Transcript Output]
```

---

## Getting Started

### 1. Install Dependencies
Make sure you have **Python 3.14** installed, then run:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install torch torchaudio silero-vad speechbrain librosa numpy python-dotenv groq sarvamai fastapi uvicorn python-multipart soundfile
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
HF_TOKEN=your_huggingface_token  # Required for PyAnnote diarization models
```

---

## Usage

### Run as a Script
To test the pipeline locally on a sample audio file:
1. Update `AUDIO_FILE` inside the `__main__` block in `pipeline.py`.
2. Run:
```bash
python pipeline.py
```

### Run as a FastAPI Service
Start the API server locally:
```bash
uvicorn app:app --port 8000 --reload
```

### Run with Docker
You can also run the application inside a container to avoid setting up dependencies locally:

1. **Build the Docker image**:
   ```bash
   docker build -t asr-pipeline .
   ```

2. **Run the container**:
   Pass your local `.env` variables to the container to authenticate with the API providers:
   ```bash
   docker run -p 8000:8000 --env-file .env asr-pipeline
   ```


---

## API Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {"status": "healthy"}
  ```

### 2. Transcribe Audio
- **Endpoint**: `POST /transcribe`
- **Request Type**: `multipart/form-data`
- **Parameters**:
  - `file` (Required): The audio file (supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
  - `speaker_names` (Optional): A JSON string mapping default speaker IDs to custom names.
    - Example: `{"SPEAKER_00": "Agent", "SPEAKER_01": "Customer"}`

- **Example request with curl**:
  ```bash
  curl -X POST \
    -F "file=@audio.wav" \
    -F 'speaker_names={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}' \
    http://127.0.0.1:8000/transcribe
  ```

- **Response Format**:
  ```json
  {
    "analysis": {
      "language": "hi",
      "confidence": 0.8357,
      "provider": "saaras",
      "speech_duration": 19.87,
      "speakers_count": 2
    },
    "dialogue": [
      {
        "speaker": "SPEAKER_01",
        "start": 0.76,
        "end": 1.65,
        "text": "और क्या चल रहा है?"
      },
      {
        "speaker": "SPEAKER_00",
        "start": 2.21,
        "end": 3.56,
        "text": "काही नाही, काय सुरू आहे?"
      }
    ],
    "transcript": "Bob: और क्या चल रहा है?\nAlice: काही नाही, काय सुरू आहे?",
    "clean_transcript": "और क्या चल रहा है?\nकाही नाही, काय सुरू आहे?"
  }
  ```