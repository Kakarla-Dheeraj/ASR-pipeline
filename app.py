import os
import shutil
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pipeline import run_asr_pipeline

app = FastAPI(
    title="ASR Multilingual Pipeline API",
    description="Multilingual ASR pipeline with Voice Activity Detection, Speaker Diarization, and Routing.",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    speaker_names: str = Form(None)
):
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats are wav, mp3, flac, ogg, m4a."
        )

    # Parse optional speaker names mapping
    names_dict = None
    if speaker_names:
        try:
            names_dict = json.loads(speaker_names)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="speaker_names must be a valid JSON string, e.g. '{\"SPEAKER_00\": \"Agent\"}'"
            )

    # Generate a unique temporary path for the uploaded file
    temp_filename = f"temp_upload_{uuid.uuid4().hex}{file_ext}"
    temp_filepath = os.path.abspath(temp_filename)

    try:
        # Save the uploaded file
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run the ASR pipeline
        result = run_asr_pipeline(temp_filepath, speaker_names=names_dict)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ASR pipeline processing failed: {str(e)}"
        )

    finally:
        # Clean up the temporary uploaded file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception as cleanup_err:
                print(f"Failed to delete temporary upload {temp_filepath}: {cleanup_err}")

