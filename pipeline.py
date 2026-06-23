from dotenv import load_dotenv
import os

from preprocess import AudioPreprocessor
from vad import VADProcessor
from speech_accumulator import SpeechAccumulator
from language_detector import LanguageDetector
from confidence_manager import ConfidenceManager
from router import Router
from whisper_transcriber import WhisperTranscriber
from saaras_transcriber import SaarasTranscriber
from diarizer import Diarizer
from audio_segmenter import AudioSegmenter

load_dotenv()

preprocessor = AudioPreprocessor()
vad = VADProcessor()
accumulator = SpeechAccumulator()
detector = LanguageDetector()

confidence_manager = (
    ConfidenceManager()
)

router = Router()

whisper = WhisperTranscriber()
saaras = SaarasTranscriber()
diarizer = Diarizer(
    os.getenv("HF_TOKEN")
)
segmenter = AudioSegmenter()


def run_asr_pipeline(audio_file_path: str, speaker_names: dict = None) -> dict:
    if speaker_names is None:
        speaker_names = {}

    audio, sr = (
        preprocessor.preprocess(
            audio_file_path
        )
    )

    speech_segments = (
        vad.get_speech_segments(
            audio,
            sr
        )
    )
    speech_duration = (
        accumulator.total_speech_duration(
            speech_segments,
            sr
        )
    )

    if speech_duration >= 120:
        speech_segments = (
            accumulator.skip_initial_speech(
                speech_segments,
                skip_seconds=10,
                sampling_rate=sr
            )
        )

    window_size = 30
    previous_chunk_len = 0
    result = {"language": "unknown", "confidence": 0.0}

    while True:
        speech_chunk = (
            accumulator.collect_speech(
                audio,
                speech_segments,
                target_duration_sec=window_size,
                sampling_rate=sr
            )
        )

        if len(
            speech_chunk
        ) == 0:
            raise ValueError(
                "No speech detected"
            )

        if len(speech_chunk) <= previous_chunk_len:
            break

        previous_chunk_len = len(speech_chunk)

        result = detector.detect(
            speech_chunk
        )

        if not (
            confidence_manager.should_retry(
                result["confidence"]
            )
        ):
            break

        window_size += 30

        if window_size > 90:
            break

    provider = (
        router.get_provider(
            result["language"]
        )
    )

    speaker_segments = (
        diarizer.diarize(
            audio_file_path
        )
    )

    chunks = (
        segmenter.split(
            audio,
            sr,
            speaker_segments
        )
    )

    dialogue = []

    for chunk in chunks:
        speaker = chunk["speaker"]
        path = chunk["path"]
        start = chunk["start"]
        end = chunk["end"]

        try:
            if provider == "whisper":
                text = (
                    whisper.transcribe(
                        path
                    )
                )

            elif provider == "saaras":
                text = (
                    saaras.transcribe(
                        path,
                        result["language"]
                    )
                )

            else:
                continue

            dialogue.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text
                }
            )

        except Exception as e:
            print(
                f"Failed for {path}: {e}"
            )
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_err:
                    print(f"Failed to delete {path}: {cleanup_err}")

    speakers = {
        segment["speaker"]
        for segment in speaker_segments
    }

    transcript_lines = []
    clean_lines = []
    for line in dialogue:
        speaker_name = speaker_names.get(line['speaker'], line['speaker'])
        transcript_lines.append(f"{speaker_name}: {line['text']}")
        clean_lines.append(line['text'])

    formatted_transcript = "\n".join(transcript_lines)
    clean_transcript = "\n".join(clean_lines)

    return {
        "analysis": {
            "language": result["language"],
            "confidence": result["confidence"],
            "provider": provider,
            "speech_duration": speech_duration,
            "speakers_count": len(speakers)
        },
        "dialogue": dialogue,
        "transcript": formatted_transcript,
        "clean_transcript": clean_transcript
    }


if __name__ == "__main__":
    AUDIO_FILE = "audio_files/Maruti.wav"
    output = run_asr_pipeline(AUDIO_FILE)

    analysis = output["analysis"]
    dialogue = output["dialogue"]

    print("\n" + "=" * 60)
    print("CALL ANALYSIS")
    print("=" * 60)

    print(
        f"Language           : "
        f"{analysis['language']}"
    )

    print(
        f"Confidence         : "
        f"{analysis['confidence']:.4f}"
    )

    print(
        f"Provider           : "
        f"{analysis['provider']}"
    )

    print(
        f"Speech Duration    : "
        f"{analysis['speech_duration']:.2f}s"
    )

    print(
        f"Speakers Detected  : "
        f"{analysis['speakers_count']}"
    )

    print("\n" + "=" * 60)
    print("DIALOGUE")
    print("=" * 60)

    for line in dialogue:
        print(
            f"\n{line['speaker']} ({line['start']:.2f}s - {line['end']:.2f}s):"
        )
        print(
            line["text"]
        )