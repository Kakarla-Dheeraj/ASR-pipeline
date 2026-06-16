from dotenv import load_dotenv

from preprocess import AudioPreprocessor
from vad import VADProcessor
from speech_accumulator import SpeechAccumulator
from language_detector import LanguageDetector
from confidence_manager import ConfidenceManager
from router import Router
from whisper_transcriber import WhisperTranscriber
from saaras_transcriber import SaarasTranscriber

load_dotenv()

AUDIO_FILE = "tamil.mp3"

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

audio, sr = (
    preprocessor.preprocess(
        AUDIO_FILE
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

print(
    f"Speech Duration: "
    f"{speech_duration:.2f}s"
)
if speech_duration >= 120:

    print(
        "Long call detected. "
        "Skipping first 10s of speech."
    )

    speech_segments = (
        accumulator.skip_initial_speech(
            speech_segments,
            skip_seconds=10,
            sampling_rate=sr
        )
    )

window_size = 30

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

    result = detector.detect(
        speech_chunk
    )

    print(
        f"Window: {window_size}s"
    )

    print(result)

    if not (
        confidence_manager.should_retry(
            result["confidence"]
        )
    ):
        break

    window_size += 30

    if window_size > 90:
        break

print("\nFinal Result:")
print(result)

provider = (
    router.get_provider(
        result["language"]
    )
)

print(
    "\nSelected Provider:"
)

print(provider)

if provider == "whisper":

    transcript = whisper.transcribe(
        AUDIO_FILE
    )

elif provider == "saaras":

    transcript = saaras.transcribe(
        AUDIO_FILE,
        result["language"]
    )

else:

    raise ValueError(
        f"Unsupported language: {result['language']}"
    )

print(
    "\nTranscript:"
)

print(transcript)