from sarvamai import SarvamAI
import os
import json

LANGUAGE_MAP = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "as": "as-IN",
    "ur": "ur-IN",
    "sa": "sa-IN",
    "ne": "ne-IN",
    "pa": "pa-IN",
    "or": "or-IN",
    "sd": "sd-IN",
}

class SaarasTranscriber:

    def __init__(self):

        self.client = SarvamAI(
            api_subscription_key=os.getenv(
                "SARVAM_API_KEY"
            )
        )

    def transcribe(
        self,
        audio_path,
        language
    ):
        language_code = LANGUAGE_MAP.get(
            language
        )

        if language_code is None:

            raise ValueError(
                f"Unsupported language: {language}"
            )

        job = (
            self.client.speech_to_text_job.create_job(
                model="saaras:v3",
                mode="transcribe",
            )
        )

        job.upload_files(
            file_paths=[audio_path]
        )

        job.start()

        job.wait_until_complete()

        file_results = (
            job.get_file_results()
        )

        if (
            len(
                file_results["failed"]
            ) > 0
        ):

            raise RuntimeError(
                file_results["failed"][0][
                    "error_message"
                ]
            )

        output_dir = "./output"


        job.download_outputs(
            output_dir=output_dir
        )

        json_file = (
            f"{output_dir}/"
            f"{os.path.basename(audio_path)}.json"
        )

        try:
            with open(
                json_file,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)
            transcript = data["transcript"]
        finally:
            if os.path.exists(json_file):
                try:
                    os.remove(json_file)
                except Exception as e:
                    print(f"Failed to delete {json_file}: {e}")

        return transcript