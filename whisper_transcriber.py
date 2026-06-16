from groq import Groq
import os


class WhisperTranscriber:

    def __init__(self):

        self.client = Groq(
            api_key=os.getenv(
                "GROQ_API_KEY"
            )
        )

    def transcribe(
        self,
        audio_path
    ):

        with open(
            audio_path,
            "rb"
        ) as file:

            response = (
                self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3"
                )
            )

        return response.text