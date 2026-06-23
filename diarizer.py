from pyannote.audio import Pipeline


class Diarizer:

    def __init__(
        self,
        hf_token
    ):

        self.pipeline = (
            Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        )

    def diarize(
        self,
        audio_file
    ):

        result = (
            self.pipeline(
                audio_file
            )
        )

        diarization = (
            result.speaker_diarization
        )

        segments = []

        for (
            turn,
            _,
            speaker
        ) in diarization.itertracks(
            yield_label=True
        ):

            segments.append(
                {
                    "speaker":
                        speaker,
                    "start":
                        turn.start,
                    "end":
                        turn.end
                }
            )

        return segments