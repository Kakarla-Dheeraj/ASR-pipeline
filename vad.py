from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)

import torch


class VADProcessor:

    def __init__(self):

        self.model = load_silero_vad()

    def get_speech_segments(
        self,
        audio,
        sampling_rate=16000
    ):

        speech_timestamps = (
            get_speech_timestamps(
                torch.tensor(audio),
                self.model,
                sampling_rate=sampling_rate
            )
        )

        return speech_timestamps