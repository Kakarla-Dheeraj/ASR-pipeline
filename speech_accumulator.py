import numpy as np


class SpeechAccumulator:

    def collect_speech(
        self,
        audio,
        speech_segments,
        target_duration_sec,
        sampling_rate=16000
    ):

        target_samples = (
            target_duration_sec *
            sampling_rate
        )

        collected = []

        total_samples = 0

        for segment in speech_segments:

            start = segment["start"]
            end = segment["end"]

            speech_chunk = audio[start:end]

            remaining = (
                target_samples -
                total_samples
            )

            if len(
                speech_chunk
            ) >= remaining:

                collected.append(
                    speech_chunk[:remaining]
                )

                break

            collected.append(
                speech_chunk
            )

            total_samples += len(
                speech_chunk
            )

        if not collected:
            return np.array([])

        return np.concatenate(
            collected
        )
    def skip_initial_speech(
        self,
        speech_segments,
        skip_seconds,
        sampling_rate=16000
    ):

        skip_samples = (
            skip_seconds *
            sampling_rate
        )

        new_segments = []

        for segment in speech_segments:

            segment_length = (
                segment["end"] -
                segment["start"]
            )

            if skip_samples >= segment_length:

                skip_samples -= segment_length
                continue

            if skip_samples > 0:

                segment = {
                    "start":
                        segment["start"] +
                        skip_samples,
                    "end":
                        segment["end"]
                }

                skip_samples = 0

            new_segments.append(
                segment
            )

        return new_segments
    def total_speech_duration(
        self,
        speech_segments,
        sampling_rate=16000
    ):

        total_samples = 0

        for segment in speech_segments:

            total_samples += (
                segment["end"] -
                segment["start"]
            )

        return (
            total_samples /
            sampling_rate
        )