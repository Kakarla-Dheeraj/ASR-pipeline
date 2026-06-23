import soundfile as sf


class AudioSegmenter:

    def merge_segments(self, segments, max_gap=2.0):
        if not segments:
            return []

        merged = []
        current = dict(segments[0])

        for next_seg in segments[1:]:
            if (
                next_seg["speaker"] == current["speaker"]
                and (next_seg["start"] - current["end"]) <= max_gap
            ):
                current["end"] = next_seg["end"]
            else:
                merged.append(current)
                current = dict(next_seg)
        merged.append(current)
        return merged

    def split(
        self,
        audio,
        sr,
        segments,
        min_duration=0.5,
        max_gap=2.0
    ):
        merged_segments = self.merge_segments(segments, max_gap=max_gap)
        paths = []

        for i, segment in enumerate(
            merged_segments
        ):
            start = int(
                segment["start"] * sr
            )
            end = int(
                segment["end"] * sr
            )

            if (
                segment["end"] -
                segment["start"]
            ) < min_duration:
                continue

            chunk = audio[start:end]
            path = f"temp_{i}.wav"

            sf.write(
                path,
                chunk,
                sr
            )

            paths.append(
                {
                    "speaker": segment["speaker"],
                    "path": path,
                    "start": segment["start"],
                    "end": segment["end"]
                }
            )

        return paths