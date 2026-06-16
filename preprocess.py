import librosa
import numpy as np


class AudioPreprocessor:

    TARGET_SR = 16000

    def load_audio(self, audio_path):

        audio, sr = librosa.load(
            audio_path,
            sr=self.TARGET_SR,
            mono=True
        )

        return audio, self.TARGET_SR

    def normalize_audio(self, audio):

        max_val = np.max(
            np.abs(audio)
        )

        if max_val > 0:
            audio = audio / max_val

        return audio

    def preprocess(self, audio_path):

        audio, sr = self.load_audio(
            audio_path
        )

        audio = self.normalize_audio(
            audio
        )

        return audio, sr