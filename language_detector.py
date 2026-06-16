from speechbrain.inference.classifiers import EncoderClassifier
import torch


class LanguageDetector:

    def __init__(self):

        self.classifier = (
            EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa"
            )
        )

    def detect(
        self,
        speech_audio
    ):

        waveform = torch.tensor(
            speech_audio,
            dtype=torch.float32
        ).unsqueeze(0)

        out_prob, score, index, label = (
            self.classifier.classify_batch(
                waveform
            )
        )

        confidence = (
            torch.exp(
                score[0]
            ).item()
        )

        language = (
            label[0]
            .split(":")[0]
        )

        return {
            "language": language,
            "confidence": confidence
        }