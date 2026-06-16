class ConfidenceManager:

    def __init__(
        self,
        high_confidence_threshold=0.80
    ):

        self.high_confidence_threshold = (
            high_confidence_threshold
        )

    def should_retry(
        self,
        confidence
    ):

        return (
            confidence <
            self.high_confidence_threshold
        )