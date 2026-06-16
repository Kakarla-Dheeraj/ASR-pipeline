SUPPORTED_SARVAM_LANGUAGES = {
    "hi",
    "bn",
    "ta",
    "te",
    "mr",
    "gu",
    "kn",
    "ml",
    "as",
    "ur",
    "sa",
    "ne",
    "pa",
    "or",
    "sd",
}

class Router:  

    def get_provider(
        self,
        language
    ):

        if language == "en":
            return "whisper"

        if language in SUPPORTED_SARVAM_LANGUAGES:
            return "saaras"

        return "unsupported"