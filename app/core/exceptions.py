"""Custom exceptions raised by the translation engine."""


class TranslationServiceError(Exception):
    """Base exception for the translation service."""


class ModelLoadError(TranslationServiceError):
    """Raised when the tokenizer or model cannot be loaded."""


class ModelNotLoadedError(TranslationServiceError):
    """Raised when an operation needs a model that has not been loaded."""


class UnsupportedLanguageError(TranslationServiceError):
    """Raised when M2M100 does not support a requested language code."""


class InvalidTranslationInputError(TranslationServiceError):
    """Raised when translation input is not a non-empty string."""


class InputTooLongError(TranslationServiceError):
    """Raised when tokenized input exceeds the configured limit."""

    def __init__(self, actual_tokens: int, maximum_tokens: int) -> None:
        self.actual_tokens = actual_tokens
        self.maximum_tokens = maximum_tokens
        super().__init__(f"Input contains {actual_tokens} tokens; the maximum is {maximum_tokens}.")


class InputTextTooLargeError(TranslationServiceError):
    """Raised when API input exceeds the configured character limit."""

    def __init__(self, actual_characters: int, maximum_characters: int) -> None:
        self.actual_characters = actual_characters
        self.maximum_characters = maximum_characters
        super().__init__(
            f"Input contains {actual_characters} characters; the maximum is {maximum_characters}."
        )


class TranslationInferenceError(TranslationServiceError):
    """Raised when model inference or decoding fails."""


class DeviceConfigurationError(TranslationServiceError):
    """Raised when the requested compute device cannot be used."""
