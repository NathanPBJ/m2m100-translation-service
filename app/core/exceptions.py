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


class TranslationInferenceError(TranslationServiceError):
    """Raised when model inference or decoding fails."""


class DeviceConfigurationError(TranslationServiceError):
    """Raised when the requested compute device cannot be used."""
