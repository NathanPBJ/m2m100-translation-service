"""Local M2M100 translation engine."""

import logging
from time import perf_counter
from typing import Any, Literal, cast

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DeviceConfigurationError,
    InputTooLongError,
    InvalidTranslationInputError,
    ModelLoadError,
    ModelNotLoadedError,
    TranslationInferenceError,
    TranslationServiceError,
    UnsupportedLanguageError,
)
from app.domain.translation import TranslationResult

logger = logging.getLogger(__name__)

DeviceName = Literal["cpu", "cuda"]


class M2M100TranslationService:
    """Load and run the local M2M100 model independently from FastAPI."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._device = self._resolve_device(self._settings.model_device)
        self._tokenizer: M2M100Tokenizer | None = None
        self._model: M2M100ForConditionalGeneration | None = None
        self._supported_languages: tuple[str, ...] = ()

    @property
    def is_loaded(self) -> bool:
        """Return whether both tokenizer and model are ready."""
        return self._tokenizer is not None and self._model is not None

    @property
    def device(self) -> str:
        """Return the selected PyTorch device name."""
        return self._device

    @property
    def model_name(self) -> str:
        """Return the configured Hugging Face model checkpoint."""
        return self._settings.model_name

    @staticmethod
    def _resolve_device(configured_device: str) -> DeviceName:
        if configured_device == "cpu":
            return "cpu"
        if configured_device == "cuda":
            if not torch.cuda.is_available():
                raise DeviceConfigurationError(
                    "CUDA was requested, but PyTorch reports that CUDA is unavailable."
                )
            return "cuda"
        if configured_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        raise DeviceConfigurationError(
            f"Unsupported model device '{configured_device}'. Use auto, cpu, or cuda."
        )

    def load_model(self) -> None:
        """Load the configured tokenizer and model once."""
        if self.is_loaded:
            return

        started_at = perf_counter()
        load_options = {
            "cache_dir": str(self._settings.model_cache_dir),
            "local_files_only": self._settings.model_local_files_only,
        }
        logger.info(
            "Loading model %s from cache directory %s (local-only=%s) on %s",
            self.model_name,
            self._settings.model_cache_dir,
            self._settings.model_local_files_only,
            self.device,
        )

        try:
            tokenizer = M2M100Tokenizer.from_pretrained(self.model_name, **load_options)
            model = M2M100ForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_options,
            )
            model.to(self.device)
            model.eval()
            supported_languages = tuple(sorted(set(tokenizer.lang_code_to_id)))
        except Exception as exc:
            logger.exception("Failed to load model %s", self.model_name)
            raise ModelLoadError(f"Could not load translation model '{self.model_name}'.") from exc

        self._tokenizer = tokenizer
        self._model = model
        self._supported_languages = supported_languages
        logger.info(
            "Model %s loaded successfully on %s in %.2f seconds",
            self.model_name,
            self.device,
            perf_counter() - started_at,
        )

    def get_supported_languages(self) -> tuple[str, ...]:
        """Return sorted language codes exposed by the loaded tokenizer."""
        self._ensure_loaded()
        return self._supported_languages

    def supports_language(self, language_code: str) -> bool:
        """Return whether a normalized code is directly supported by M2M100."""
        self._ensure_loaded()
        if not isinstance(language_code, str):
            return False
        normalized_code = language_code.strip().lower()
        return normalized_code != "auto" and normalized_code in self._supported_languages

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text directly from the source language to the target language."""
        self._ensure_loaded()
        self._validate_text(text)
        source_code = self._validate_language(source_language, "Source")
        target_code = self._validate_language(target_language, "Target")

        if source_code == target_code:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_code,
                target_language=target_code,
                model_name=self.model_name,
                device=self.device,
                status="unchanged",
            )

        tokenizer = cast(M2M100Tokenizer, self._tokenizer)
        model = cast(M2M100ForConditionalGeneration, self._model)
        tokenizer.src_lang = source_code

        try:
            encoded_inputs = tokenizer(text, return_tensors="pt", truncation=False)
            sequence_length = self._get_sequence_length(encoded_inputs)
            if sequence_length > self._settings.model_max_input_tokens:
                raise InputTooLongError(
                    "Input contains "
                    f"{sequence_length} tokens; the maximum is "
                    f"{self._settings.model_max_input_tokens}."
                )

            model_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}
            target_language_id = tokenizer.get_lang_id(target_code)
            with torch.inference_mode():
                generated_tokens = model.generate(
                    **model_inputs,
                    forced_bos_token_id=target_language_id,
                    max_new_tokens=self._settings.model_max_new_tokens,
                    num_beams=self._settings.model_num_beams,
                )
            translated_text = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0]
            if not translated_text.strip():
                raise TranslationInferenceError("The translation model returned an empty result.")
        except TranslationServiceError:
            raise
        except Exception as exc:
            logger.exception(
                "Translation inference failed for language pair %s to %s",
                source_code,
                target_code,
            )
            raise TranslationInferenceError(
                f"Translation inference failed for language pair {source_code} to {target_code}."
            ) from exc

        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_code,
            target_language=target_code,
            model_name=self.model_name,
            device=self.device,
            status="translated",
        )

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "The translation model is not loaded. Call load_model() first."
            )

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise InvalidTranslationInputError("Translation text must be a non-empty string.")

    def _validate_language(self, language_code: str, label: str) -> str:
        if not isinstance(language_code, str) or not language_code.strip():
            raise UnsupportedLanguageError(f"{label} language code is required.")

        normalized_code = language_code.strip().lower()
        if normalized_code == "auto":
            raise UnsupportedLanguageError(
                "Language code 'auto' is not supported because automatic detection "
                "is not implemented."
            )
        if normalized_code not in self._supported_languages:
            raise UnsupportedLanguageError(
                f"{label} language code '{normalized_code}' is not supported."
            )
        return normalized_code

    @staticmethod
    def _get_sequence_length(encoded_inputs: dict[str, Any]) -> int:
        input_ids = encoded_inputs.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise TranslationInferenceError("Tokenizer output does not contain valid input IDs.")
        return int(input_ids.shape[-1])
