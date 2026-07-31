"""Local M2M100 translation engine."""

import logging
from threading import RLock
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
    TextChunkingError,
    TranslationInferenceError,
    TranslationOutputTruncatedError,
    TranslationServiceError,
    UnsupportedLanguageError,
)
from app.domain.text_chunking import TextChunk, TextChunkingResult
from app.domain.translation import TranslationResult
from app.services.text_chunking import TokenAwareTextChunkingService

logger = logging.getLogger(__name__)

DeviceName = Literal["cpu", "cuda"]


class M2M100TranslationService:
    """Load and run the local M2M100 model independently from FastAPI."""

    def __init__(
        self,
        settings: Settings | None = None,
        chunking_service: TokenAwareTextChunkingService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._device = self._resolve_device(self._settings.model_device)
        self._tokenizer: M2M100Tokenizer | None = None
        self._model: M2M100ForConditionalGeneration | None = None
        self._supported_languages: tuple[str, ...] = ()
        self._chunking_service = chunking_service or TokenAwareTextChunkingService()
        self._state_lock = RLock()

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
        with self._state_lock:
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
                raise ModelLoadError(
                    f"Could not load translation model '{self.model_name}'."
                ) from exc

            self._tokenizer = tokenizer
            self._model = model
            self._supported_languages = supported_languages
            logger.info(
                "Model %s loaded successfully on %s in %.2f seconds",
                self.model_name,
                self.device,
                perf_counter() - started_at,
            )

    def unload_model(self) -> None:
        """Release in-memory model resources without deleting the disk cache."""
        with self._state_lock:
            logger.info("Unloading model %s from %s", self.model_name, self.device)
            was_loaded = self.is_loaded
            self._model = None
            self._tokenizer = None
            self._supported_languages = ()
            if was_loaded and self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model %s unloaded successfully", self.model_name)

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
        """Translate one strict model input without automatically chunking it."""
        with self._state_lock:
            self._ensure_loaded()
            self._validate_text(text)
            source_code = self._validate_language(source_language, "Source")
            target_code = self._validate_language(target_language, "Target")

            if source_code == target_code:
                return self._unchanged_result(text, source_code, target_code)

            translated_text, _ = self._translate_single_chunk_unlocked(
                text=text,
                source_code=source_code,
                target_code=target_code,
                chunk_index=0,
                total_chunks=1,
            )
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_code,
                target_language=target_code,
                model_name=self.model_name,
                device=self.device,
                status="translated",
                chunked=False,
                chunk_count=1,
                chunk_token_limit=self._settings.long_text_chunk_max_tokens,
            )

    def translate_long_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate one string as ordered token-safe chunks when required."""
        with self._state_lock:
            self._ensure_loaded()
            self._validate_text(text)
            source_code = self._validate_language(source_language, "Source")
            target_code = self._validate_language(target_language, "Target")

            if source_code == target_code:
                return self._unchanged_result(text, source_code, target_code)

            tokenizer = cast(M2M100Tokenizer, self._tokenizer)
            tokenizer.src_lang = source_code
            token_count = self._count_tokens_unlocked(text)
            if self._settings.long_text_chunking_enabled:
                chunking_result = self._chunking_service.chunk_text(
                    text=text,
                    token_counter=self._count_tokens_unlocked,
                    maximum_tokens=self._settings.long_text_chunk_max_tokens,
                    maximum_chunks=self._settings.long_text_max_chunks,
                )
            else:
                if token_count > self._settings.model_max_input_tokens:
                    raise InputTooLongError(
                        actual_tokens=token_count,
                        maximum_tokens=self._settings.model_max_input_tokens,
                    )
                chunking_result = TextChunkingResult(
                    chunks=(
                        self._make_single_chunk(
                            text,
                            token_count,
                        ),
                    ),
                    chunked=False,
                    chunk_count=1,
                    maximum_chunk_tokens=token_count,
                )

            started_at = perf_counter()
            translated_chunks: list[str] = []
            for chunk in chunking_result.chunks:
                translated_chunk, _ = self._translate_single_chunk_unlocked(
                    text=chunk.text,
                    source_code=source_code,
                    target_code=target_code,
                    chunk_index=chunk.index,
                    total_chunks=chunking_result.chunk_count,
                )
                translated_chunks.append(translated_chunk)

            translated_text = chunking_result.prefix_separator + "".join(
                translated_chunk + chunk.separator_after
                for translated_chunk, chunk in zip(
                    translated_chunks,
                    chunking_result.chunks,
                    strict=True,
                )
            )
            logger.info(
                "Long-text translation completed for %s to %s: %d characters, "
                "%d chunks, maximum %d source tokens, %.2f seconds",
                source_code,
                target_code,
                len(text),
                chunking_result.chunk_count,
                chunking_result.maximum_chunk_tokens,
                perf_counter() - started_at,
            )
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_code,
                target_language=target_code,
                model_name=self.model_name,
                device=self.device,
                status="translated",
                chunked=chunking_result.chunked,
                chunk_count=chunking_result.chunk_count,
                chunk_token_limit=self._settings.long_text_chunk_max_tokens,
            )

    def count_tokens(self, text: str, source_language: str) -> int:
        """Count source tokens, including tokenizer special tokens."""
        with self._state_lock:
            self._ensure_loaded()
            self._validate_text(text)
            source_code = self._validate_language(source_language, "Source")
            tokenizer = cast(M2M100Tokenizer, self._tokenizer)
            tokenizer.src_lang = source_code
            return self._count_tokens_unlocked(text)

    def prepare_text_chunks(
        self,
        text: str,
        source_language: str,
    ) -> TextChunkingResult:
        """Prepare and validate chunks without running model generation."""
        with self._state_lock:
            self._ensure_loaded()
            self._validate_text(text)
            source_code = self._validate_language(source_language, "Source")
            tokenizer = cast(M2M100Tokenizer, self._tokenizer)
            tokenizer.src_lang = source_code
            return self._chunking_service.chunk_text(
                text=text,
                token_counter=self._count_tokens_unlocked,
                maximum_tokens=self._settings.long_text_chunk_max_tokens,
                maximum_chunks=self._settings.long_text_max_chunks,
            )

    def _translate_single_chunk_unlocked(
        self,
        *,
        text: str,
        source_code: str,
        target_code: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[str, int]:
        tokenizer = cast(M2M100Tokenizer, self._tokenizer)
        model = cast(M2M100ForConditionalGeneration, self._model)
        tokenizer.src_lang = source_code

        try:
            encoded_inputs = tokenizer(text, return_tensors="pt", truncation=False)
            sequence_length = self._get_sequence_length(encoded_inputs)
            if sequence_length > self._settings.model_max_input_tokens:
                raise InputTooLongError(
                    actual_tokens=sequence_length,
                    maximum_tokens=self._settings.model_max_input_tokens,
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
            self._ensure_output_complete(
                generated_tokens,
                tokenizer,
                model,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            translated_text = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0]
            if not translated_text.strip():
                raise TranslationInferenceError("The translation model returned an empty result.")
            return translated_text, sequence_length
        except TranslationServiceError:
            raise
        except Exception as exc:
            logger.exception(
                "Translation inference failed for %s to %s at chunk %d of %d",
                source_code,
                target_code,
                chunk_index + 1,
                total_chunks,
            )
            raise TranslationInferenceError(
                f"Translation inference failed for language pair {source_code} to {target_code}."
            ) from exc

    def _count_tokens_unlocked(self, text: str) -> int:
        tokenizer = cast(M2M100Tokenizer, self._tokenizer)
        try:
            encoded_inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                verbose=False,
            )
            return self._get_sequence_length(encoded_inputs)
        except TextChunkingError:
            raise
        except Exception as exc:
            raise TranslationInferenceError("Tokenizer could not count source tokens.") from exc

    @staticmethod
    def _make_single_chunk(text: str, token_count: int) -> TextChunk:
        return TextChunk(index=0, text=text, token_count=token_count)

    def _unchanged_result(
        self,
        text: str,
        source_code: str,
        target_code: str,
    ) -> TranslationResult:
        return TranslationResult(
            original_text=text,
            translated_text=text,
            source_language=source_code,
            target_language=target_code,
            model_name=self.model_name,
            device=self.device,
            status="unchanged",
            chunked=False,
            chunk_count=0,
            chunk_token_limit=self._settings.long_text_chunk_max_tokens,
        )

    def _ensure_output_complete(
        self,
        generated_tokens: Any,
        tokenizer: M2M100Tokenizer,
        model: M2M100ForConditionalGeneration,
        *,
        chunk_index: int,
        total_chunks: int,
    ) -> None:
        try:
            sequence = generated_tokens[0]
            generated_length = (
                int(sequence.shape[-1]) if hasattr(sequence, "shape") else len(sequence)
            )
            final_token = sequence[-1]
            if hasattr(final_token, "item"):
                final_token = final_token.item()
            final_token_id = int(final_token)
        except Exception as exc:
            raise TranslationInferenceError(
                "The translation model returned invalid generated tokens."
            ) from exc

        eos_token_ids = {
            int(token_id)
            for token_id in (
                getattr(tokenizer, "eos_token_id", None),
                getattr(getattr(model, "config", None), "eos_token_id", None),
            )
            if token_id is not None
        }
        if (
            generated_length >= self._settings.model_max_new_tokens
            and eos_token_ids
            and final_token_id not in eos_token_ids
        ):
            logger.error(
                "Translation output reached generation limit without EOS "
                "at chunk %d of %d (generated tokens=%d)",
                chunk_index + 1,
                total_chunks,
                generated_length,
            )
            raise TranslationOutputTruncatedError(
                "Translation output reached its generation limit without an EOS token."
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
