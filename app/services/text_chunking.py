"""Token-aware long-text chunking without framework or tokenizer dependencies."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.core.exceptions import (
    InvalidTranslationInputError,
    TextChunkingError,
    TextChunkingFailedError,
    TooManyChunksError,
)
from app.domain.text_chunking import TextChunk, TextChunkingResult

TokenCounter = Callable[[str], int]

_PARAGRAPH_SEPARATOR = re.compile(
    r"[^\S\r\n]*(?:\r\n|\r|\n)(?:[^\S\r\n]*(?:\r\n|\r|\n))*[^\S\r\n]*"
)
_TRAILING_WHITESPACE = re.compile(r"\s+$", flags=re.UNICODE)
_BOUNDARY_PUNCTUATION = frozenset(".!?…。！？;:：；")
_FALLBACK_PUNCTUATION = frozenset(",，、/\\|-–—()[]{}")
_COMMON_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    }
)


@dataclass(frozen=True, slots=True)
class _TextUnit:
    text: str
    separator_after: str = ""


class TokenAwareTextChunkingService:
    """Split text by paragraphs, sentences, then safe Unicode character spans."""

    def chunk_text(
        self,
        text: str,
        token_counter: TokenCounter,
        maximum_tokens: int,
        maximum_chunks: int,
    ) -> TextChunkingResult:
        """Return ordered chunks whose source reconstruction is identical."""
        if not isinstance(text, str) or not text.strip():
            raise InvalidTranslationInputError("Translation text must be a non-empty string.")
        if not callable(token_counter):
            raise TextChunkingFailedError("A valid token counter is required.")
        if (
            not isinstance(maximum_tokens, int)
            or isinstance(maximum_tokens, bool)
            or maximum_tokens <= 0
        ):
            raise TextChunkingFailedError("Maximum chunk tokens must be a positive integer.")
        if (
            not isinstance(maximum_chunks, int)
            or isinstance(maximum_chunks, bool)
            or maximum_chunks <= 0
        ):
            raise TextChunkingFailedError("Maximum chunks must be a positive integer.")

        whole_token_count = self._safe_token_count(text, token_counter)
        if whole_token_count <= maximum_tokens:
            return TextChunkingResult(
                chunks=(TextChunk(index=0, text=text, token_count=whole_token_count),),
                chunked=False,
                chunk_count=1,
                maximum_chunk_tokens=whole_token_count,
            )

        prefix_separator, body = self._extract_leading_whitespace(text)
        paragraph_units = self._split_paragraphs(body)
        raw_chunks: list[_TextUnit] = []
        for paragraph in paragraph_units:
            paragraph_chunks = self._chunk_paragraph(
                paragraph.text,
                token_counter,
                maximum_tokens,
            )
            if not paragraph_chunks:
                raise TextChunkingFailedError("A source paragraph did not produce a valid chunk.")
            final_chunk = paragraph_chunks[-1]
            paragraph_chunks[-1] = _TextUnit(
                text=final_chunk.text,
                separator_after=final_chunk.separator_after + paragraph.separator_after,
            )
            raw_chunks.extend(paragraph_chunks)
            if len(raw_chunks) > maximum_chunks:
                raise TooManyChunksError(len(raw_chunks), maximum_chunks)

        chunks = tuple(
            TextChunk(
                index=index,
                text=unit.text,
                token_count=self._safe_token_count(unit.text, token_counter),
                separator_after=unit.separator_after,
            )
            for index, unit in enumerate(raw_chunks)
        )
        self._validate_result(
            original_text=text,
            prefix_separator=prefix_separator,
            chunks=chunks,
            maximum_tokens=maximum_tokens,
            maximum_chunks=maximum_chunks,
        )
        maximum_chunk_tokens = max(chunk.token_count for chunk in chunks)
        return TextChunkingResult(
            chunks=chunks,
            chunked=len(chunks) > 1,
            chunk_count=len(chunks),
            maximum_chunk_tokens=maximum_chunk_tokens,
            prefix_separator=prefix_separator,
        )

    @staticmethod
    def _extract_leading_whitespace(text: str) -> tuple[str, str]:
        body_start = 0
        while body_start < len(text) and text[body_start].isspace():
            body_start += 1
        return text[:body_start], text[body_start:]

    def _split_paragraphs(self, text: str) -> list[_TextUnit]:
        paragraphs: list[_TextUnit] = []
        cursor = 0
        for match in _PARAGRAPH_SEPARATOR.finditer(text):
            content = text[cursor : match.start()]
            separator = match.group(0)
            if content:
                paragraphs.append(_TextUnit(content, separator))
            elif paragraphs:
                previous = paragraphs[-1]
                paragraphs[-1] = _TextUnit(
                    previous.text,
                    previous.separator_after + separator,
                )
            cursor = match.end()

        remainder = text[cursor:]
        if remainder:
            content, trailing = self._extract_trailing_whitespace(remainder)
            if content:
                paragraphs.append(_TextUnit(content, trailing))
            elif paragraphs:
                previous = paragraphs[-1]
                paragraphs[-1] = _TextUnit(
                    previous.text,
                    previous.separator_after + remainder,
                )
        if not paragraphs:
            raise TextChunkingFailedError("Text does not contain a translatable source span.")
        return paragraphs

    @staticmethod
    def _extract_trailing_whitespace(text: str) -> tuple[str, str]:
        match = _TRAILING_WHITESPACE.search(text)
        if match is None:
            return text, ""
        return text[: match.start()], text[match.start() :]

    def _chunk_paragraph(
        self,
        paragraph: str,
        token_counter: TokenCounter,
        maximum_tokens: int,
    ) -> list[_TextUnit]:
        if self._safe_token_count(paragraph, token_counter) <= maximum_tokens:
            return [_TextUnit(paragraph)]

        sentences = self._split_sentences(paragraph)
        expanded_units: list[_TextUnit] = []
        for sentence in sentences:
            if self._safe_token_count(sentence.text, token_counter) <= maximum_tokens:
                expanded_units.append(sentence)
                continue
            fallback_parts = self._split_oversized_text(
                sentence.text,
                token_counter,
                maximum_tokens,
            )
            fallback_parts[-1] = _TextUnit(
                fallback_parts[-1].text,
                fallback_parts[-1].separator_after + sentence.separator_after,
            )
            expanded_units.extend(fallback_parts)
        return self._greedy_pack(expanded_units, token_counter, maximum_tokens)

    def _split_sentences(self, text: str) -> list[_TextUnit]:
        units: list[_TextUnit] = []
        start = 0
        index = 0
        while index < len(text):
            character = text[index]
            if character not in _BOUNDARY_PUNCTUATION or not self._is_sentence_boundary(
                text, index
            ):
                index += 1
                continue

            punctuation_end = index + 1
            while punctuation_end < len(text) and text[punctuation_end] in _BOUNDARY_PUNCTUATION:
                punctuation_end += 1
            separator_end = punctuation_end
            while separator_end < len(text) and text[separator_end].isspace():
                separator_end += 1
            sentence = text[start:punctuation_end]
            separator = text[punctuation_end:separator_end]
            if sentence:
                units.append(_TextUnit(sentence, separator))
            start = separator_end
            index = separator_end

        if start < len(text):
            content, trailing = self._extract_trailing_whitespace(text[start:])
            if content:
                units.append(_TextUnit(content, trailing))
            elif units:
                previous = units[-1]
                units[-1] = _TextUnit(
                    previous.text,
                    previous.separator_after + text[start:],
                )
        return units or [_TextUnit(text)]

    def _is_sentence_boundary(self, text: str, index: int) -> bool:
        character = text[index]
        if character != ".":
            return True
        previous_character = text[index - 1] if index > 0 else ""
        next_character = text[index + 1] if index + 1 < len(text) else ""
        if previous_character.isdigit() and next_character.isdigit():
            return False
        if next_character.isalpha() and index + 2 < len(text) and text[index + 2] == ".":
            return False

        token_start = index
        while token_start > 0 and not text[token_start - 1].isspace():
            token_start -= 1
        token_end = index + 1
        while token_end < len(text) and not text[token_end].isspace():
            token_end += 1
        token = text[token_start:token_end].lower()
        if "@" in token or token.startswith(("http://", "https://", "www.")):
            return False
        prefix = text[token_start : index + 1].lower()
        if prefix in _COMMON_ABBREVIATIONS:
            return False
        context = text[max(0, index - 5) : index + 1].lower()
        return not any(context.endswith(abbreviation) for abbreviation in _COMMON_ABBREVIATIONS)

    def _split_oversized_text(
        self,
        text: str,
        token_counter: TokenCounter,
        maximum_tokens: int,
    ) -> list[_TextUnit]:
        parts: list[_TextUnit] = []
        remaining = text
        while remaining:
            if self._safe_token_count(remaining, token_counter) <= maximum_tokens:
                parts.append(_TextUnit(remaining))
                break

            maximum_end = self._largest_fitting_prefix(
                remaining,
                token_counter,
                maximum_tokens,
            )
            preferred_end = self._preferred_boundary(remaining, maximum_end)
            split_at = preferred_end if preferred_end > 0 else maximum_end
            if split_at <= 0:
                raise TextChunkingFailedError(
                    "A Unicode source character exceeds the configured token budget."
                )
            part = remaining[:split_at]
            if not part or not part.strip():
                split_at = maximum_end
                part = remaining[:split_at]
            if not part or not part.strip():
                raise TextChunkingFailedError(
                    "Oversized source text could not be split with forward progress."
                )
            parts.append(_TextUnit(part))
            remaining = remaining[split_at:]
        return parts

    def _largest_fitting_prefix(
        self,
        text: str,
        token_counter: TokenCounter,
        maximum_tokens: int,
    ) -> int:
        low = 1
        high = len(text)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            token_count = self._safe_token_count(text[:middle], token_counter)
            if token_count <= maximum_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _preferred_boundary(text: str, maximum_end: int) -> int:
        minimum_preferred_end = max(1, maximum_end // 2)
        for index in range(maximum_end, minimum_preferred_end - 1, -1):
            character = text[index - 1]
            if character.isspace() or character in _FALLBACK_PUNCTUATION:
                return index
        return maximum_end

    def _greedy_pack(
        self,
        units: list[_TextUnit],
        token_counter: TokenCounter,
        maximum_tokens: int,
    ) -> list[_TextUnit]:
        packed: list[_TextUnit] = []
        current: _TextUnit | None = None
        for unit in units:
            if not unit.text or not unit.text.strip():
                raise TextChunkingFailedError("Chunking produced an empty source span.")
            if current is None:
                current = unit
                continue
            candidate_text = current.text + current.separator_after + unit.text
            if self._safe_token_count(candidate_text, token_counter) <= maximum_tokens:
                current = _TextUnit(candidate_text, unit.separator_after)
            else:
                packed.append(current)
                current = unit
        if current is not None:
            packed.append(current)
        return packed

    @staticmethod
    def _safe_token_count(text: str, token_counter: TokenCounter) -> int:
        try:
            token_count = token_counter(text)
        except TextChunkingError:
            raise
        except Exception as exc:
            raise TextChunkingFailedError("Tokenizer could not count source tokens.") from exc
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            raise TextChunkingFailedError("Tokenizer returned an invalid token count.")
        return token_count

    @staticmethod
    def _validate_result(
        *,
        original_text: str,
        prefix_separator: str,
        chunks: tuple[TextChunk, ...],
        maximum_tokens: int,
        maximum_chunks: int,
    ) -> None:
        if not chunks:
            raise TextChunkingFailedError("Chunking produced no source chunks.")
        if len(chunks) > maximum_chunks:
            raise TooManyChunksError(len(chunks), maximum_chunks)
        if any(
            chunk.index != index
            or not chunk.text
            or not chunk.text.strip()
            or chunk.token_count > maximum_tokens
            for index, chunk in enumerate(chunks)
        ):
            raise TextChunkingFailedError("Chunking produced an invalid source chunk.")
        reconstructed = prefix_separator + "".join(
            chunk.text + chunk.separator_after for chunk in chunks
        )
        if reconstructed != original_text:
            raise TextChunkingFailedError("Source reconstruction validation failed.")
