"""Immutable domain types for token-aware text chunking."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One non-empty source span and the exact separator that follows it."""

    index: int
    text: str
    token_count: int
    separator_after: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index cannot be negative.")
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text must contain non-whitespace content.")
        if self.token_count <= 0:
            raise ValueError("Chunk token count must be positive.")


@dataclass(frozen=True, slots=True)
class TextChunkingResult:
    """Validated chunks that can reconstruct the original source exactly."""

    chunks: tuple[TextChunk, ...]
    chunked: bool
    chunk_count: int
    maximum_chunk_tokens: int
    prefix_separator: str = ""

    def __post_init__(self) -> None:
        if self.chunk_count != len(self.chunks):
            raise ValueError("Chunk count must equal the number of chunks.")
        if self.chunked != (self.chunk_count > 1):
            raise ValueError("Chunked state must match the number of chunks.")
        if self.chunk_count <= 0:
            raise ValueError("Chunking result must contain at least one chunk.")
        if tuple(chunk.index for chunk in self.chunks) != tuple(range(self.chunk_count)):
            raise ValueError("Chunk indexes must be consecutive and start at zero.")
        if self.maximum_chunk_tokens != max(chunk.token_count for chunk in self.chunks):
            raise ValueError("Maximum chunk tokens must match the chunk data.")

    def reconstruct_source(self) -> str:
        """Rebuild source text including all preserved separators."""
        return self.prefix_separator + "".join(
            chunk.text + chunk.separator_after for chunk in self.chunks
        )
