"""
LiveSubtitler — converts streaming partial/final transcription results
into FAB-ready subtitle segments.

Tracks word-level stability across consecutive partials and commits
stable prefixes in configurable-width chunks. On a final, flushes
everything that is still pending.

Ported from AWS Transcribe segmenter, adapted for voxtral-server.
"""

from __future__ import annotations

import time


class LiveSubtitler:
    """
    Stateful segmenter for a single transcription stream.

    The transcription engine emits a growing partial transcript until it
    fires a final. This class tracks word-level stability across consecutive
    partials and commits stable prefixes in max_chars chunks. On a final it
    flushes everything that is still pending.

    State resets automatically after every on_final() call.
    """

    def __init__(
        self,
        max_chars: int = 42,
        min_chars: int = 15,
        stability_secs: float = 0.5,
    ):
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._stability_secs = stability_secs

        # Number of words from the current window already committed
        self._committed_words: int = 0
        # word-index -> (word, first_seen_timestamp)
        self._stability: dict[int, tuple[str, float]] = {}

    def on_partial(self, transcript: str, ts: float) -> list[str]:
        """Return any newly committable segments (may be empty)."""
        words = transcript.split()
        self._update_stability(words, ts)
        stable_end = self._stable_boundary(words, ts)
        pending = " ".join(words[self._committed_words:stable_end])
        return self._commit(pending, flush=False)

    def on_final(self, transcript: str, ts: float) -> list[str]:
        """Flush all remaining text and reset state."""
        words = transcript.split()
        remaining = " ".join(words[self._committed_words:])
        segments = self._commit(remaining, flush=True)
        self._committed_words = 0
        self._stability = {}
        return segments

    def _update_stability(self, words: list[str], ts: float) -> None:
        # Remove entries for positions no longer in the transcript
        for i in list(self._stability):
            if i >= len(words):
                del self._stability[i]
        # Update uncommitted positions
        for i in range(self._committed_words, len(words)):
            word = words[i]
            if i in self._stability:
                prev, first_ts = self._stability[i]
                if prev != word:
                    self._stability[i] = (word, ts)  # word changed — reset clock
            else:
                self._stability[i] = (word, ts)

    def _stable_boundary(self, words: list[str], ts: float) -> int:
        """Rightmost index (exclusive) where all words since _committed_words are stable."""
        end = self._committed_words
        for i in range(self._committed_words, len(words)):
            entry = self._stability.get(i)
            if not entry:
                break
            word, first_ts = entry
            if words[i] != word or (ts - first_ts) < self._stability_secs:
                break
            end = i + 1
        return end

    def _commit(self, text: str, flush: bool) -> list[str]:
        """
        Slice text into max_chars chunks and advance _committed_words.
        When flush=False: skips the tail if it is shorter than min_chars.
        When flush=True:  commits everything, even a short last chunk.
        """
        segments: list[str] = []
        remaining = text.strip()

        while remaining:
            if not flush and len(remaining) < self._min_chars:
                break
            chunk = self._cut(remaining)
            segments.append(chunk)
            self._committed_words += len(chunk.split())
            remaining = remaining[len(chunk):].strip()

        return segments

    def _cut(self, text: str) -> str:
        """Return up to max_chars, breaking at a word boundary if possible."""
        if len(text) <= self._max_chars:
            return text
        chunk = text[:self._max_chars]
        space = chunk.rfind(" ")
        return chunk[:space] if space > 0 else chunk
