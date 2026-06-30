"""
Streaming sentence splitter for TTS latency optimisation.

Why: the LLM streams tokens one-by-one. Waiting for the full response before
starting TTS adds 500ms–2s of unnecessary delay. Instead, we detect sentence
boundaries (. ? !) as tokens arrive and dispatch each complete sentence to TTS
immediately — so the first audio chunk can begin while the LLM is still generating.

Boundary rule: sentence ends at . ? ! when followed by whitespace.
Known limitation: abbreviations like "Dr." or "U.S." may cause spurious splits.
Mitigation: the LLM system prompt (T09) tells the agent to avoid abbreviations
in spoken responses.
"""

import re


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+")


class SentenceSplitter:
    """
    Feed LLM tokens via push(); complete sentences are returned for TTS.
    Call flush() after the stream ends to get any trailing text.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def push(self, token: str) -> list[str]:
        """
        Append token to internal buffer.
        Returns zero or more complete sentences ready for TTS.
        Each returned sentence includes its terminal punctuation.
        """
        self._buffer += token
        parts = _SENTENCE_BOUNDARY.split(self._buffer)
        if len(parts) == 1:
            return []
        sentences = [p.strip() for p in parts[:-1] if p.strip()]
        self._buffer = parts[-1]
        return sentences

    def flush(self) -> str:
        """
        Return remaining buffered text (no terminal punctuation found).
        Resets internal state. Call at end of LLM stream.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining
