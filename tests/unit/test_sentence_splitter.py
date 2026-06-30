"""Tests for core/pipeline/sentence_splitter.py"""

from core.pipeline.sentence_splitter import SentenceSplitter


# ---------------------------------------------------------------------------
# push() — token-by-token feeding
# ---------------------------------------------------------------------------

def test_no_sentence_end_returns_empty():
    s = SentenceSplitter()
    assert s.push("Hello") == []
    assert s.push(" there") == []


def test_sentence_ends_at_period_with_space():
    s = SentenceSplitter()
    assert s.push("Hello. ") == ["Hello."]


def test_sentence_ends_at_question_mark():
    s = SentenceSplitter()
    s.push("How are you")
    result = s.push("? Next")
    assert result == ["How are you?"]


def test_sentence_ends_at_exclamation():
    s = SentenceSplitter()
    s.push("Great")
    result = s.push("! ")
    assert result == ["Great!"]


def test_split_happens_only_with_trailing_space():
    # Period at very end of buffer — no trailing space — not yet a sentence end
    s = SentenceSplitter()
    result = s.push("Hello.")
    assert result == []


def test_two_sentences_in_one_push():
    s = SentenceSplitter()
    result = s.push("Hello. World. ")
    assert result == ["Hello.", "World."]


def test_incomplete_sentence_stays_buffered():
    s = SentenceSplitter()
    s.push("Hello. ")   # flushes "Hello."
    s.push("How are")   # buffered
    assert s.flush() == "How are"


def test_token_by_token_sentence_detected():
    s = SentenceSplitter()
    tokens = ["Thank", " you", " for", " calling", "."]
    sentences = []
    for t in tokens:
        sentences.extend(s.push(t))
    # No space after "." yet — not flushed
    assert sentences == []
    # Space arrives → sentence complete
    sentences.extend(s.push(" Now"))
    assert sentences == ["Thank you for calling."]


def test_multiple_pushes_accumulate_then_flush():
    s = SentenceSplitter()
    for token in ["This", " is", " a", " test"]:
        s.push(token)
    assert s.flush() == "This is a test"


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------

def test_flush_returns_remaining_buffer():
    s = SentenceSplitter()
    s.push("Partial sentence")
    assert s.flush() == "Partial sentence"


def test_flush_clears_buffer():
    s = SentenceSplitter()
    s.push("Hello")
    s.flush()
    assert s.flush() == ""


def test_flush_strips_whitespace():
    s = SentenceSplitter()
    s.push("  Hello  ")
    assert s.flush() == "Hello"


def test_flush_after_complete_sentence():
    s = SentenceSplitter()
    s.push("Done. ")  # "Done." flushed on push
    s.push("More")
    assert s.flush() == "More"


# ---------------------------------------------------------------------------
# Real-world LLM token stream simulations
# ---------------------------------------------------------------------------

def test_sales_response_two_sentences():
    s = SentenceSplitter()
    tokens = [
        "Thank", " you", " for", " letting", " me", " know", ".",
        " Can", " I", " ask", " what", " solution", " you",
        " currently", " use", "?",
    ]
    sentences = []
    for t in tokens:
        sentences.extend(s.push(t))
    remainder = s.flush()
    assert "Thank you for letting me know." in sentences
    assert remainder == "Can I ask what solution you currently use?"


def test_single_sentence_no_trailing_space():
    # Common at end of LLM stream — no space after final punctuation
    s = SentenceSplitter()
    for t in ["Hello", " there", "!"]:
        s.push(t)
    assert s.flush() == "Hello there!"
