"""Unit tests for LiveSubtitler segmenter."""

from voxtral_server.transcription.segmenter import LiveSubtitler


def test_empty_partial():
    s = LiveSubtitler()
    assert s.on_partial("", 0.0) == []


def test_empty_final():
    s = LiveSubtitler()
    assert s.on_final("", 0.0) == []


def test_short_partial_not_committed_before_stability():
    """Words not stable for 0.5s should not be committed."""
    s = LiveSubtitler(stability_secs=0.5)
    assert s.on_partial("hello world test", 0.0) == []
    # Still not stable at 0.3s
    assert s.on_partial("hello world test", 0.3) == []


def test_partial_committed_after_stability():
    """Words stable for >= 0.5s and text >= 15 chars should be committed."""
    s = LiveSubtitler(stability_secs=0.5, min_chars=15)
    s.on_partial("Dies ist ein langer Satz", 0.0)
    # Same text at 0.6s — all words stable
    segments = s.on_partial("Dies ist ein langer Satz", 0.6)
    assert len(segments) == 1
    assert segments[0] == "Dies ist ein langer Satz"


def test_partial_too_short_not_committed():
    """Text shorter than min_chars should not be committed even if stable."""
    s = LiveSubtitler(stability_secs=0.5, min_chars=15)
    s.on_partial("short", 0.0)
    segments = s.on_partial("short", 0.6)
    assert segments == []


def test_unstable_tail_not_committed():
    """Words that change between partials should not be committed."""
    s = LiveSubtitler(stability_secs=0.5, min_chars=10)
    s.on_partial("Dies ist ein Test", 0.0)
    # Tail changes at 0.6s — "Test" becomes "Beispiel"
    segments = s.on_partial("Dies ist ein Beispiel", 0.6)
    # Only the stable prefix "Dies ist ein" could be committed (13 chars >= 10)
    # But "Beispiel" is new (unstable), so stable_boundary stops at index 3
    assert len(segments) == 1
    assert segments[0] == "Dies ist ein"


def test_final_flushes_all():
    """on_final should flush all uncommitted text."""
    s = LiveSubtitler()
    # Partial — nothing committed (not stable yet)
    s.on_partial("Hallo Welt", 0.0)
    assert s._committed_words == 0
    # Final flushes everything
    segments = s.on_final("Hallo Welt", 0.1)
    assert segments == ["Hallo Welt"]


def test_final_resets_state():
    """After on_final, state should be reset for the next window."""
    s = LiveSubtitler()
    s.on_final("first sentence", 0.0)
    # State reset — new partial starts fresh
    s.on_partial("second sentence here", 1.0)
    segments = s.on_partial("second sentence here", 1.6)
    assert len(segments) == 1
    assert segments[0] == "second sentence here"


def test_max_chars_chunking():
    """Long text should be split into chunks <= max_chars."""
    s = LiveSubtitler(max_chars=20)
    segments = s.on_final("Dies ist ein sehr langer Satz der aufgeteilt werden muss", 0.0)
    assert len(segments) >= 2
    for seg in segments:
        assert len(seg) <= 20, f"Segment too long: '{seg}' ({len(seg)} chars)"


def test_incremental_commits():
    """Words should be committed incrementally as they stabilize."""
    s = LiveSubtitler(stability_secs=0.5, min_chars=5)
    # First batch of words
    s.on_partial("alpha beta gamma", 0.0)
    segments1 = s.on_partial("alpha beta gamma", 0.6)
    assert len(segments1) == 1
    assert segments1[0] == "alpha beta gamma"

    # More words arrive — old ones already committed
    s.on_partial("alpha beta gamma delta epsilon", 1.0)
    segments2 = s.on_partial("alpha beta gamma delta epsilon", 1.6)
    assert len(segments2) == 1
    assert "delta" in segments2[0]
    assert "alpha" not in segments2[0]  # already committed


def test_final_only_uncommitted():
    """on_final should only flush text after the committed cursor."""
    s = LiveSubtitler(stability_secs=0.5, min_chars=5)
    s.on_partial("already committed words", 0.0)
    s.on_partial("already committed words", 0.6)  # commits these

    # Final has more text
    segments = s.on_final("already committed words and more", 1.0)
    assert len(segments) == 1
    assert segments[0] == "and more"


def test_word_boundary_cut():
    """Chunks should be cut at word boundaries."""
    s = LiveSubtitler(max_chars=10)
    segments = s.on_final("abc defgh ijklm", 0.0)
    # "abc defgh" = 9 chars (fits), "ijklm" = 5 chars
    assert segments[0] == "abc defgh"
    assert segments[1] == "ijklm"


def test_growing_text_cumulative():
    """Simulate growing_text that grows over successive partials."""
    s = LiveSubtitler(stability_secs=0.3, min_chars=10)
    s.on_partial("Der Sprecher", 0.0)
    s.on_partial("Der Sprecher sagt", 0.2)
    # "Der" and "Sprecher" have been stable since 0.0 (0.4s > 0.3s threshold)
    # "sagt" is new at 0.2, so not stable yet at 0.4
    segments = s.on_partial("Der Sprecher sagt etwas", 0.4)
    assert len(segments) == 1
    assert segments[0] == "Der Sprecher"
