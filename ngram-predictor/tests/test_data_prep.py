"""Tests for data preparation — Normalizer class."""

import os
import tempfile

from src.data_prep.normalizer import Normalizer


def test_normalize_lowercases():
    """normalize() converts text to lowercase."""
    n = Normalizer()
    result = n.normalize("HELLO WORLD")
    assert result == "hello world"


def test_normalize_removes_punctuation():
    """normalize() removes punctuation characters."""
    n = Normalizer()
    result = n.normalize("hello, world!")
    assert result == "hello world"


def test_normalize_removes_numbers():
    """normalize() removes all digits."""
    n = Normalizer()
    result = n.normalize("chapter 3 begins")
    assert "3" not in result


def test_normalize_strips_whitespace():
    """normalize() collapses extra whitespace."""
    n = Normalizer()
    result = n.normalize("hello   world   test")
    assert result == "hello world test"


def test_normalize_combined():
    """normalize() applies all steps in sequence."""
    n = Normalizer()
    result = n.normalize("Hello, World! Chapter 3...")
    assert result == "hello world chapter"


def test_strip_gutenberg_removes_header_and_footer():
    """strip_gutenberg() removes Gutenberg markers."""
    n = Normalizer()
    text = (
        "Some header text\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "This is the actual content.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Some footer text"
    )
    result = n.strip_gutenberg(text)
    assert "actual content" in result
    assert "header text" not in result
    assert "footer text" not in result


def test_sentence_tokenize_returns_list():
    """sentence_tokenize() returns a list with at least one element on non-empty input."""
    n = Normalizer()
    result = n.sentence_tokenize("Hello world. How are you.")
    assert isinstance(result, list)
    assert len(result) >= 1


def test_word_tokenize_returns_strings_no_empty():
    """word_tokenize() returns a list of strings with no empty tokens."""
    n = Normalizer()
    result = n.word_tokenize("hello world test")
    assert isinstance(result, list)
    assert all(isinstance(w, str) for w in result)
    assert all(len(w) > 0 for w in result)


def test_load_raises_on_missing_folder():
    """load() raises FileNotFoundError for a non-existent folder."""
    n = Normalizer()
    try:
        n.load("/nonexistent/folder/path")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_save_and_load_roundtrip():
    """save() writes sentences that can be read back correctly."""
    n = Normalizer()
    sentences = [["hello", "world"], ["foo", "bar"]]

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "output.txt")
        n.save(sentences, filepath)

        with open(filepath, "r") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 2
        assert lines[0] == "hello world"
        assert lines[1] == "foo bar"
