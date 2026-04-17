"""Tests for evaluation — Evaluator class."""

import os
import tempfile

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.evaluation.evaluator import Evaluator


def _build_test_evaluator():
    """Helper: build a small model and evaluator for testing."""
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "floor"],
        ["the", "cat", "is", "on", "the", "mat"],
        ["holmes", "looked", "at", "the", "letter"],
        ["holmes", "looked", "at", "watson"],
    ]
    tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    for s in sentences:
        tmpfile.write(" ".join(s) + "\n")
    tmpfile.close()

    model = NGramModel(ngram_order=4, unk_threshold=2)
    model.build_counts_and_probabilities(tmpfile.name)
    os.unlink(tmpfile.name)

    normalizer = Normalizer()
    return Evaluator(model, normalizer)


def _create_eval_file(text):
    """Helper: write text to a temp file and return its path."""
    tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmpfile.write(text)
    tmpfile.close()
    return tmpfile.name


def test_score_word_returns_negative_float_for_seen_word():
    """score_word() returns a negative float for a seen word."""
    e = _build_test_evaluator()
    # Use unigram context (empty) so multiple words compete => probability < 1 => log2 < 0
    result = e.score_word("the", [])
    assert result is not None
    assert isinstance(result, float)
    assert result < 0


def test_score_word_returns_none_for_zero_probability():
    """score_word() returns None for a word with zero probability at all orders."""
    e = _build_test_evaluator()
    # Use a word not in vocab at all, with a context that won't match
    result = e.score_word("xyznonexistent", ["aaa", "bbb", "ccc"])
    assert result is None


def test_compute_perplexity_returns_positive_float():
    """compute_perplexity() returns a positive float greater than 1."""
    e = _build_test_evaluator()
    eval_file = _create_eval_file("the cat sat on the mat")
    try:
        perplexity, evaluated, skipped = e.compute_perplexity(eval_file)
        assert isinstance(perplexity, float)
        assert perplexity > 1
        assert evaluated > 0
    finally:
        os.unlink(eval_file)


def test_compute_perplexity_counts_evaluated_and_skipped():
    """compute_perplexity() returns evaluated and skipped counts."""
    e = _build_test_evaluator()
    eval_file = _create_eval_file("the cat sat on the mat zzz qqq")
    try:
        perplexity, evaluated, skipped = e.compute_perplexity(eval_file)
        assert evaluated >= 0
        assert skipped >= 0
        assert evaluated + skipped > 0
    finally:
        os.unlink(eval_file)
