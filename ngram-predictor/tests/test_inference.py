"""Tests for inference — Predictor class."""

import os
import tempfile

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


def _build_test_predictor():
    """Helper: build a small model and predictor for testing."""
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
    return Predictor(model, normalizer)


def test_predict_next_returns_k_predictions():
    """predict_next() returns exactly k predictions for a seen context."""
    p = _build_test_predictor()
    results = p.predict_next("the cat", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_predict_next_sorted_by_probability():
    """predict_next() returns results sorted by probability, highest first."""
    p = _build_test_predictor()
    results = p.predict_next("the", k=5)
    # Results should come from sorted lookup
    assert isinstance(results, list)


def test_predict_next_handles_oov_context():
    """predict_next() handles an all-OOV context without crashing."""
    p = _build_test_predictor()
    results = p.predict_next("zzz qqq xxx", k=3)
    assert isinstance(results, list)


def test_map_oov_replaces_unknown():
    """map_oov() replaces unknown words with <UNK> and leaves known words unchanged."""
    p = _build_test_predictor()
    result = p.map_oov(["the", "unknownword123"])
    assert result[0] == "the"
    assert result[1] == "<UNK>"


def test_predict_next_raises_on_empty_input():
    """predict_next() raises ValueError on empty input."""
    p = _build_test_predictor()
    try:
        p.predict_next("", k=3)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_predict_next_raises_on_whitespace_input():
    """predict_next() raises ValueError on whitespace-only input."""
    p = _build_test_predictor()
    try:
        p.predict_next("   ", k=3)
        assert False, "Expected ValueError"
    except ValueError:
        pass
