"""Tests for UI — PredictorUI class."""

import os
import tempfile

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.ui.app import PredictorUI


def _build_test_ui():
    """Helper: build a small model, predictor, and UI for testing."""
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
    predictor = Predictor(model, normalizer)
    return PredictorUI(predictor)


def test_get_predictions_returns_list_of_strings():
    """get_predictions() returns a list of strings."""
    ui = _build_test_ui()
    result = ui.get_predictions("the cat", k=3)
    assert isinstance(result, list)
    assert all(isinstance(w, str) for w in result)


def test_get_predictions_handles_empty_input():
    """get_predictions() handles empty input without crashing."""
    ui = _build_test_ui()
    result = ui.get_predictions("", k=3)
    assert isinstance(result, list)
    assert len(result) == 0


def test_get_predictions_handles_whitespace_input():
    """get_predictions() handles whitespace-only input without crashing."""
    ui = _build_test_ui()
    result = ui.get_predictions("   ", k=3)
    assert isinstance(result, list)
    assert len(result) == 0
