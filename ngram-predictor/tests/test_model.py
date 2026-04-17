"""Tests for the n-gram model — NGramModel class."""

import os
import tempfile

from src.model.ngram_model import NGramModel


def _create_token_file(sentences):
    """Helper: write sentences to a temp file and return its path."""
    tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    for sentence in sentences:
        tmpfile.write(" ".join(sentence) + "\n")
    tmpfile.close()
    return tmpfile.name


def _build_test_model():
    """Helper: build a small model for testing."""
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "floor"],
        ["the", "cat", "is", "on", "the", "mat"],
        ["holmes", "looked", "at", "the", "letter"],
        ["holmes", "looked", "at", "watson"],
        ["a", "rare", "word"],  # "rare" and "a" will be low frequency
    ]
    token_file = _create_token_file(sentences)
    model = NGramModel(ngram_order=4, unk_threshold=2)
    model.build_counts_and_probabilities(token_file)
    os.unlink(token_file)
    return model


def test_build_vocab_includes_unk():
    """build_vocab() replaces low-frequency words with <UNK> and includes <UNK> in vocab."""
    model = _build_test_model()
    assert "<UNK>" in model.vocab


def test_build_vocab_replaces_low_frequency():
    """build_vocab() removes words below UNK_THRESHOLD from vocab."""
    model = _build_test_model()
    # "rare" appears only once, threshold is 2
    assert "rare" not in model.vocab


def test_lookup_returns_nonempty_for_seen_context():
    """lookup() returns a non-empty dict for a seen context."""
    model = _build_test_model()
    # "the" appears many times, so unigram at least should work
    result = model.lookup(["the"])
    assert isinstance(result, dict)
    assert len(result) > 0


def test_lookup_returns_nonempty_for_unseen_context():
    """lookup() falls back to unigram for an unseen context."""
    model = _build_test_model()
    result = model.lookup(["zzz", "qqq", "xxx"])
    # Should fall back to unigram (all words map to <UNK>)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_lookup_returns_dict():
    """lookup() always returns a dict."""
    model = _build_test_model()
    result = model.lookup(["holmes", "looked", "at"])
    assert isinstance(result, dict)


def test_probabilities_sum_to_approximately_one():
    """Probabilities for any context sum to approximately 1."""
    model = _build_test_model()
    # Check unigram probabilities sum
    unigram_probs = model.probs.get("1gram", {})
    total = sum(unigram_probs.values())
    assert abs(total - 1.0) < 0.01


def test_save_and_load_model():
    """save_model/save_vocab and load roundtrip correctly."""
    model = _build_test_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.json")
        vocab_path = os.path.join(tmpdir, "vocab.json")
        model.save_model(model_path)
        model.save_vocab(vocab_path)

        loaded = NGramModel()
        loaded.load(model_path, vocab_path)

        assert loaded.ngram_order == model.ngram_order
        assert "<UNK>" in loaded.vocab
        assert "1gram" in loaded.probs
