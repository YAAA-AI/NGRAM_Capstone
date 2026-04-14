"""Normalizer module for ngram-predictor."""


class Normalizer:
    """Handle token normalization for training and inference."""

    def normalize(self, text: str) -> str:
        """Normalize a text string."""
        return text.strip().lower()
