"""Normalizer module for ngram-predictor."""


import re


class Normalizer:
    """Handle token normalization for training and inference."""

    def normalize(self, text: str) -> str:
        """Convert text to lowercase, remove punctuation, and split into words.

        Args:
            text: Raw input text string.

        Returns:
            A list of lowercase word strings.
        """
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()
