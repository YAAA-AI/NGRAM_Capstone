"""Predictor module for ngram-predictor."""


class Predictor:
    """Generate next-token predictions from an n-gram model."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, k=5):
        """Return top-k predictions for a given context."""
        return []
