"""N-gram model implementation."""


class NGramModel:
    """Simple n-gram model class."""

    def __init__(self):
        self.vocab = {}

    def train(self, tokens):
        """Train the n-gram model on a token sequence."""
        self.vocab = {token: tokens.count(token) for token in set(tokens)}
