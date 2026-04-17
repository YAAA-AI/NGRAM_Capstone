"""Predictor module for ngram-predictor.

Responsible for accepting a pre-loaded NGramModel and Normalizer via the
constructor, normalizing input text, and returning the top-k predicted
next words sorted by probability. Backoff lookup is delegated to
NGramModel.lookup().
"""

import os
import logging

logger = logging.getLogger(__name__)


class Predictor:
    """Generate next-word predictions from an n-gram model.

    Accepts a pre-loaded NGramModel and Normalizer instance. Normalizes
    user input, maps out-of-vocabulary words, and delegates lookup to
    the model's backoff logic.
    """

    def __init__(self, model, normalizer):
        """Accept a pre-loaded NGramModel and Normalizer instance.

        Do not load files here.

        Args:
            model: A pre-loaded NGramModel instance.
            normalizer: A Normalizer instance.
        """
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text):
        """Normalize input text and extract context words.

        Calls Normalizer.normalize(text) and extracts the last
        NGRAM_ORDER - 1 words as context.

        Args:
            text: The raw user input string.

        Returns:
            A list of context words (up to ngram_order - 1 words).
        """
        normalized = self.normalizer.normalize(text)
        words = normalized.split() if isinstance(normalized, str) else normalized
        max_context = self.model.ngram_order - 1
        return words[-max_context:] if words else []

    def map_oov(self, context):
        """Replace out-of-vocabulary words with <UNK>.

        Leaves known words unchanged.

        Args:
            context: A list of context words.

        Returns:
            A list of words with OOV words replaced by '<UNK>'.
        """
        return [w if w in self.model.vocab else "<UNK>" for w in context]

    def predict_next(self, text, k=3):
        """Orchestrate normalize -> map_oov -> NGramModel.lookup() -> return top-k words.

        Args:
            text: The raw user input string.
            k: Number of top predictions to return.

        Returns:
            A list of up to k predicted next words sorted by probability
            (highest first). Returns empty list if no predictions found.

        Raises:
            ValueError: If input text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty. Please type at least one word.")

        context = self.normalize(text)
        if not context:
            return []

        context = self.map_oov(context)
        candidates = self.model.lookup(context)

        if not candidates:
            return []

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_candidates[:k]]


def main():
    """Run the predictor module standalone."""
    from dotenv import load_dotenv
    from src.model.ngram_model import NGramModel
    from src.data_prep.normalizer import Normalizer

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    load_dotenv(os.path.join(base_dir, "config", ".env"))

    model_path = os.path.join(base_dir, os.environ.get("MODEL", "data/model/model.json"))
    vocab_path = os.path.join(base_dir, os.environ.get("VOCAB", "data/model/vocab.json"))
    top_k = int(os.environ.get("TOP_K", "3"))

    model = NGramModel()
    model.load(model_path, vocab_path)

    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)

    print("N-gram Predictor ready. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\n> ")
            if user_input.strip().lower() == "quit":
                print("Goodbye.")
                break
            predictions = predictor.predict_next(user_input, k=top_k)
            print(f"Predictions: {predictions}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
