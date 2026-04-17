"""Evaluator module for ngram-predictor.

Responsible for computing perplexity on a held-out evaluation corpus
using the n-gram model's backoff lookup.
"""

import math
import os
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate model quality by computing perplexity on held-out data.

    Uses NGramModel.lookup() to get probabilities for each word in the
    evaluation corpus and computes cross-entropy and perplexity.
    """

    def __init__(self, model, normalizer):
        """Accept a pre-loaded NGramModel and Normalizer instance.

        Args:
            model: A pre-loaded NGramModel instance.
            normalizer: A Normalizer instance.
        """
        self.model = model
        self.normalizer = normalizer

    def score_word(self, word, context):
        """Return log2 P(word | context) via NGramModel.lookup().

        Args:
            word: The target word to score.
            context: A list of context words.

        Returns:
            log2 probability of word given context, or None if zero
            probability at all orders.
        """
        candidates = self.model.lookup(context)
        if word in candidates and candidates[word] > 0:
            return math.log2(candidates[word])
        return None

    def compute_perplexity(self, eval_file):
        """Compute perplexity over the full evaluation corpus.

        Reads the eval file, normalizes and tokenizes it, then scores
        each word using backoff lookup.

        Args:
            eval_file: Path to the evaluation tokens file.

        Returns:
            A tuple of (perplexity, evaluated_count, skipped_count).
        """
        with open(eval_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the evaluation corpus
        normalized = self.normalizer.normalize(text)
        words = normalized.split() if isinstance(normalized, str) else normalized

        if not words:
            logger.warning("Evaluation corpus is empty")
            return float('inf'), 0, 0

        total_log_prob = 0.0
        evaluated = 0
        skipped = 0

        for i in range(1, len(words)):
            word = words[i]
            # Build context of up to ngram_order - 1 words
            ctx_len = min(i, self.model.ngram_order - 1)
            context = words[i - ctx_len:i]

            log_prob = self.score_word(word, context)
            if log_prob is not None:
                total_log_prob += log_prob
                evaluated += 1
            else:
                skipped += 1

        if evaluated == 0:
            logger.warning("No words could be evaluated")
            return float('inf'), 0, skipped

        skip_ratio = skipped / (evaluated + skipped)
        if skip_ratio > 0.20:
            logger.warning("More than 20%% of words were skipped (%.1f%%)",
                           skip_ratio * 100)

        cross_entropy = -total_log_prob / evaluated
        perplexity = 2 ** cross_entropy

        return perplexity, evaluated, skipped

    def run(self, eval_file):
        """Orchestrate perplexity computation and print results.

        Args:
            eval_file: Path to the evaluation tokens file.
        """
        perplexity, evaluated, skipped = self.compute_perplexity(eval_file)
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Words evaluated: {evaluated:,}")
        print(f"Words skipped (zero probability): {skipped:,}")
        return perplexity, evaluated, skipped


def main():
    """Run the evaluator module standalone."""
    from dotenv import load_dotenv
    from src.model.ngram_model import NGramModel
    from src.data_prep.normalizer import Normalizer

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    load_dotenv(os.path.join(base_dir, "config", ".env"))

    model_path = os.path.join(base_dir, os.environ.get("MODEL", "data/model/model.json"))
    vocab_path = os.path.join(base_dir, os.environ.get("VOCAB", "data/model/vocab.json"))
    eval_tokens = os.path.join(base_dir, os.environ.get("EVAL_TOKENS", "data/processed/eval_tokens.txt"))

    model = NGramModel()
    model.load(model_path, vocab_path)

    normalizer = Normalizer()
    evaluator = Evaluator(model, normalizer)
    evaluator.run(eval_tokens)


if __name__ == "__main__":
    main()
