"""N-gram model implementation.

Responsible for building, storing, and exposing n-gram probability tables
and backoff lookup across all orders from 1 up to NGRAM_ORDER.
"""

import json
import math
import os
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class NGramModel:
    """N-gram language model with MLE probabilities and backoff lookup.

    Builds vocabulary, counts n-grams at all orders from 1 up to ngram_order,
    computes MLE probabilities, saves/loads model artifacts, and provides
    a single backoff lookup method used by Predictor and Evaluator.

    Attributes:
        ngram_order: Maximum n-gram order.
        unk_threshold: Minimum word count to be included in vocabulary.
        vocab: Set of vocabulary words (includes <UNK>).
        counts: Dict mapping order keys (e.g. '1gram') to Counter of n-gram counts.
        probs: Dict mapping order keys to dicts of {context: {word: probability}}.
    """

    def __init__(self, ngram_order=4, unk_threshold=3):
        """Initialize the model with configuration values.

        Args:
            ngram_order: Maximum n-gram order (read from config/.env).
            unk_threshold: Minimum word frequency threshold (from config/.env).
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.counts = {}
        self.probs = {}
        # In-memory attributes (used by from_word_lists for Streamlit UI)
        self.context_indices = []
        self.unigram_probs = {}
        self.vocab_size = 0

    def build_vocab(self, token_file):
        """Build vocabulary from a tokenized file. Replace low-frequency words with <UNK>.

        Reads the token file, counts word frequencies, and replaces any word
        appearing fewer than unk_threshold times with <UNK>.

        Args:
            token_file: Path to the tokenized training file (one sentence per line).

        Returns:
            A list of sentences (lists of tokens) with low-frequency words
            replaced by <UNK>.
        """
        word_counts = Counter()
        sentences = []

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)
                    word_counts.update(tokens)

        self.vocab = set()
        for word, count in word_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(word)
        self.vocab.add("<UNK>")

        # Replace low-frequency words with <UNK>
        processed_sentences = []
        for sentence in sentences:
            processed = [w if w in self.vocab else "<UNK>" for w in sentence]
            processed_sentences.append(processed)

        logger.info("Vocabulary size: %d (UNK_THRESHOLD=%d)", len(self.vocab), self.unk_threshold)
        return processed_sentences

    def build_counts_and_probabilities(self, token_file):
        """Count all n-grams at orders 1 through ngram_order and compute MLE probabilities.

        Probabilities depend on counts, so they are computed together to avoid
        hidden ordering bugs.

        Args:
            token_file: Path to the tokenized training file (one sentence per line).
        """
        sentences = self.build_vocab(token_file)

        # Build counts at all orders
        self.counts = {}
        for order in range(1, self.ngram_order + 1):
            key = f"{order}gram"
            counter = Counter()
            for sentence in sentences:
                for i in range(len(sentence) - order + 1):
                    ngram = tuple(sentence[i:i + order])
                    counter[ngram] += 1
            self.counts[key] = counter
            logger.info("Built %d unique %d-grams", len(counter), order)

        # Compute MLE probabilities
        self.probs = {}

        # Unigram probabilities: count / total words
        unigram_key = "1gram"
        total_words = sum(self.counts[unigram_key].values())
        self.probs[unigram_key] = {}
        for ngram, count in self.counts[unigram_key].items():
            word = ngram[0]
            self.probs[unigram_key][word] = count / total_words

        # Higher-order probabilities: count(ngram) / count(prefix)
        for order in range(2, self.ngram_order + 1):
            key = f"{order}gram"
            prefix_key = f"{order - 1}gram"
            self.probs[key] = {}
            for ngram, count in self.counts[key].items():
                context = " ".join(ngram[:-1])
                word = ngram[-1]
                prefix = ngram[:-1]
                prefix_count = self.counts[prefix_key][prefix]
                if prefix_count > 0:
                    if context not in self.probs[key]:
                        self.probs[key][context] = {}
                    self.probs[key][context][word] = count / prefix_count

        logger.info("MLE probabilities computed for orders 1 through %d", self.ngram_order)

    def lookup(self, context):
        """Backoff lookup: try highest-order context first, fall back to lower orders.

        This is the single source of backoff logic in the project. Returns a dict
        of {word: probability} from the highest order that matches. Returns empty
        dict if no match at any order.

        Args:
            context: A list or tuple of context words (already normalized and OOV-mapped).

        Returns:
            A dict of {word: probability} from the first matching order,
            or an empty dict if no match at any order.
        """
        context = list(context)

        # Map OOV words in context to <UNK>
        context = [w if w in self.vocab else "<UNK>" for w in context]

        for order in range(self.ngram_order, 0, -1):
            key = f"{order}gram"
            if key not in self.probs:
                continue

            if order == 1:
                # Unigram: return all unigram probabilities
                return dict(self.probs[key])

            ctx_len = order - 1
            if len(context) >= ctx_len:
                ctx_words = context[-ctx_len:]
                ctx_str = " ".join(ctx_words)
                if ctx_str in self.probs[key]:
                    return dict(self.probs[key][ctx_str])

        return {}

    def save_model(self, model_path):
        """Save all probability tables to model.json.

        Args:
            model_path: Path to write the model JSON file.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.probs, f, indent=2)
        logger.info("Model saved to %s", model_path)

    def save_vocab(self, vocab_path):
        """Save vocabulary list to vocab.json.

        Args:
            vocab_path: Path to write the vocabulary JSON file.
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.vocab), f, indent=2)
        logger.info("Vocab saved to %s (%d words)", vocab_path, len(self.vocab))

    def load(self, model_path, vocab_path):
        """Load model.json and vocab.json into the instance.

        Called once in main() before passing the model to Predictor.

        Args:
            model_path: Path to model.json.
            vocab_path: Path to vocab.json.

        Raises:
            FileNotFoundError: If model or vocab file does not exist.
            json.JSONDecodeError: If files are malformed.
        """
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                self.probs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"model.json not found at {model_path}. Run the Model module first."
            )
        except json.JSONDecodeError:
            raise json.JSONDecodeError(
                f"model.json is malformed at {model_path}. Re-run the Model module.", "", 0
            )

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = set(json.load(f))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"vocab.json not found at {vocab_path}. Run the Model module first."
            )

        # Determine ngram_order from loaded model keys
        max_order = 0
        for key in self.probs:
            order = int(key.replace("gram", ""))
            if order > max_order:
                max_order = order
        self.ngram_order = max_order

        logger.info("Model loaded from %s (order=%d, vocab=%d)",
                     model_path, self.ngram_order, len(self.vocab))

    # ------------------------------------------------------------------
    # In-memory model building and prediction (for Streamlit UI)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ngram_tuples(words, n):
        """Build a list of n-gram tuples from a word list.

        Args:
            words: List of words.
            n: N-gram size.

        Returns:
            A list of n-gram tuples.
        """
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def _count_ngram_tuples(word_lists, n):
        """Count n-gram frequencies across multiple word lists.

        Args:
            word_lists: List of word lists.
            n: N-gram size.

        Returns:
            A Counter mapping n-gram tuples to counts.
        """
        all_ngrams = []
        for words in word_lists:
            all_ngrams.extend(
                NGramModel._build_ngram_tuples(words, n)
            )
        return Counter(all_ngrams)

    @staticmethod
    def _build_context_index(ngram_counts):
        """Build a lookup index mapping (n-1)-word contexts to candidate next words.

        Args:
            ngram_counts: Counter of n-gram tuples.

        Returns:
            Dict mapping context tuples to sorted (word, count) lists.
        """
        context_index = defaultdict(list)
        for ngram, count in ngram_counts.items():
            context = ngram[:-1]
            context_index[context].append((ngram[-1], count))
        for context in context_index:
            context_index[context].sort(key=lambda x: x[1], reverse=True)
        return dict(context_index)

    @classmethod
    def from_word_lists(cls, word_lists, max_n=4, progress_placeholder=None):
        """Build a model from already-extracted word lists (in-memory).

        Used by the Streamlit UI to build a model without file I/O.

        Args:
            word_lists: List of word lists (one per document).
            max_n: Maximum n-gram order.
            progress_placeholder: Optional Streamlit container for progress.

        Returns:
            An NGramModel instance with context_indices and unigram_probs.
        """
        instance = cls(ngram_order=max_n)

        if progress_placeholder:
            progress_placeholder.write("Building n-gram counts...")

        instance.context_indices = []
        for n in range(max_n, 1, -1):
            ngram_counts = cls._count_ngram_tuples(word_lists, n=n)
            if progress_placeholder:
                progress_placeholder.write(
                    f"Built {len(ngram_counts)} unique {n}-grams")
            instance.context_indices.append(
                cls._build_context_index(ngram_counts))

        unigram_counts = cls._count_ngram_tuples(word_lists, n=1)
        if progress_placeholder:
            progress_placeholder.write(
                f"Built {len(unigram_counts)} unique 1-grams")
        total_unigrams = sum(unigram_counts.values())
        instance.unigram_probs = {
            w[0]: c / total_unigrams for w, c in unigram_counts.items()
        }
        instance.vocab_size = len(instance.unigram_probs)

        if progress_placeholder:
            progress_placeholder.write(
                "Model ready! Start typing to see predictions.")

        return instance

    def predict_from_words(self, words, top_k=5,
                           use_unigram_fallback=False,
                           smoothing="mle-backoff"):
        """Predict next words from a pre-normalized word list.

        Uses context_indices built by from_word_lists(). Supports MLE
        backoff and Katz smoothing.

        Args:
            words: List of normalized words.
            top_k: Number of predictions to return.
            use_unigram_fallback: Whether to fall back to unigram probs.
            smoothing: 'mle-backoff' or 'katz'.

        Returns:
            A list of up to top_k predicted next words.
        """
        num = len(self.context_indices)
        words = words[-num:]
        if not words:
            return []

        if smoothing == "katz":
            return self._predict_katz(words, top_k)

        for j, context_index in enumerate(self.context_indices):
            context_len = min(len(words), num - j)
            context = tuple(words[-context_len:]) if context_len > 0 else ()
            if context in context_index:
                results = [w for w, c in context_index[context][:top_k]]
                if use_unigram_fallback and len(results) < top_k:
                    seen = set(results)
                    ranked = sorted(self.unigram_probs,
                                    key=self.unigram_probs.get, reverse=True)
                    for w in ranked:
                        if w not in seen:
                            results.append(w)
                            seen.add(w)
                            if len(results) >= top_k:
                                break
                return results

        if use_unigram_fallback:
            ranked = sorted(self.unigram_probs,
                            key=self.unigram_probs.get, reverse=True)
            return ranked[:top_k]

        return []

    def evaluate_words(self, eval_words, use_unigram_fallback=False,
                       smoothing="mle-backoff"):
        """Compute perplexity on a pre-normalized word list.

        Uses context_indices built by from_word_lists().

        Args:
            eval_words: List of normalized words.
            use_unigram_fallback: Whether to use unigram fallback.
            smoothing: 'mle-backoff' or 'katz'.

        Returns:
            A tuple of (perplexity, evaluated_count, skipped_count).
        """
        if smoothing == "katz":
            return self._evaluate_katz(eval_words)

        num = len(self.context_indices)
        ctx_lengths = [num - j for j in range(num)]
        n_words = len(eval_words)
        total_log_prob = 0.0
        evaluated = 0
        skipped = 0

        lookup_tables = []
        for ctx_index in self.context_indices:
            table = {}
            for ctx, candidates in ctx_index.items():
                total = sum(c for _, c in candidates)
                log2_total = math.log2(total)
                table[ctx] = {
                    w: math.log2(c) - log2_total for w, c in candidates
                }
            lookup_tables.append(table)

        for i in range(1, n_words):
            word = eval_words[i]
            found = False
            for order, table in zip(ctx_lengths, lookup_tables):
                ctx_len = min(i, order)
                ctx = tuple(eval_words[i - ctx_len:i])
                word_probs = table.get(ctx)
                if word_probs is not None:
                    lp = word_probs.get(word)
                    if lp is not None:
                        total_log_prob += lp
                        evaluated += 1
                        found = True
                        break
            if not found:
                if use_unigram_fallback:
                    p = self.unigram_probs.get(word, 0)
                    if p > 0:
                        total_log_prob += math.log2(p)
                        evaluated += 1
                        found = True
                if not found:
                    skipped += 1

        if evaluated == 0:
            return float('inf'), 0, skipped

        cross_entropy = -total_log_prob / evaluated
        perplexity = 2 ** cross_entropy
        return perplexity, evaluated, skipped

    # --- Katz backoff ---

    def _build_katz_tables(self, k=5):
        """Precompute Katz backoff discount probabilities and alpha weights.

        Args:
            k: Discount threshold.
        """
        num = len(self.context_indices)
        self._katz_disc_probs = [None] * num
        self._katz_alphas = [None] * num

        for idx in range(num - 1, -1, -1):
            ctx_index = self.context_indices[idx]
            all_counts = []
            for candidates in ctx_index.values():
                for _, count in candidates:
                    all_counts.append(count)
            freq_of_freq = Counter(all_counts)
            n1 = freq_of_freq.get(1, 0)
            nk1 = freq_of_freq.get(k + 1, 0)

            def _discount(r, _ff=freq_of_freq, _n1=n1, _nk1=nk1, _k=k):
                if r == 0:
                    return 0.0
                if r > _k:
                    return 1.0
                nr = _ff.get(r, 0)
                nr1 = _ff.get(r + 1, 0)
                if nr == 0 or _n1 == 0:
                    return 1.0
                raw = (r + 1) * nr1 / (r * nr)
                kterm = (_k + 1) * _nk1 / _n1
                denom = 1.0 - kterm
                if abs(denom) < 1e-10:
                    return 1.0
                d = (raw - kterm) / denom
                return max(0.0, min(1.0, d))

            disc_probs = {}
            alphas = {}
            for ctx, candidates in ctx_index.items():
                total = sum(c for _, c in candidates)
                probs = {}
                seen_mass = 0.0
                backoff_seen_mass = 0.0
                for word, count in candidates:
                    d = _discount(count)
                    p = d * count / total
                    probs[word] = p
                    seen_mass += p
                    shorter = ctx[1:] if len(ctx) > 0 else ()
                    bp = self._katz_lower_prob(word, shorter, idx + 1)
                    backoff_seen_mass += bp
                disc_probs[ctx] = probs
                if backoff_seen_mass < 1.0 - 1e-10:
                    alphas[ctx] = max(
                        0.0, (1.0 - seen_mass) / (1.0 - backoff_seen_mass))
                else:
                    alphas[ctx] = 0.0
            self._katz_disc_probs[idx] = disc_probs
            self._katz_alphas[idx] = alphas

    def _katz_lower_prob(self, word, context, order_idx):
        """Compute Katz backoff probability recursively.

        Args:
            word: Target word.
            context: Tuple of context words.
            order_idx: Index into self.context_indices.

        Returns:
            Katz backoff probability.
        """
        num = len(self.context_indices)
        if order_idx >= num:
            return self.unigram_probs.get(
                word, 1.0 / (self.vocab_size + 1))

        disc_probs = self._katz_disc_probs[order_idx]
        if disc_probs is not None and context in disc_probs:
            if word in disc_probs[context]:
                return disc_probs[context][word]
            else:
                alpha = self._katz_alphas[order_idx].get(context, 0.0)
                shorter = context[1:] if len(context) > 0 else ()
                return alpha * self._katz_lower_prob(
                    word, shorter, order_idx + 1)
        else:
            shorter = context[1:] if len(context) > 0 else ()
            return self._katz_lower_prob(word, shorter, order_idx + 1)

    def _predict_katz(self, words, top_k=5):
        """Predict next words using Katz backoff smoothing.

        Args:
            words: Normalized input words.
            top_k: Number of predictions.

        Returns:
            A list of up to top_k predicted next words.
        """
        if not hasattr(self, '_katz_disc_probs'):
            self._build_katz_tables()

        num = len(self.context_indices)
        for j in range(num):
            context_len = min(len(words), num - j)
            context = tuple(words[-context_len:]) if context_len > 0 else ()
            if context in self._katz_disc_probs[j]:
                seen = self._katz_disc_probs[j][context]
                results = sorted(
                    seen.keys(), key=seen.get, reverse=True)[:top_k]
                if len(results) < top_k:
                    alpha = self._katz_alphas[j].get(context, 0.0)
                    if alpha > 0:
                        seen_set = set(results)
                        ranked = sorted(
                            self.unigram_probs,
                            key=self.unigram_probs.get, reverse=True)
                        for w in ranked:
                            if w not in seen_set:
                                results.append(w)
                                seen_set.add(w)
                                if len(results) >= top_k:
                                    break
                return results

        ranked = sorted(self.unigram_probs,
                        key=self.unigram_probs.get, reverse=True)
        return ranked[:top_k]

    def _evaluate_katz(self, eval_words):
        """Compute perplexity using Katz backoff smoothing.

        Args:
            eval_words: List of normalized words.

        Returns:
            A tuple of (perplexity, evaluated_count, skipped_count).
        """
        if not hasattr(self, '_katz_disc_probs'):
            self._build_katz_tables()

        num = len(self.context_indices)
        n_words = len(eval_words)
        total_log_prob = 0.0
        evaluated = 0
        skipped = 0

        for i in range(1, n_words):
            word = eval_words[i]
            context_len = min(i, num)
            context = tuple(eval_words[i - context_len:i])
            start_idx = num - context_len
            prob = self._katz_lower_prob(word, context, start_idx)
            if prob > 0:
                total_log_prob += math.log2(prob)
                evaluated += 1
            else:
                skipped += 1

        if evaluated == 0:
            return float('inf'), 0, skipped

        cross_entropy = -total_log_prob / evaluated
        perplexity = 2 ** cross_entropy
        return perplexity, evaluated, skipped


def main():
    """Run the model module standalone."""
    from dotenv import load_dotenv

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    load_dotenv(os.path.join(base_dir, "config", ".env"))

    ngram_order = int(os.environ.get("NGRAM_ORDER", "4"))
    unk_threshold = int(os.environ.get("UNK_THRESHOLD", "3"))
    train_tokens = os.path.join(base_dir, os.environ.get("TRAIN_TOKENS", "data/processed/train_tokens.txt"))
    model_path = os.path.join(base_dir, os.environ.get("MODEL", "data/model/model.json"))
    vocab_path = os.path.join(base_dir, os.environ.get("VOCAB", "data/model/vocab.json"))

    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
    model.build_counts_and_probabilities(train_tokens)
    model.save_model(model_path)
    model.save_vocab(vocab_path)


if __name__ == "__main__":
    main()
