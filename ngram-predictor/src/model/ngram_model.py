from __future__ import annotations

import math
import ssl
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from time import sleep
from urllib.request import urlopen

import urllib3
from bs4 import BeautifulSoup

from src.data_prep.normalizer import Normalizer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NGram = Tuple[str, ...]
Context = Tuple[str, ...]


def create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context that skips certificate verification."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def build_ngrams(words: List[str], n: int) -> List[NGram]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def count_ngrams(word_lists: List[List[str]], n: int) -> Counter:
    all_ngrams: List[NGram] = []
    for words in word_lists:
        all_ngrams.extend(build_ngrams(words, n))
    return Counter(all_ngrams)


def build_context_index(ngram_counts: Counter) -> Dict[Context, List[Tuple[str, int]]]:
    context_index: Dict[Context, List[Tuple[str, int]]] = defaultdict(list)
    for ngram, count in ngram_counts.items():
        context_index[ngram[:-1]].append((ngram[-1], count))

    for ctx in context_index:
        context_index[ctx].sort(key=lambda x: x[1], reverse=True)

    return dict(context_index)


@dataclass
class NGramModel:
    """N-gram language model with MLE backoff and optional unigram fallback."""

    # Ordered from highest order to lower order: max_n..2
    context_indices: List[Dict[Context, List[Tuple[str, int]]]]
    unigram_probs: Dict[str, float]
    vocab_size: int
    normalizer: Normalizer

    @classmethod
    def from_word_lists(
        cls,
        word_lists: List[List[str]],
        max_n: int = 4,
        normalizer: Optional[Normalizer] = None,
        progress_cb=None,
    ) -> "NGramModel":
        normalizer = normalizer or Normalizer()

        if progress_cb:
            progress_cb("Building n-gram counts...")

        context_indices: List[Dict[Context, List[Tuple[str, int]]]] = []
        for n in range(max_n, 1, -1):
            ngram_counts = count_ngrams(word_lists, n=n)
            if progress_cb:
                progress_cb(f"Built {len(ngram_counts)} unique {n}-grams")
            context_indices.append(build_context_index(ngram_counts))

        # Unigrams
        unigram_counts = count_ngrams(word_lists, n=1)
        total = sum(unigram_counts.values()) or 1
        unigram_probs = {ng[0]: c / total for ng, c in unigram_counts.items()}

        if progress_cb:
            progress_cb("Model ready!")

        return cls(context_indices, unigram_probs, len(unigram_probs), normalizer)

    @classmethod
    def from_gutenberg_ids(
        cls,
        book_ids: List[int],
        max_n: int = 4,
        normalizer: Optional[Normalizer] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        progress_cb=None,
        max_retries: int = 3,
    ) -> "NGramModel":
        ssl_context = ssl_context or create_ssl_context()
        normalizer = normalizer or Normalizer()

        links = [f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}-images.html" for bid in book_ids]
        word_lists: List[List[str]] = []

        for i, link in enumerate(links, 1):
            words: List[str] = []
            for attempt in range(1, max_retries + 1):
                try:
                    if progress_cb:
                        progress_cb(f"Downloading {i}/{len(links)} (attempt {attempt}) ...")
                    html = urlopen(link, context=ssl_context).read()
                    soup = BeautifulSoup(html, "html.parser")
                    words = normalizer.normalize(soup.get_text())
                    break
                except Exception as e:
                    if attempt < max_retries:
                        sleep(0.1 * attempt)
                    else:
                        if progress_cb:
                            progress_cb(f"FAILED book {book_ids[i-1]} ({type(e).__name__})")
                        words = []

            word_lists.append(words)

        return cls.from_word_lists(word_lists, max_n=max_n, normalizer=normalizer, progress_cb=progress_cb)

    def predict(self, user_input: str, top_k: int = 5, use_unigram_fallback: bool = False) -> List[str]:
        num = len(self.context_indices)
        words = self.normalizer.normalize(user_input)[-num:]
        if not words:
            return []

        for j, ctx_index in enumerate(self.context_indices):
            ctx_len = min(len(words), num - j)
            ctx = tuple(words[-ctx_len:]) if ctx_len > 0 else ()
            if ctx in ctx_index:
                results = [w for w, _ in ctx_index[ctx][:top_k]]
                if use_unigram_fallback and len(results) < top_k:
                    results = self._fill_with_unigrams(results, top_k)
                return results

        if use_unigram_fallback:
            ranked = sorted(self.unigram_probs, key=self.unigram_probs.get, reverse=True)
            return ranked[:top_k]

        return []

    def _fill_with_unigrams(self, current: List[str], top_k: int) -> List[str]:
        seen = set(current)
        ranked = sorted(self.unigram_probs, key=self.unigram_probs.get, reverse=True)
        for w in ranked:
            if w not in seen:
                current.append(w)
                seen.add(w)
            if len(current) >= top_k:
                break
        return current

    def evaluate(self, eval_words: List[str], use_unigram_fallback: bool = False) -> Tuple[float, int, int]:
        """Return (perplexity, evaluated_count, skipped_count)."""
        num = len(self.context_indices)
        ctx_lengths = [num - j for j in range(num)]

        # Precompute log-prob tables
        lookup_tables = []
        for ctx_index in self.context_indices:
            table = {}
            for ctx, candidates in ctx_index.items():
                total = sum(c for _, c in candidates)
                if total <= 0:
                    continue
                log2_total = math.log2(total)
                table[ctx] = {w: math.log2(c) - log2_total for w, c in candidates if c > 0}
            lookup_tables.append(table)

        total_log_prob = 0.0
        evaluated = 0
        skipped = 0

        for i in range(1, len(eval_words)):
            word = eval_words[i]
            found = False

            for order, table in zip(ctx_lengths, lookup_tables):
                ctx_len = min(i, order)
                ctx = tuple(eval_words[i - ctx_len:i])
                probs = table.get(ctx)
                if probs is None:
                    continue
                lp = probs.get(word)
                if lp is not None:
                    total_log_prob += lp
                    evaluated += 1
                    found = True
                    break

            if not found and use_unigram_fallback:
                p = self.unigram_probs.get(word, 0.0)
                if p > 0:
                    total_log_prob += math.log2(p)
                    evaluated += 1
                    found = True

            if not found:
                skipped += 1

        if evaluated == 0:
            return float("inf"), 0, skipped

        cross_entropy = -total_log_prob / evaluated
        perplexity = 2 ** cross_entropy
        return perplexity, evaluated, skipped
