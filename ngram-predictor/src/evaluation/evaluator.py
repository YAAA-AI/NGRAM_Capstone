from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from urllib.request import urlopen

from bs4 import BeautifulSoup

from src.model.ngram_model import NGramModel, create_ssl_context


@dataclass
class Evaluator:
    model: NGramModel

    def evaluate_gutenberg_book(self, book_id: int, smoothing: str = "none") -> Dict[str, Any]:
        use_fallback = (smoothing == "mle-backoff")
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}-images.html"

        html = urlopen(url, context=create_ssl_context()).read()
        soup = BeautifulSoup(html, "html.parser")
        eval_words = self.model.normalizer.normalize(soup.get_text())

        if not eval_words:
            return {"error": "Evaluation book is empty.", "book_id": book_id}

        perplexity, evaluated, skipped = self.model.evaluate(eval_words, use_unigram_fallback=use_fallback)
        return {
            "book_id": book_id,
            "smoothing_used": smoothing,
            "perplexity": perplexity,
            "evaluated": evaluated,
            "skipped": skipped,
        }
