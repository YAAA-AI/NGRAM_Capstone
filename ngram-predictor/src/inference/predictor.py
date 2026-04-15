from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.model.ngram_model import NGramModel


@dataclass
class Predictor:
    model: NGramModel
    smoothing: str = "none"  # "none" or "mle-backoff"
    top_k: int = 5

    def suggest(self, text: str) -> List[str]:
        use_fallback = (self.smoothing == "mle-backoff")
        return self.model.predict(text, top_k=self.top_k, use_unigram_fallback=use_fallback)

    def apply_suggestion(self, current_text: str, next_word: str) -> str:
        return current_text.rstrip() + " " + next_word + " "
