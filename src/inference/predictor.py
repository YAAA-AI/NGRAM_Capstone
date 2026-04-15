from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.model.ngram_model import NGramModel


@dataclass
class Predictor:
    model: NGramModel
    smoothing: str = "none"  # "none" or "mle-backoff"
    top_k: int = 5

    def predict_next(self, text: str, k: int) -> List[str]:
        use_fallback = (self.smoothing == "mle-backoff")
        return self.model.predict(text, top_k=k, use_unigram_fallback=use_fallback)

    def suggest(self, text: str) -> List[str]:
        return self.predict_next(text, self.top_k)

    def apply_suggestion(self, current_text: str, next_word: str) -> str:
        return current_text.rstrip() + " " + next_word + " "
