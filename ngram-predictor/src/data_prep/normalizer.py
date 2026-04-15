from __future__ import annotations

import re
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.request import urlopen

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class Normalizer:
    """Normalize raw text into tokens (words)."""

    lowercase: bool = True
    keep_apostrophes: bool = False

    def normalize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()

        if self.keep_apostrophes:
            # Keep apostrophes inside words (e.g., don't -> don't)
            text = re.sub(r"[^\w\s']", " ", text)
        else:
            text = re.sub(r"[^\w\s]", " ", text)

        return text.split()


def create_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_gutenberg_words(book_id: int, normalizer: Normalizer, ssl_context: ssl.SSLContext | None = None) -> List[str]:
    ssl_context = ssl_context or create_ssl_context()
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}-images.html"
    html = urlopen(url, context=ssl_context).read()
    soup = BeautifulSoup(html, "html.parser")
    return normalizer.normalize(soup.get_text())


def save_tokens(file_path: Path, tokens: List[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(tokens), encoding="utf-8")


def load_tokens(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def prepare_training_tokens(
    output_path: Path,
    normalizer: Normalizer | None = None,
    book_ids: List[int] | None = None,
) -> List[str]:
    normalizer = normalizer or Normalizer()
    book_ids = book_ids or [1661, 834, 108, 2852]

    tokens: List[str] = []
    ssl_context = create_ssl_context()
    for book_id in book_ids:
        try:
            tokens.extend(fetch_gutenberg_words(book_id, normalizer, ssl_context=ssl_context))
        except Exception:
            continue

    if not tokens:
        fallback_text = (
            "holmes looked at watson and watson looked back "
            "the game is afoot and the game is up and over"
        )
        tokens = normalizer.normalize(fallback_text)

    save_tokens(output_path, tokens)
    return tokens
