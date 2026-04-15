from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


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
