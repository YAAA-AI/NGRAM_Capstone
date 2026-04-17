"""Normalizer module for ngram-predictor.

Responsible for loading, cleaning, tokenizing, and saving the corpus.
Processes raw Project Gutenberg text files into clean, tokenized output
suitable for n-gram model training.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


class Normalizer:
    """Handle loading, cleaning, tokenizing, and saving of text corpora.

    Used in two contexts:
    - Data Prep (Module 1): processes whole raw files via the full pipeline.
    - Inference (Module 3): normalizes single input strings via normalize().
    """

    def load(self, folder_path):
        """Load all .txt files from a folder.

        Args:
            folder_path: Path to the folder containing .txt files.

        Returns:
            A list of strings, one per file loaded.

        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env."
            )
        texts = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                logger.info("Loading %s", filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
        logger.info("Loaded %d files from %s", len(texts), folder_path)
        return texts

    def strip_gutenberg(self, text):
        """Remove Gutenberg header and footer from the text.

        Removes all text before and including the START marker,
        and all text from and including the END marker.

        Args:
            text: Raw text string from a Gutenberg book.

        Returns:
            The text with header and footer removed.
        """
        start_pattern = r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
        end_pattern = r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"

        start_match = re.search(start_pattern, text, re.IGNORECASE)
        if start_match:
            text = text[start_match.end():]

        end_match = re.search(end_pattern, text, re.IGNORECASE)
        if end_match:
            text = text[:end_match.start()]

        return text

    def lowercase(self, text):
        """Lowercase all text.

        Args:
            text: Input text string.

        Returns:
            The text converted to lowercase.
        """
        return text.lower()

    def remove_punctuation(self, text):
        """Remove all punctuation from text.

        Args:
            text: Input text string.

        Returns:
            The text with punctuation replaced by spaces.
        """
        return re.sub(r'[^\w\s]', ' ', text)

    def remove_numbers(self, text):
        """Remove all numbers from text.

        Args:
            text: Input text string.

        Returns:
            The text with digits removed.
        """
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text):
        """Remove extra whitespace and blank lines.

        Args:
            text: Input text string.

        Returns:
            The text with consecutive whitespace collapsed to single spaces,
            and leading/trailing whitespace stripped.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def normalize(self, text):
        """Apply all normalization steps in order.

        Applies: lowercase -> remove punctuation -> remove numbers -> remove whitespace.
        This is the single method that other modules call to normalize text consistently.

        Args:
            text: Raw input text string.

        Returns:
            A normalized string (lowercase, no punctuation, no numbers,
            no extra whitespace).
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        """Split text into a list of sentences.

        Uses common sentence-ending punctuation as delimiters. Falls back
        to splitting on newlines if no sentence boundaries are found.

        Args:
            text: Normalized text string.

        Returns:
            A list of sentence strings.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def word_tokenize(self, sentence):
        """Split a single sentence into a list of tokens.

        Args:
            sentence: A single sentence string.

        Returns:
            A list of word strings with no empty tokens.
        """
        return [w for w in sentence.split() if w]

    def save(self, sentences, filepath):
        """Write tokenized sentences to output file.

        Format: one sentence per line, tokens separated by spaces.

        Args:
            sentences: List of lists of tokens (one list per sentence).
            filepath: Path to the output file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for tokens in sentences:
                if tokens:
                    f.write(" ".join(tokens) + "\n")
        logger.info("Saved %d sentences to %s", len(sentences), filepath)

    def process_folder(self, folder_path, output_path):
        """Run the full data prep pipeline on a folder of raw text files.

        Steps: load -> strip_gutenberg -> normalize -> sentence_tokenize
        -> word_tokenize -> save.

        Args:
            folder_path: Path to folder containing raw .txt files.
            output_path: Path to write the tokenized output file.

        Returns:
            A list of lists of tokens (one list per sentence).
        """
        texts = self.load(folder_path)
        all_tokenized = []
        for text in texts:
            text = self.strip_gutenberg(text)
            text = self.normalize(text)
            sentences = self.sentence_tokenize(text)
            for sentence in sentences:
                tokens = self.word_tokenize(sentence)
                if tokens:
                    all_tokenized.append(tokens)
        self.save(all_tokenized, output_path)
        logger.info("Processed %d sentences from %s", len(all_tokenized), folder_path)
        return all_tokenized


def main():
    """Run the data prep module standalone."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", ".env"))

    normalizer = Normalizer()

    train_raw_dir = os.environ.get("TRAIN_RAW_DIR", "data/raw/train/")
    train_tokens = os.environ.get("TRAIN_TOKENS", "data/processed/train_tokens.txt")

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    train_raw_path = os.path.join(base_dir, train_raw_dir)
    train_tokens_path = os.path.join(base_dir, train_tokens)

    normalizer.process_folder(train_raw_path, train_tokens_path)


if __name__ == "__main__":
    main()
