# %%
import re
import os
import math
from urllib.request import urlopen
import ssl
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import requests
from time import sleep
import urllib3
import streamlit as st
from st_keyup import st_keyup

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_ssl_context():
    """Create an SSL context that skips certificate verification.

    Returns:
        An ssl.SSLContext with hostname checking and certificate
        verification disabled.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def load_config():
    """Load configuration from config/.env file.

    Reads key=value pairs from config/.env relative to this script's
    directory. Lines starting with '#' and blank lines are ignored.

    Returns:
        A dict with configuration values. Defaults to SMOOTHING='mle-backoff'.
    """
    config = {"SMOOTHING": "mle-backoff"}
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    return config


def normalize_text(text):
    """Convert text to lowercase, remove punctuation, and split into words.

    Args:
        text: Raw input text string.

    Returns:
        A list of lowercase word strings.
    """
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def discover_gutenberg_books(start_id, max_books, max_consecutive_failures=20, progress_placeholder=None):
    """Discover available books from Project Gutenberg by scanning sequential IDs.

    Args:
        start_id: The Gutenberg book ID to begin scanning from.
        max_books: Maximum number of valid books to find before stopping.
        max_consecutive_failures: Number of consecutive missing IDs before giving up.
        progress_placeholder: Optional Streamlit container to display progress messages.

    Returns:
        A list of valid HTML URLs for discovered Gutenberg books.
    """
    html_links = []
    n = start_id
    consecutive_failures = 0

    if progress_placeholder:
        progress_placeholder.write(f"Discovering books starting from ID {start_id}...")

    while len(html_links) < max_books and consecutive_failures < max_consecutive_failures:
        url = f"https://www.gutenberg.org/cache/epub/{n}/pg{n}-images.html"

        try:
            response = requests.head(url, timeout=5, allow_redirects=True, verify=False)

            if response.status_code == 200:
                html_links.append(url)
                consecutive_failures = 0
                if progress_placeholder:
                    progress_placeholder.write(f"Found book {n} (total: {len(html_links)})")
            else:
                consecutive_failures += 1

        except requests.RequestException:
            consecutive_failures += 1

        n += 1
        sleep(0.05)

    if progress_placeholder:
        progress_placeholder.write(f"Discovery complete! Found {len(html_links)} valid books")
    return html_links


def extract_words_from_html_links(links, ssl_context, max_retries=3, progress_placeholder=None):
    """Download and extract words from a list of HTML book URLs.

    Args:
        links: List of URLs pointing to HTML versions of Gutenberg books.
        ssl_context: SSL context for HTTPS connections.
        max_retries: Maximum number of download attempts per book.
        progress_placeholder: Optional Streamlit container to display progress messages.

    Returns:
        A list of word lists, one per book. Failed downloads yield empty lists.
    """
    result = []

    for link_index, link in enumerate(links, 1):
        words = None

        for attempt in range(1, max_retries + 1):
            try:
                if progress_placeholder:
                    progress_placeholder.write(f"Processing book {link_index}/{len(links)}...")
                html = urlopen(link, context=ssl_context).read()
                soup = BeautifulSoup(html, 'html.parser')
                words = normalize_text(soup.get_text())
                break
            except Exception as e:
                if attempt < max_retries:
                    if progress_placeholder:
                        progress_placeholder.write(f"  Book {link_index}: Attempt {attempt}/{max_retries} failed, retrying...")
                    sleep(0.1 * attempt)
                else:
                    if progress_placeholder:
                        progress_placeholder.write(f"Book {link_index}: Failed after {max_retries} attempts - {type(e).__name__}")

        result.append(words if words is not None else [])

    return result


def build_ngrams(words, n):
    """Build a list of n-gram tuples from a word list.

    Args:
        words: List of words to create n-grams from.
        n: The size of each n-gram.

    Returns:
        A list of n-gram tuples.
    """
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def count_ngrams(word_lists, n):
    """Count n-gram frequencies across multiple word lists.

    Args:
        word_lists: A list of word lists (one per book).
        n: The size of each n-gram.

    Returns:
        A Counter mapping each n-gram tuple to its total frequency.
    """
    all_ngrams = []
    for words in word_lists:
        all_ngrams.extend(build_ngrams(words, n))
    return Counter(all_ngrams)


def build_context_index(ngram_counts):
    """Build a lookup index mapping (n-1)-word contexts to candidate next words.

    Args:
        ngram_counts: A Counter of n-gram tuples and their frequencies.

    Returns:
        A dict mapping context tuples to lists of (word, count) pairs sorted by
        frequency in descending order.
    """
    context_index = defaultdict(list)
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        context_index[context].append((ngram[-1], count))
    for context in context_index:
        context_index[context].sort(key=lambda x: x[1], reverse=True)
    return dict(context_index)


class NgramModel:
    """N-gram language model with configurable MLE backoff.

    Builds context indices from n-gram down to 2-gram, plus unigram
    probabilities. At inference time, backs off from the highest order
    to lower orders, with an optional unigram fallback.

    Attributes:
        context_indices: List of context index dicts, ordered from highest
            to lowest n-gram order.
        unigram_probs: Dict mapping each word to its unigram probability.
        vocab_size: Number of unique words in the training vocabulary.
    """

    def __init__(self, context_indices, unigram_probs, vocab_size):
        """Initialize the model with precomputed data structures.

        Args:
            context_indices: List of context index dicts (highest order first).
            unigram_probs: Dict of word to unigram probability.
            vocab_size: Vocabulary size.
        """
        self.context_indices = context_indices
        self.unigram_probs = unigram_probs
        self.vocab_size = vocab_size

    @classmethod
    def from_word_lists(cls, word_lists, max_n=4, progress_placeholder=None):
        """Build a model from already-extracted word lists.

        Args:
            word_lists: List of word lists (one per book).
            max_n: Maximum n-gram order (default: 4).
            progress_placeholder: Optional Streamlit container for progress.

        Returns:
            An NgramModel instance.
        """
        if progress_placeholder:
            progress_placeholder.write("Building n-gram counts...")

        context_indices = []
        for n in range(max_n, 1, -1):
            ngram_counts = count_ngrams(word_lists, n=n)
            if progress_placeholder:
                progress_placeholder.write(f"Built {len(ngram_counts)} unique {n}-grams")
            context_indices.append(build_context_index(ngram_counts))

        unigram_counts = count_ngrams(word_lists, n=1)
        if progress_placeholder:
            progress_placeholder.write(f"Built {len(unigram_counts)} unique 1-grams")
        total_unigrams = sum(unigram_counts.values())
        unigram_probs = {word[0]: count / total_unigrams
                         for word, count in unigram_counts.items()}

        if progress_placeholder:
            progress_placeholder.write("Model ready! Start typing to see predictions.")

        return cls(context_indices, unigram_probs, len(unigram_probs))

    @classmethod
    def build(cls, html_links, max_n=4, ssl_context=None, progress_placeholder=None):
        """Build a model from Project Gutenberg book URLs.

        Downloads books, extracts text, builds n-gram counts from max_n
        down to 2, and computes unigram probabilities.

        Args:
            html_links: List of URLs pointing to HTML book pages.
            max_n: Maximum n-gram order (default: 4).
            ssl_context: SSL context for HTTPS connections.
            progress_placeholder: Optional Streamlit container for progress.

        Returns:
            An NgramModel instance.
        """
        if progress_placeholder:
            progress_placeholder.write("Extracting text from books...")
        word_lists = extract_words_from_html_links(
            html_links, ssl_context=ssl_context, progress_placeholder=progress_placeholder
        )

        return cls.from_word_lists(word_lists, max_n=max_n, progress_placeholder=progress_placeholder)

    def predict(self, user_input, top_k=5, use_unigram_fallback=False, smoothing="mle-backoff"):
        """Predict the most likely next words given user input text.

        Uses backoff from highest to lowest n-gram order. If no context
        matches and use_unigram_fallback is True, returns the most
        frequent unigrams.

        Args:
            user_input: The text typed by the user.
            top_k: Number of top predictions to return.
            use_unigram_fallback: Whether to fall back to unigram
                probabilities when no n-gram context matches.
            smoothing: Smoothing method ('mle-backoff' or 'katz').

        Returns:
            A list of up to top_k predicted next words.
        """
        num = len(self.context_indices)
        words = normalize_text(user_input)[-num:]
        if not words:
            return []

        if smoothing == "katz":
            return self._predict_katz(words, top_k)

        for j, context_index in enumerate(self.context_indices):
            context_len = min(len(words), num - j)
            context = tuple(words[-context_len:]) if context_len > 0 else ()
            if context in context_index:
                results = [word for word, count in context_index[context][:top_k]]
                if use_unigram_fallback and len(results) < top_k:
                    seen = set(results)
                    ranked = sorted(self.unigram_probs, key=self.unigram_probs.get, reverse=True)
                    for w in ranked:
                        if w not in seen:
                            results.append(w)
                            seen.add(w)
                            if len(results) >= top_k:
                                break
                return results

        if use_unigram_fallback:
            ranked = sorted(self.unigram_probs, key=self.unigram_probs.get, reverse=True)
            return ranked[:top_k]

        return []

    def evaluate(self, eval_words, use_unigram_fallback=False, smoothing="mle-backoff"):
        """Compute perplexity of the model on an evaluation word list.

        Pre-builds O(1) lookup tables from context indices, then scores
        each word in the corpus using the backoff strategy.

        Args:
            eval_words: List of words from the evaluation corpus.
            use_unigram_fallback: Whether to use unigram probabilities
                for words not found in any higher-order context.
            smoothing: Smoothing method ('mle-backoff' or 'katz').

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
                table[ctx] = {w: math.log2(c) - log2_total for w, c in candidates}
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

    def _build_katz_tables(self, k=5):
        """Precompute Katz backoff discount probabilities and alpha weights.

        Uses Good-Turing discounting for counts <= k and no discounting
        for counts > k. Builds tables from lowest to highest n-gram order
        so that backoff probabilities are available during construction.

        Args:
            k: Discount threshold. Counts above k are not discounted.
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
                    alphas[ctx] = max(0.0, (1.0 - seen_mass) / (1.0 - backoff_seen_mass))
                else:
                    alphas[ctx] = 0.0

            self._katz_disc_probs[idx] = disc_probs
            self._katz_alphas[idx] = alphas

    def _katz_lower_prob(self, word, context, order_idx):
        """Compute Katz backoff probability recursively.

        Args:
            word: The target word.
            context: Tuple of context words.
            order_idx: Index into self.context_indices to start from.

        Returns:
            The Katz backoff probability of word given context.
        """
        num = len(self.context_indices)
        if order_idx >= num:
            return self.unigram_probs.get(word, 1.0 / (self.vocab_size + 1))

        disc_probs = self._katz_disc_probs[order_idx]
        if disc_probs is not None and context in disc_probs:
            if word in disc_probs[context]:
                return disc_probs[context][word]
            else:
                alpha = self._katz_alphas[order_idx].get(context, 0.0)
                shorter = context[1:] if len(context) > 0 else ()
                return alpha * self._katz_lower_prob(word, shorter, order_idx + 1)
        else:
            shorter = context[1:] if len(context) > 0 else ()
            return self._katz_lower_prob(word, shorter, order_idx + 1)

    def _predict_katz(self, words, top_k=5):
        """Predict next words using Katz backoff smoothing.

        Args:
            words: Normalized input words (already trimmed to context size).
            top_k: Number of top predictions to return.

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
                results = sorted(seen.keys(), key=seen.get, reverse=True)[:top_k]
                if len(results) < top_k:
                    alpha = self._katz_alphas[j].get(context, 0.0)
                    if alpha > 0:
                        seen_set = set(results)
                        ranked = sorted(self.unigram_probs, key=self.unigram_probs.get, reverse=True)
                        for w in ranked:
                            if w not in seen_set:
                                results.append(w)
                                seen_set.add(w)
                                if len(results) >= top_k:
                                    break
                return results

        ranked = sorted(self.unigram_probs, key=self.unigram_probs.get, reverse=True)
        return ranked[:top_k]

    def _evaluate_katz(self, eval_words):
        """Compute perplexity using Katz backoff smoothing.

        Args:
            eval_words: List of words from the evaluation corpus.

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


# --- Streamlit App ---

st.set_page_config(page_title="N-gram Text Predictor", layout="wide")
st.title("N-gram Text Predictor")

config = load_config()

# Settings
with st.expander("Settings", expanded=True):
    settings_col1, settings_col2 = st.columns([1, 1])
    with settings_col1:
        max_n = st.number_input("N-gram Order", value=4, min_value=2, max_value=10, step=1)
    with settings_col2:
        max_books = st.number_input("Number of Books", value=4, min_value=1, max_value=100, step=1)

    book_ids = []
    default_ids = [1661, 834, 108, 2852, 11, 84, 98, 1342, 2701, 1080]
    max_per_row = 10
    for row_start in range(0, max_books, max_per_row):
        row_end = min(row_start + max_per_row, max_books)
        row_cols = st.columns(row_end - row_start)
        for col_idx, idx in enumerate(range(row_start, row_end)):
            with row_cols[col_idx]:
                default_val = default_ids[idx] if idx < len(default_ids) else 1
                bid = st.number_input(f"Book {idx + 1} ID", value=default_val, min_value=1, step=1, key=f"book_id_{idx}")
                book_ids.append(bid)

    load_btn = st.button("Load Books & Build Model")

# Progress area
progress_placeholder = st.container()

# Load books and build model
if load_btn:
    with st.spinner("Downloading books..."):
        html_links = [f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}-images.html" for bid in book_ids]
        progress_placeholder.write(f"Loading {len(book_ids)} books: {book_ids}")
        word_lists = extract_words_from_html_links(
            html_links, ssl_context=create_ssl_context(),
            progress_placeholder=progress_placeholder,
        )
        st.session_state["word_lists"] = word_lists
        st.session_state.pop("eval_results", None)
        total_words = sum(len(wl) for wl in word_lists)
        progress_placeholder.write(f"Books loaded! {total_words:,} total words from {len(word_lists)} books.")

    with st.spinner("Building model..."):
        st.session_state["model"] = NgramModel.from_word_lists(
            st.session_state["word_lists"], max_n=max_n,
            progress_placeholder=progress_placeholder,
        )

# Smoothing (shown after model is built)
if "model" in st.session_state:
    smoothing_options = ["mle-backoff", "katz"]
    default_idx = smoothing_options.index(config["SMOOTHING"]) if config["SMOOTHING"] in smoothing_options else 0
    smoothing = st.selectbox("Smoothing Method", smoothing_options, index=default_idx,
                             format_func=lambda x: {"mle-backoff": "None",
                                                     "katz": "Katz Backoff"}[x],
                             help="Applied to both evaluation and predictions")
    st.session_state["smoothing"] = smoothing
else:
    st.session_state["smoothing"] = config.get("SMOOTHING", "mle-backoff")

# --- Evaluator ---
st.divider()

@st.fragment
def evaluation_section():
    """Render the model evaluation UI as an isolated Streamlit fragment."""
    st.subheader("Model Evaluation (Perplexity)")

    with st.expander("Evaluation Settings", expanded=True):
        eval_col1, eval_col2 = st.columns([1, 1])
        with eval_col1:
            eval_book_id = st.number_input("Evaluation Book ID", value=3289, min_value=1, step=1)
        with eval_col2:
            eval_btn = st.button("Evaluate Perplexity", disabled="model" not in st.session_state)

    if "model" not in st.session_state:
        st.info("Load a model first before evaluating.")

    if eval_btn and "model" in st.session_state:
        eval_url = f"https://www.gutenberg.org/cache/epub/{eval_book_id}/pg{eval_book_id}-images.html"
        cur_smoothing = st.session_state.get("smoothing", "mle-backoff")
        use_fallback = cur_smoothing == "mle-backoff"
        with st.spinner(f"Evaluating with '{cur_smoothing}' smoothing on book {eval_book_id}..."):
            try:
                html = urlopen(eval_url, context=create_ssl_context()).read()
                soup = BeautifulSoup(html, 'html.parser')
                eval_words = normalize_text(soup.get_text())

                if not eval_words:
                    st.session_state["eval_results"] = {"error": "Evaluation book is empty."}
                else:
                    perplexity, evaluated, skipped = st.session_state["model"].evaluate(
                        eval_words, use_unigram_fallback=use_fallback, smoothing=cur_smoothing,
                    )
                    st.session_state["eval_results"] = {
                        "perplexity": perplexity,
                        "evaluated": evaluated,
                        "skipped": skipped,
                        "smoothing_used": cur_smoothing,
                    }
            except Exception as e:
                st.session_state["eval_results"] = {"error": f"{type(e).__name__}: {e}"}

    if "eval_results" in st.session_state:
        r = st.session_state["eval_results"]
        if "error" in r:
            st.error(r["error"])
        else:
            method_label = {"mle-backoff": "None (MLE Backoff with Unigram Fallback)",
                            "katz": "Katz Backoff (Good-Turing Discounting)"}.get(
                                r.get("smoothing_used", "mle-backoff"), r.get("smoothing_used", "?"))
            st.success(f"Evaluation complete — Smoothing: **{method_label}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Perplexity", f"{r['perplexity']:.2f}")
            m2.metric("Words Evaluated", f"{r['evaluated']:,}")
            m3.metric("Words Skipped (zero probability)", f"{r['skipped']:,}")

evaluation_section()

# --- Text Prediction ---
st.divider()

# Initialize text in session state
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""
if "widget_key_counter" not in st.session_state:
    st.session_state["widget_key_counter"] = 0

@st.fragment
def prediction_section():
    """Render the text input and prediction buttons as an isolated fragment."""
    widget_key = f"user_text_{st.session_state['widget_key_counter']}"
    user_text = st_keyup(
        "Type Here", value=st.session_state["user_text"],
        debounce=10, key=widget_key,
    )
    if user_text is None:
        user_text = st.session_state.get("user_text", "")
    st.session_state["user_text"] = user_text

    # Predictions
    st.subheader("Suggested Next Words")

    if "model" in st.session_state and user_text.strip():
        cur_smoothing = st.session_state.get("smoothing", "mle-backoff")
        use_fallback = cur_smoothing == "mle-backoff"
        predictions = st.session_state["model"].predict(
            user_text, top_k=5, use_unigram_fallback=use_fallback, smoothing=cur_smoothing,
        )

        if predictions:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(predictions):
                    if col.button(predictions[i], key=f"pred_{i}", use_container_width=True):
                        st.session_state["user_text"] = user_text.rstrip() + " " + predictions[i] + " "
                        st.session_state["widget_key_counter"] += 1
                        st.rerun()
        else:
            st.info("No predictions available for this input.")
    elif "model" not in st.session_state:
        if "word_lists" in st.session_state:
            st.info("Build a model to start predicting.")
        else:
            st.info("Load books to start predicting.")
    else:
        st.info("Start typing to see predictions.")

prediction_section()

# %%

