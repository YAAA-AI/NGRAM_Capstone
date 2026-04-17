"""UI entrypoint for ngram-predictor.

Implements a Streamlit-based browser interface for the n-gram text predictor.
Runs alongside the CLI — does not replace it.

Launch from anywhere with:
    python -m streamlit run <path-to>/ngram-predictor/src/ui/app.py
"""

import os
import sys
import ssl
import logging
from urllib.request import urlopen
from time import sleep

import urllib3
import streamlit as st
from st_keyup import st_keyup
from bs4 import BeautifulSoup

# Add project root to path so module imports work when run via streamlit
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class PredictorUI:
    """Streamlit-based user interface for n-gram prediction.

    Accepts a Predictor instance via the constructor and provides
    a browser-based interface for typing text and seeing predictions.
    """

    def __init__(self, predictor):
        """Accept a Predictor instance.

        Args:
            predictor: A Predictor instance with model and normalizer loaded.
        """
        self.predictor = predictor

    def get_predictions(self, text, k=5):
        """Get predictions for the given text.

        Args:
            text: The user's input text string.
            k: Number of top predictions to return.

        Returns:
            A list of predicted next word strings. Returns empty list
            if input is empty or no predictions found.
        """
        if not text or not text.strip():
            return []
        try:
            return self.predictor.predict_next(text, k=k)
        except ValueError:
            return []

    def run(self):
        """Start the Streamlit UI (placeholder for dependency-injected use)."""
        print("PredictorUI: use 'python -m streamlit run src/ui/app.py' to launch.")


# ---------------------------------------------------------------------------
# Helper functions for the Streamlit app
# ---------------------------------------------------------------------------

def _create_ssl_context():
    """Create an SSL context that skips certificate verification."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _load_config():
    """Load configuration from config/.env file."""
    config = {"SMOOTHING": "mle-backoff"}
    env_path = os.path.join(_PROJECT_ROOT, "config", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    return config


def _extract_words_from_html_links(links, ssl_context, normalizer,
                                   max_retries=3, progress_placeholder=None):
    """Download and extract words from a list of HTML book URLs.

    Args:
        links: List of URLs to download.
        ssl_context: SSL context for HTTPS requests.
        normalizer: A Normalizer instance for consistent text normalization.
        max_retries: Number of download retries per book.
        progress_placeholder: Optional Streamlit container for progress.

    Returns:
        A list of word lists (one per book).
    """
    result = []
    for link_index, link in enumerate(links, 1):
        words = None
        for attempt in range(1, max_retries + 1):
            try:
                if progress_placeholder:
                    progress_placeholder.write(
                        f"Processing book {link_index}/{len(links)}...")
                html = urlopen(link, context=ssl_context).read()
                soup = BeautifulSoup(html, 'html.parser')
                words = normalizer.normalize(soup.get_text()).split()
                break
            except Exception as e:
                if attempt < max_retries:
                    if progress_placeholder:
                        progress_placeholder.write(
                            f"  Book {link_index}: Attempt {attempt}/{max_retries} failed, retrying...")
                    sleep(0.1 * attempt)
                else:
                    if progress_placeholder:
                        progress_placeholder.write(
                            f"Book {link_index}: Failed after {max_retries} attempts - {type(e).__name__}")
        result.append(words if words is not None else [])
    return result


# ---------------------------------------------------------------------------
# Streamlit App (runs when this file is executed via `streamlit run`)
# ---------------------------------------------------------------------------

def run_app():
    """Main Streamlit application — no global variables."""
    st.set_page_config(page_title="N-gram Text Predictor", layout="wide")
    st.title("N-gram Text Predictor")

    config = _load_config()
    normalizer = Normalizer()

    # Settings
    with st.expander("Settings", expanded=True):
        settings_col1, settings_col2 = st.columns([1, 1])
        with settings_col1:
            max_n = st.number_input("N-gram Order", value=4, min_value=2,
                                    max_value=10, step=1, key="setting_max_n")
        with settings_col2:
            max_books = st.number_input("Number of Books", value=4, min_value=1,
                                        max_value=100, step=1, key="setting_max_books")

        book_ids = []
        default_ids = [1661, 834, 108, 2852, 11, 84, 98, 1342, 2701, 1080]
        max_per_row = 10
        for row_start in range(0, max_books, max_per_row):
            row_end = min(row_start + max_per_row, max_books)
            row_cols = st.columns(row_end - row_start)
            for col_idx, idx in enumerate(range(row_start, row_end)):
                with row_cols[col_idx]:
                    default_val = default_ids[idx] if idx < len(default_ids) else 1
                    bid = st.number_input(f"Book {idx + 1} ID", value=default_val,
                                          min_value=1, step=1, key=f"book_id_{idx}")
                    book_ids.append(bid)

        load_btn = st.button("Load Books & Build Model")

    # Progress area
    progress_placeholder = st.container()

    # Load books and build model
    if load_btn:
        with st.spinner("Downloading books..."):
            html_links = [
                f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}-images.html"
                for bid in book_ids
            ]
            progress_placeholder.write(f"Loading {len(book_ids)} books: {book_ids}")
            word_lists = _extract_words_from_html_links(
                html_links, ssl_context=_create_ssl_context(),
                normalizer=normalizer,
                progress_placeholder=progress_placeholder,
            )
            st.session_state["word_lists"] = word_lists
            st.session_state.pop("eval_results", None)
            total_words = sum(len(wl) for wl in word_lists)
            progress_placeholder.write(
                f"Books loaded! {total_words:,} total words from {len(word_lists)} books.")

        with st.spinner("Building model..."):
            st.session_state["model"] = NGramModel.from_word_lists(
                st.session_state["word_lists"], max_n=max_n,
                progress_placeholder=progress_placeholder,
            )

    # Smoothing (shown after model is built)
    if "model" in st.session_state:
        smoothing_options = ["mle-backoff", "katz"]
        default_idx = (smoothing_options.index(config["SMOOTHING"])
                       if config.get("SMOOTHING") in smoothing_options else 0)
        smoothing = st.selectbox(
            "Smoothing Method", smoothing_options, index=default_idx,
            format_func=lambda x: {"mle-backoff": "None", "katz": "Katz Backoff"}[x],
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
                eval_book_id = st.number_input("Evaluation Book ID", value=3289,
                                               min_value=1, step=1,
                                               key="eval_book_id")
            with eval_col2:
                eval_btn = st.button("Evaluate Perplexity",
                                     disabled="model" not in st.session_state)

        if "model" not in st.session_state:
            st.info("Load a model first before evaluating.")

        if eval_btn and "model" in st.session_state:
            eval_url = (f"https://www.gutenberg.org/cache/epub/{eval_book_id}"
                        f"/pg{eval_book_id}-images.html")
            cur_smoothing = st.session_state.get("smoothing", "mle-backoff")
            use_fallback = cur_smoothing == "mle-backoff"
            with st.spinner(f"Evaluating with '{cur_smoothing}' smoothing on book {eval_book_id}..."):
                try:
                    html = urlopen(eval_url, context=_create_ssl_context()).read()
                    soup = BeautifulSoup(html, 'html.parser')
                    eval_words = normalizer.normalize(soup.get_text()).split()
                    if not eval_words:
                        st.session_state["eval_results"] = {"error": "Evaluation book is empty."}
                    else:
                        perplexity, evaluated, skipped = st.session_state["model"].evaluate_words(
                            eval_words, use_unigram_fallback=use_fallback,
                            smoothing=cur_smoothing)
                        st.session_state["eval_results"] = {
                            "perplexity": perplexity, "evaluated": evaluated,
                            "skipped": skipped, "smoothing_used": cur_smoothing,
                        }
                except Exception as e:
                    st.session_state["eval_results"] = {"error": f"{type(e).__name__}: {e}"}

        if "eval_results" in st.session_state:
            r = st.session_state["eval_results"]
            if "error" in r:
                st.error(r["error"])
            else:
                method_label = {
                    "mle-backoff": "None (MLE Backoff with Unigram Fallback)",
                    "katz": "Katz Backoff (Good-Turing Discounting)",
                }.get(r.get("smoothing_used", "mle-backoff"), r.get("smoothing_used", "?"))
                st.success(f"Evaluation complete — Smoothing: **{method_label}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Perplexity", f"{r['perplexity']:.2f}")
                m2.metric("Words Evaluated", f"{r['evaluated']:,}")
                m3.metric("Words Skipped (zero probability)", f"{r['skipped']:,}")

    evaluation_section()

    # --- Text Prediction ---
    st.divider()

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

        st.subheader("Suggested Next Words")

        if "model" in st.session_state and user_text.strip():
            cur_smoothing = st.session_state.get("smoothing", "mle-backoff")
            use_fallback = cur_smoothing == "mle-backoff"
            words = normalizer.normalize(user_text).split()
            predictions = st.session_state["model"].predict_from_words(
                words, top_k=5, use_unigram_fallback=use_fallback,
                smoothing=cur_smoothing,
            )
            if predictions:
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(predictions):
                        if col.button(predictions[i], key=f"pred_{i}",
                                      use_container_width=True):
                            st.session_state["user_text"] = (
                                user_text.rstrip() + " " + predictions[i] + " ")
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


run_app()
