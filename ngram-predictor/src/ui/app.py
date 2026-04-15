from __future__ import annotations

import streamlit as st
from st_keyup import st_keyup

from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator


class PredictorUI:
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        st.set_page_config(page_title="N-gram Text Predictor", layout="wide")
        st.title("N-gram Text Predictor")

        with st.expander("Settings", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                max_n = st.number_input("N-gram Order", value=4, min_value=2, max_value=10, step=1)
            with c2:
                max_books = st.number_input("Number of Books", value=4, min_value=1, max_value=100, step=1)

            default_ids = [1661, 834, 108, 2852, 11, 84, 98, 1342, 2701, 1080]
            book_ids = []
            max_per_row = 10
            for row_start in range(0, int(max_books), max_per_row):
                row_end = min(row_start + max_per_row, int(max_books))
                cols = st.columns(row_end - row_start)
                for col_idx, idx in enumerate(range(row_start, row_end)):
                    with cols[col_idx]:
                        default_val = default_ids[idx] if idx < len(default_ids) else 1
                        bid = st.number_input(
                            f"Book {idx+1} ID",
                            value=int(default_val),
                            min_value=1,
                            step=1,
                            key=f"book_id_{idx}",
                        )
                        book_ids.append(int(bid))

            load_btn = st.button("Load Books & Build Model")

        progress = st.container()

        if load_btn:
            def progress_cb(msg: str):
                progress.write(msg)

            with st.spinner("Building model..."):
                model = NGramModel.from_gutenberg_ids(book_ids, max_n=int(max_n), progress_cb=progress_cb)
                st.session_state["model"] = model
                st.session_state["predictor"] = Predictor(
                    model=model,
                    smoothing=self.config.get("SMOOTHING", "none"),
                    top_k=5,
                )
                st.session_state["evaluator"] = Evaluator(model=model)

        st.divider()
        self._smoothing_section()
        st.divider()
        self._evaluation_section()
        st.divider()
        self._prediction_section()

    def _smoothing_section(self):
        if "predictor" not in st.session_state:
            st.session_state["smoothing"] = self.config.get("SMOOTHING", "none")
            return

        opts = ["none", "mle-backoff"]
        default = st.session_state.get("smoothing", self.config.get("SMOOTHING", "none"))
        idx = opts.index(default) if default in opts else 0

        smoothing = st.selectbox(
            "Smoothing Method",
            opts,
            index=idx,
            format_func=lambda x: {"none": "None", "mle-backoff": "MLE Backoff"}[x],
        )
        st.session_state["smoothing"] = smoothing
        st.session_state["predictor"].smoothing = smoothing

    @st.fragment
    def _evaluation_section(self):
        st.subheader("Model Evaluation (Perplexity)")

        with st.expander("Evaluation Settings", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                eval_book_id = st.number_input("Evaluation Book ID", value=3289, min_value=1, step=1)
            with c2:
                eval_btn = st.button("Evaluate Perplexity", disabled="evaluator" not in st.session_state)

        if "evaluator" not in st.session_state:
            st.info("Load a model first before evaluating.")
            return

        if eval_btn:
            smoothing = st.session_state.get("smoothing", "none")
            with st.spinner(f"Evaluating with '{smoothing}' smoothing..."):
                try:
                    res = st.session_state["evaluator"].evaluate_gutenberg_book(int(eval_book_id), smoothing=smoothing)
                    st.session_state["eval_results"] = res
                except Exception as e:
                    st.session_state["eval_results"] = {"error": f"{type(e).__name__}: {e}"}

        if "eval_results" in st.session_state:
            r = st.session_state["eval_results"]
            if "error" in r:
                st.error(r["error"])
            else:
                st.success(f"Evaluation complete — Book {r['book_id']}, smoothing: **{r['smoothing_used']}**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Perplexity", f"{r['perplexity']:.2f}")
                m2.metric("Words Evaluated", f"{r['evaluated']:,}")
                m3.metric("Words Skipped", f"{r['skipped']:,}")

    @st.fragment
    def _prediction_section(self):
        st.subheader("Text Prediction")

        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""
        if "widget_key_counter" not in st.session_state:
            st.session_state["widget_key_counter"] = 0

        widget_key = f"user_text_{st.session_state['widget_key_counter']}"
        user_text = st_keyup("Type Here", value=st.session_state["user_text"], debounce=10, key=widget_key)
        if user_text is None:
            user_text = st.session_state["user_text"]
        st.session_state["user_text"] = user_text

        st.subheader("Suggested Next Words")

        if "predictor" not in st.session_state:
            st.info("Load books to start predicting.")
            return

        if not user_text.strip():
            st.info("Start typing to see predictions.")
            return

        preds = st.session_state["predictor"].suggest(user_text)
        if preds:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(preds):
                    if col.button(preds[i], key=f"pred_{i}", use_container_width=True):
                        st.session_state["user_text"] = st.session_state["predictor"].apply_suggestion(user_text, preds[i])
                        st.session_state["widget_key_counter"] += 1
                        st.rerun()
        else:
            st.info("No predictions available for this input.")


def run_app(config: dict):
    PredictorUI(config).run()
