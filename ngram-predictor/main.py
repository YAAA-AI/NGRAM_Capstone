from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    """Load config from config/.env (simple key=value)."""
    cfg = {"SMOOTHING": "none"}
    env_path = ROOT / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    return cfg


def is_running_under_streamlit() -> bool:
    return any("streamlit" in m for m in sys.modules)


def cmd_ui(_args) -> None:
    """UI command.

    Preferred:
        streamlit run main.py -- ui

    Convenience:
        python main.py ui
    """
    if not is_running_under_streamlit():
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "main.py"), "--", "ui"],
            check=True,
        )
        return

    from src.ui.app import run_app
    cfg = load_config()
    run_app(cfg)


def cmd_build(args) -> None:
    from src.model.ngram_model import NGramModel

    book_ids = [int(x) for x in args.book_ids]
    model = NGramModel.from_gutenberg_ids(book_ids, max_n=args.max_n, progress_cb=print)
    print(f"Built model with vocab size: {model.vocab_size}")


def cmd_predict(args) -> None:
    from src.model.ngram_model import NGramModel
    from src.inference.predictor import Predictor

    cfg = load_config()
    book_ids = [int(x) for x in args.book_ids]
    model = NGramModel.from_gutenberg_ids(book_ids, max_n=args.max_n, progress_cb=print)

    smoothing = args.smoothing or cfg.get("SMOOTHING", "none")
    predictor = Predictor(model=model, smoothing=smoothing, top_k=args.top_k)

    print("Type text (Ctrl+C to exit):")
    while True:
        text = input("> ")
        print(predictor.suggest(text))


def cmd_evaluate(args) -> None:
    from src.model.ngram_model import NGramModel
    from src.evaluation.evaluator import Evaluator

    cfg = load_config()
    train_ids = [int(x) for x in args.train_ids]
    model = NGramModel.from_gutenberg_ids(train_ids, max_n=args.max_n, progress_cb=print)

    evaluator = Evaluator(model=model)
    smoothing = args.smoothing or cfg.get("SMOOTHING", "none")
    res = evaluator.evaluate_gutenberg_book(int(args.eval_book_id), smoothing=smoothing)
    print(res)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ngram-predictor",
        description="N-gram predictor (single entry point for CLI + Streamlit UI).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ui = sub.add_parser("ui", help="Run Streamlit UI (use: streamlit run main.py -- ui)")
    p_ui.set_defaults(func=cmd_ui)

    p_build = sub.add_parser("build", help="Build model from Gutenberg book ids.")
    p_build.add_argument("--book-ids", nargs="+", required=True)
    p_build.add_argument("--max-n", type=int, default=4)
    p_build.set_defaults(func=cmd_build)

    p_pred = sub.add_parser("predict", help="Interactive CLI prediction.")
    p_pred.add_argument("--book-ids", nargs="+", required=True)
    p_pred.add_argument("--max-n", type=int, default=4)
    p_pred.add_argument("--top-k", type=int, default=5)
    p_pred.add_argument("--smoothing", type=str, default=None, choices=["none", "mle-backoff"])
    p_pred.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("evaluate", help="Evaluate perplexity on a Gutenberg book id.")
    p_eval.add_argument("--train-ids", nargs="+", required=True)
    p_eval.add_argument("--eval-book-id", required=True)
    p_eval.add_argument("--max-n", type=int, default=4)
    p_eval.add_argument("--smoothing", type=str, default=None, choices=["none", "mle-backoff"])
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
