from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(dotenv_path: str | Path | None = None) -> bool:
        """Fallback dotenv loader when python-dotenv is unavailable."""
        if dotenv_path is None:
            return False

        path = Path(dotenv_path)
        if not path.exists():
            return False

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
        return True


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "processed"
TRAIN_TOKENS_PATH = DATA_DIR / "train_tokens.txt"
MODEL_PATH = DATA_DIR / "model.json"
VOCAB_PATH = DATA_DIR / "vocab.json"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _csv_int_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if not raw:
        return default

    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out or default


def step_dataprep() -> None:
    from src.data_prep.normalizer import Normalizer, prepare_training_tokens

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer()
    train_book_ids = _csv_int_env("TRAIN_BOOK_IDS", [1661, 834, 108, 2852])
    tokens = prepare_training_tokens(TRAIN_TOKENS_PATH, normalizer=normalizer, book_ids=train_book_ids)
    print(f"Dataprep complete. Wrote {len(tokens)} tokens to {TRAIN_TOKENS_PATH}")


def step_model() -> None:
    from src.data_prep.normalizer import load_tokens
    from src.model.ngram_model import NGramModel

    if not TRAIN_TOKENS_PATH.exists():
        raise FileNotFoundError(f"Missing training tokens file: {TRAIN_TOKENS_PATH}")

    tokens = load_tokens(TRAIN_TOKENS_PATH)
    if not tokens:
        raise ValueError(f"No tokens found in: {TRAIN_TOKENS_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    max_n = _int_env("MAX_N", 4)
    model = NGramModel.from_word_lists([tokens], max_n=max_n)
    model.save(MODEL_PATH, VOCAB_PATH)
    print(f"Model complete. Saved model to {MODEL_PATH} and vocab to {VOCAB_PATH}")


def step_inference() -> None:
    from src.data_prep.normalizer import Normalizer
    from src.inference.predictor import Predictor
    from src.model.ngram_model import NGramModel

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    normalizer = Normalizer()
    model = NGramModel.load(MODEL_PATH, VOCAB_PATH, normalizer=normalizer)
    predictor = Predictor(
        model=model,
        smoothing=os.getenv("SMOOTHING", "none"),
        top_k=_int_env("TOP_K", 3),
    )

    print("Interactive inference started. Type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if text.lower() == "quit":
            print("Goodbye.")
            break

        predictions = predictor.predict_next(text, predictor.top_k)
        print(f"Predictions: {predictions}")


def run_step(step: str) -> None:
    if step == "dataprep":
        step_dataprep()
        return
    if step == "model":
        step_model()
        return
    if step == "inference":
        step_inference()
        return
    if step == "all":
        step_dataprep()
        step_model()
        step_inference()
        return

    raise ValueError(f"Unknown step: {step}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="N-gram predictor CLI pipeline")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["dataprep", "model", "inference", "all"],
        help="Which pipeline step to run.",
    )
    return parser


def is_running_under_streamlit() -> bool:
    return any("streamlit" in module_name for module_name in sys.modules) or bool(os.getenv("STREAMLIT_SERVER_PORT"))


def run_streamlit_app() -> None:
    from src.ui.app import run_app

    cfg = {"SMOOTHING": os.getenv("SMOOTHING", "none")}
    run_app(cfg)


def main() -> None:
    load_dotenv(ROOT / "config" / ".env")

    if is_running_under_streamlit():
        run_streamlit_app()
        return

    args = build_parser().parse_args()
    run_step(args.step)

if __name__ == "__main__":
    main()
