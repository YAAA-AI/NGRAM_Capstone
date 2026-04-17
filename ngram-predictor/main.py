"""N-gram Text Predictor — CLI entry point.

Single entry point for the project. Loads configuration, instantiates all
objects in the correct order, and calls each module in sequence based on
the --step argument.

Usage:
    python main.py --step dataprep    # Run Normalizer — produce train_tokens.txt
    python main.py --step model       # Run NGramModel — produce model.json and vocab.json
    python main.py --step inference   # Start the interactive CLI prediction loop
    python main.py --step evaluate    # Run Evaluator — compute perplexity on eval corpus
    python main.py --step all         # Run dataprep → model → inference in sequence
"""

import argparse
import os
import sys
import logging

# Ensure project root is on sys.path so `from src...` imports work
# regardless of which directory the user runs the script from.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator


def load_config():
    """Load configuration from config/.env file using python-dotenv.

    Reads environment variables from config/.env relative to this script's
    directory. Must be called before any other step runs.

    Returns:
        A dict with all configuration values and their defaults.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", ".env")
    load_dotenv(env_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    config = {
        "TRAIN_RAW_DIR": os.path.join(base_dir, os.environ.get("TRAIN_RAW_DIR", "data/raw/train/")),
        "EVAL_RAW_DIR": os.path.join(base_dir, os.environ.get("EVAL_RAW_DIR", "data/raw/eval/")),
        "TRAIN_TOKENS": os.path.join(base_dir, os.environ.get("TRAIN_TOKENS", "data/processed/train_tokens.txt")),
        "EVAL_TOKENS": os.path.join(base_dir, os.environ.get("EVAL_TOKENS", "data/processed/eval_tokens.txt")),
        "MODEL": os.path.join(base_dir, os.environ.get("MODEL", "data/model/model.json")),
        "VOCAB": os.path.join(base_dir, os.environ.get("VOCAB", "data/model/vocab.json")),
        "UNK_THRESHOLD": int(os.environ.get("UNK_THRESHOLD", "3")),
        "TOP_K": int(os.environ.get("TOP_K", "3")),
        "NGRAM_ORDER": int(os.environ.get("NGRAM_ORDER", "4")),
        "SMOOTHING": os.environ.get("SMOOTHING", "mle-backoff"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    }
    return config


def step_dataprep(config, normalizer):
    """Run the data preparation step.

    Processes raw training text files into tokenized output.
    Also processes evaluation corpus if the eval raw directory exists.

    Args:
        config: Configuration dict.
        normalizer: Normalizer instance.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data preparation...")

    normalizer.process_folder(config["TRAIN_RAW_DIR"], config["TRAIN_TOKENS"])
    logger.info("Training tokens written to %s", config["TRAIN_TOKENS"])

    # Also process evaluation corpus if available
    if os.path.exists(config["EVAL_RAW_DIR"]):
        eval_files = [f for f in os.listdir(config["EVAL_RAW_DIR"]) if f.endswith(".txt")]
        if eval_files:
            normalizer.process_folder(config["EVAL_RAW_DIR"], config["EVAL_TOKENS"])
            logger.info("Evaluation tokens written to %s", config["EVAL_TOKENS"])


def step_model(config):
    """Run the model building step.

    Builds vocabulary, counts n-grams, computes probabilities, and saves
    model.json and vocab.json.

    Args:
        config: Configuration dict.

    Returns:
        The trained NGramModel instance.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building n-gram model (order=%d, unk_threshold=%d)...",
                config["NGRAM_ORDER"], config["UNK_THRESHOLD"])

    model = NGramModel(
        ngram_order=config["NGRAM_ORDER"],
        unk_threshold=config["UNK_THRESHOLD"],
    )
    model.build_counts_and_probabilities(config["TRAIN_TOKENS"])
    model.save_model(config["MODEL"])
    model.save_vocab(config["VOCAB"])

    logger.info("Model built and saved.")
    return model


def step_inference(config, model, normalizer):
    """Run the interactive CLI prediction loop.

    Args:
        config: Configuration dict.
        model: A loaded NGramModel instance.
        normalizer: A Normalizer instance.
    """
    predictor = Predictor(model, normalizer)
    top_k = config["TOP_K"]

    print(f"\nN-gram Predictor ready (order={model.ngram_order}, top_k={top_k}).")
    print("Type a sequence of words to get predictions. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("> ")
            if user_input.strip().lower() == "quit":
                print("Goodbye.")
                break
            predictions = predictor.predict_next(user_input, k=top_k)
            print(f"Predictions: {predictions}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        except ValueError as e:
            print(f"Error: {e}")


def step_evaluate(config, model, normalizer):
    """Run the model evaluation step.

    Args:
        config: Configuration dict.
        model: A loaded NGramModel instance.
        normalizer: A Normalizer instance.
    """
    evaluator = Evaluator(model, normalizer)
    evaluator.run(config["EVAL_TOKENS"])


def main():
    """Parse CLI arguments and run the requested pipeline step."""
    parser = argparse.ArgumentParser(
        description="N-gram Text Predictor — CLI entry point"
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "evaluate", "all"],
        required=True,
        help="Which pipeline step to run: dataprep, model, inference, evaluate, or all",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Path to a custom evaluation corpus (overrides config/.env EVAL_TOKENS)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Configure logging
    logging.basicConfig(
        level=config["LOG_LEVEL"].upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.eval_file:
        config["EVAL_TOKENS"] = args.eval_file

    # Instantiate shared objects
    normalizer = Normalizer()

    try:
        if args.step == "dataprep":
            step_dataprep(config, normalizer)

        elif args.step == "model":
            step_model(config)

        elif args.step == "inference":
            model = NGramModel(ngram_order=config["NGRAM_ORDER"])
            model.load(config["MODEL"], config["VOCAB"])
            step_inference(config, model, normalizer)

        elif args.step == "evaluate":
            model = NGramModel(ngram_order=config["NGRAM_ORDER"])
            model.load(config["MODEL"], config["VOCAB"])
            step_evaluate(config, model, normalizer)

        elif args.step == "all":
            step_dataprep(config, normalizer)
            model = step_model(config)
            step_inference(config, model, normalizer)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Missing config variable: {e}. Check config/.env.")
        sys.exit(1)


if __name__ == "__main__":
    main()

