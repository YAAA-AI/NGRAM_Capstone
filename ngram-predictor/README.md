# N-gram Text Predictor

A smart text prediction application that learns from Project Gutenberg books to suggest the next word as you type, with built-in model evaluation via perplexity.

## Overview

This application uses n-gram language modeling to predict the next word based on what you've typed. It downloads books from Project Gutenberg, builds statistical models with a configurable n-gram order (2 to 10), and provides real-time word suggestions. It also includes a perplexity evaluator to measure model quality against a held-out book.

## Features

- **Smart Word Prediction**: Suggests 5 most likely next words based on your input
- **Real-time Updates**: Predictions update live as you type (no Enter key needed)
- **Click-to-Insert**: Click any suggestion button to insert the word
- **Configurable N-gram Order**: Set n-gram order from 2 to 10 (default: 4)
- **Smoothing**: Choose from None (MLE without unigram) or MLE Backoff with Unigram Fallback — configurable from the GUI and `config/.env`
- **Custom Book Selection**: Choose up to 100 specific Gutenberg books by ID
- **Dynamic UI**: Book ID inputs arranged in rows of 10, adapting to the number selected
- **Model Evaluation**: Compute perplexity against any Gutenberg book to measure model quality
- **Non-blocking Evaluation**: Evaluation runs as an isolated fragment — typing and predictions remain functional during and after evaluation
- **Persistent Results**: Evaluation results persist across reruns until a new evaluation is triggered
- **Smart Spacing**: Automatically handles spacing when inserting predicted words

## Python Version
If using Python 3.14, NumPy may not build correctly with MinGW. If the app crashes after install, run:

```bash
pip uninstall numpy -y
pip install --only-binary=:all: numpy
```

## Requirements

- Python 3.14+
- See `requirements.txt` for the full dependency list.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the app:

```bash
streamlit run main.py
```

### Step 1: Configure and Load Books

1. Set **N-gram Order** (2–10, default: 4)
2. Set **Number of Books** (1–100)
3. Enter specific **Gutenberg Book IDs** in the input fields (defaults: 1661, 834, 108, 2852)
4. Click **"Load Books & Build Model"** — downloads books and builds the n-gram model in one step
5. Wait while the app downloads, processes, and builds

### Step 2: Select Smoothing

1. After the model is built, a **Smoothing Method** dropdown appears
2. Choose **None** (pure MLE backoff) or **MLE Backoff** (with unigram fallback)
3. Smoothing applies to both predictions and evaluation — change it anytime without rebuilding

### Step 3: Start Typing

1. Click in the **"Type Here"** text box
2. Start typing — predictions update live as you type (no Enter needed)

### Step 4: Use Predictions

1. As you type, 5 buttons below show suggested next words
2. With MLE Backoff, if fewer than 5 n-gram matches exist, remaining slots are filled with top unigram words
3. Click any button to insert that word into your text
4. The app automatically handles spacing
5. Continue typing or clicking suggestions to build your text

### Step 5: Evaluate the Model (Optional)

1. Expand the **Model Evaluation (Perplexity)** section
2. Enter an **Evaluation Book ID** (default: 3289 — The Valley of Fear)
3. Click **"Evaluate Perplexity"**
4. View results: Perplexity, Words Evaluated, and Words Skipped
5. Evaluation runs in an isolated fragment — you can continue typing immediately after

## Tips for Best Results

- **More Books = Better Predictions**: Loading 20–50 books gives more accurate predictions than 4
- **Type Multiple Words**: The app works best when you've typed at least 2–3 words
- **Higher N-gram Order**: Larger n-grams capture more context but require more training data
- **Genre Matters**: Books with similar themes give better predictions for related topics
- **Lower Perplexity = Better Model**: A lower perplexity score means the model predicts the evaluation text more accurately

## Understanding the Interface

### Settings Panel
- **N-gram Order**: Maximum n-gram size to build (2–10, default: 4)
- **Number of Books**: How many books to load (1–100)
- **Book ID Inputs**: Specific Gutenberg IDs for each book, arranged in rows of 10
- **Load Books & Build Model**: Downloads books and builds the model in one step

### Smoothing Method (appears after model is built)
- **None**: Pure MLE backoff from n-gram down to 2-gram; returns empty if no context matches
- **MLE Backoff**: Falls back to unigram probabilities when no higher-order context is found; also fills remaining suggestion slots (up to 5) with top unigram words
- Change anytime without rebuilding the model — applies to both predictions and evaluation

### Progress Area
- Shows real-time status of book downloading and model building
- Displays n-gram statistics

### Model Evaluation (Fragment)
- Runs as an isolated `@st.fragment` — does not interfere with text prediction
- **Evaluation Book ID**: Gutenberg ID of the book to evaluate against (default: 3289)
- **Evaluate Perplexity**: Compute and display perplexity, words evaluated, and words skipped
- Results persist in session state until a new evaluation is run
- Perplexity = $2^H$ where H is cross-entropy over the evaluation corpus

### Text Input (Fragment)
- Main typing area where you compose text
- Predictions update live as you type (no Enter needed)
- Runs as an isolated `@st.fragment` — typing does not trigger full page reruns

### Suggested Next Words
- 5 buttons showing most likely next words
- With MLE Backoff, unfilled slots are supplemented with top unigram words
- Click any button to insert that word

## Technical Details

### How It Works

1. **Book Selection**: You choose specific Gutenberg book IDs to use as training data
2. **Text Extraction**: Downloads and extracts plain text from books using BeautifulSoup
3. **N-gram Building**: Creates statistical models from 2-grams up to the configured n-gram order
4. **Context Indexing**: Builds fast lookup dictionaries (converted from defaultdict to dict) for instant predictions
5. **Prediction**: Uses backoff — tries the highest n-gram context first, then falls back to lower orders. With MLE Backoff, supplements partial results with unigram probabilities to always show up to 5 suggestions
6. **Evaluation**: Computes perplexity on a held-out book using the same backoff strategy

### N-gram Sizes (default order = 4)

- **4-grams**: Most specific (uses your last 3 words to predict the next)
- **3-grams**: Falls back to last 2 words if 4-gram not found
- **2-grams**: Falls back to last 1 word if 3-gram not found

### Perplexity

- Measures how well the model predicts unseen text
- Lower is better
- Words with zero probability at all n-gram orders are skipped

## CLI Usage

`main.py` accepts a `--step` argument that controls which part of the pipeline to run. Steps must be run in order (`dataprep` → `model` → `inference`) unless you use `all`.

| Command | What it runs |
|---|---|
| `python main.py --step dataprep` | Run `Normalizer` — produce `train_tokens.txt` |
| `python main.py --step model` | Run `NGramModel` — produce `model.json` and `vocab.json` |
| `python main.py --step inference` | Start the interactive CLI prediction loop |
| `python main.py --step all` | Run dataprep → model → inference in sequence |

Environment variables are loaded from `config/.env` before any step runs.

| Variable | Description | Default |
|---|---|---|
| `SMOOTHING` | `none` or `mle-backoff` | `none` |
| `TOP_K` | Predictions shown per prompt in CLI inference | `3` |
| `MAX_N` | Max n-gram order for the model step | `4` |
| `TRAIN_BOOK_IDS` | Comma-separated Gutenberg IDs for dataprep | `1661,834,108,2852` |

`main.py` is also the Streamlit entrypoint — `streamlit run main.py` launches the full UI without requiring `--step`.

## Default Book IDs

| ID | Title |
|----|-------|
| 1661 | The Adventures of Sherlock Holmes |
| 834 | The Count of Monte Cristo |
| 108 | A Tale of Two Cities |
| 2852 | The Hound of the Baskervilles |

## Troubleshooting

### "No predictions available"
- You may need to type more words (at least 2–3)
- The books loaded might not contain relevant vocabulary
- Try loading more books

### Application is slow to start
- First-time model building takes time (normal)
- More books = longer loading time

### Books not found
- Some Project Gutenberg IDs don't have HTML versions
- The app automatically skips invalid IDs
- Verify the ID exists at `https://www.gutenberg.org/ebooks/<ID>`
