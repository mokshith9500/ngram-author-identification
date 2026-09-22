"""
Step 1 — Build and Experiment with a BPE Tokenizer

Trains one shared Byte-Pair-Encoding tokenizer on the combined text of
The Hobbit (Tolkien) and The Lost World (Doyle), at four vocabulary
sizes (500 / 1000 / 2000 / 5000), and runs sanity checks on the result.

No n-grams, probabilities, smoothing, perplexity, or author
classification happen here — that's later steps.
"""

import re
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
HOBBIT_PATH = DATA_DIR / "hobbit.txt"
LOST_WORLD_PATH = DATA_DIR / "lost_world.txt"

TOKENIZER_DIR = Path(__file__).parent
VOCAB_SIZES = [500, 1000, 2000, 5000]

SPECIAL_TOKENS = ["[UNK]"]

TEST_SENTENCE = "Bilbo went into the forest."
SUBWORD_TEST_WORDS = [
    "walking",
    "walked",
    "forest",
    "forests",
    "unusual",
    "adventure",
]


# ---------------------------------------------------------------------------
# 1 & 2. Load + minimally clean each book, then combine
# ---------------------------------------------------------------------------

# Project Gutenberg wraps its text in standard header/footer boilerplate
# delimited by these markers. We strip that boilerplate (it's not part of
# the author's actual prose) but otherwise leave the text untouched.
GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)


def load_and_clean(path: Path) -> str:
    """Read a book file and apply only minimal, non-destructive cleanup.

    We deliberately do NOT: lowercase, strip punctuation, remove stop
    words, or stem. Those transformations would destroy exactly the
    stylistic fingerprints (word choice, punctuation habits, sentence
    shape) that a later author-identification model needs to see.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # Strip Project Gutenberg boilerplate, if present.
    text = GUTENBERG_START_RE.split(text, maxsplit=1)
    text = text[-1] if len(text) > 1 else text[0]
    text = GUTENBERG_END_RE.split(text, maxsplit=1)[0]

    # Normalize excessive whitespace: collapse runs of blank lines and
    # trailing spaces, but keep paragraph breaks and all words/punctuation.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def build_combined_corpus() -> tuple[str, str, str]:
    hobbit_text = load_and_clean(HOBBIT_PATH)
    lost_world_text = load_and_clean(LOST_WORLD_PATH)
    combined_text = hobbit_text + "\n\n" + lost_world_text
    return hobbit_text, lost_world_text, combined_text


# ---------------------------------------------------------------------------
# 3 & 4. Train a BPE tokenizer (whitespace pre-tokenizer) from scratch
# ---------------------------------------------------------------------------

def train_bpe_tokenizer(corpus_path: Path, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    tokenizer.train([str(corpus_path)], trainer)
    return tokenizer


def load_tokenizer(vocab_size: int) -> Tokenizer:
    """Load a tokenizer previously trained and saved by this script."""
    path = TOKENIZER_DIR / f"tokenizer_vocab{vocab_size}.json"
    return Tokenizer.from_file(str(path))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(tokenizer: Tokenizer, vocab_size: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"Vocabulary size requested: {vocab_size}")
    print(f"Vocabulary size actual:    {tokenizer.get_vocab_size()}")
    print(f"{'=' * 60}")

    # Sanity check #2 — familiar sentence
    encoding = tokenizer.encode(TEST_SENTENCE)
    decoded = tokenizer.decode(encoding.ids)

    print(f"\nSentence: {TEST_SENTENCE!r}")
    print(f"Tokens:   {encoding.tokens}")
    print(f"IDs:      {encoding.ids}")
    print(f"Decoded:  {decoded!r}")

    # Sanity check #3 — subword behavior on related words
    print("\nSubword behavior:")
    for word in SUBWORD_TEST_WORDS:
        pieces = tokenizer.encode(word).tokens
        print(f"  {word:12s} -> {pieces}")

    # Optional — sample of learned vocabulary
    vocab = tokenizer.get_vocab()
    sample = sorted(vocab.items(), key=lambda kv: kv[1])[:15]
    print("\nSample vocabulary (first 15 by id):")
    for token, idx in sample:
        print(f"  {idx:5d}  {token!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading and cleaning books...")
    hobbit_text, lost_world_text, combined_text = build_combined_corpus()

    print(f"  hobbit.txt      : {len(hobbit_text):>10,} characters")
    print(f"  lost_world.txt  : {len(lost_world_text):>10,} characters")
    print(f"  combined corpus : {len(combined_text):>10,} characters")

    combined_path = DATA_DIR / "combined_corpus.txt"
    combined_path.write_text(combined_text, encoding="utf-8")
    print(f"\nCombined corpus written to: {combined_path}")

    for vocab_size in VOCAB_SIZES:
        print(f"\nTraining BPE tokenizer with vocab_size={vocab_size}...")
        tokenizer = train_bpe_tokenizer(combined_path, vocab_size)

        out_path = TOKENIZER_DIR / f"tokenizer_vocab{vocab_size}.json"
        tokenizer.save(str(out_path))
        print(f"Saved: {out_path}")

        run_sanity_checks(tokenizer, vocab_size)

    print(f"\n{'=' * 60}")
    print("Step 1 complete. Saved tokenizers:")
    for vocab_size in VOCAB_SIZES:
        print(f"  tokenizer_vocab{vocab_size}.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
