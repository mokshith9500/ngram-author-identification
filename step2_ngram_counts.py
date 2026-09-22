"""
Step 2 — N-gram Counts Per Author

Builds raw (unsmoothed) unigram / bigram / trigram frequency counts
over BPE token ID sequences, separately for each book/author, at each
of the four vocab sizes trained in Step 1.

Pipeline position:
    Raw books -> BPE tokenizer -> Token IDs -> [N-gram counts]  <- here
                                                -> add-k smoothing (later)
                                                -> perplexity (later)
                                                -> author ID (later)

This step does NOT retrain tokenizers, compute probabilities, apply
smoothing, or do any author identification -- that's later steps.
It only counts.

Design notes:
  - N-grams are built over each book's FULL token ID stream as one
    continuous sequence (no per-sentence <s>/</s> padding). Adding
    sentence-boundary tokens would mean touching the trained
    tokenizers' vocab, which Step 1 already fixed and this step must
    not retrain.
  - Counts are kept per author (Hobbit vs Lost World), not merged,
    since the end goal is comparing per-author language models.
"""

import pickle
from collections import Counter
from pathlib import Path

from step1_bpe_tokenizer import (
    HOBBIT_PATH,
    LOST_WORLD_PATH,
    VOCAB_SIZES,
    load_and_clean,
    load_tokenizer,
)

NGRAM_ORDERS = (1, 2, 3)
ORDER_NAMES = {1: "unigram", 2: "bigram", 3: "trigram"}

COUNTS_PATH = Path(__file__).parent / "ngram_counts.pkl"
TOP_K = 10

SECTION_RULE = "=" * 78


def print_section(title: str) -> None:
    print(f"\n{SECTION_RULE}")
    print(title)
    print(SECTION_RULE)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def build_ngram_counts(token_ids: list[int], n: int) -> Counter:
    """Raw frequency count of every n-gram (as an id-tuple) in `token_ids`."""
    if len(token_ids) < n:
        return Counter()
    ngrams = zip(*(token_ids[offset:] for offset in range(n)))
    return Counter(ngrams)


def compute_all_counts(
    book_texts: dict[str, str],
    vocab_sizes=VOCAB_SIZES,
    orders=NGRAM_ORDERS,
) -> tuple[dict, dict]:
    """Build n-gram counts for every (vocab size, book, order) combination.

    Returns:
      all_counts[vocab_size][book_name][n]  -> Counter[id-tuple -> count]
      token_ids[(book_name, vocab_size)]    -> list[int], the encoded book
    """
    all_counts: dict = {}
    token_ids: dict = {}

    for vocab_size in vocab_sizes:
        tokenizer = load_tokenizer(vocab_size)
        all_counts[vocab_size] = {}
        for book_name, text in book_texts.items():
            ids = tokenizer.encode(text).ids
            token_ids[(book_name, vocab_size)] = ids
            all_counts[vocab_size][book_name] = {
                n: build_ngram_counts(ids, n) for n in orders
            }

    return all_counts, token_ids


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(all_counts: dict, token_ids: dict, book_names: list[str]) -> None:
    print_section("N-GRAM COUNT SUMMARY (per author, per vocab size)")

    header = f"{'Vocab':<8}{'Author':<14}{'Order':<10}{'Total n-grams':<16}{'Unique n-grams':<16}"
    print(header)
    print("-" * len(header))

    for vocab_size in VOCAB_SIZES:
        for book_name in book_names:
            n_tokens = len(token_ids[(book_name, vocab_size)])
            for n in NGRAM_ORDERS:
                counter = all_counts[vocab_size][book_name][n]
                total = sum(counter.values())
                unique = len(counter)
                print(
                    f"{vocab_size:<8}{book_name:<14}{ORDER_NAMES[n]:<10}"
                    f"{total:<16,}{unique:<16,}"
                )
        print(f"  (token stream length at vocab{vocab_size}: "
              + ", ".join(f"{b}={len(token_ids[(b, vocab_size)]):,}" for b in book_names)
              + ")")


def print_sample_ngrams(all_counts: dict, tokenizers: dict, book_names: list[str]) -> None:
    print_section(f"TOP {TOP_K} MOST FREQUENT N-GRAMS (illustrative sample)")

    # Keep the sample focused: show bigrams and trigrams (unigrams are just
    # the vocab frequency distribution) at the smallest and largest vocab
    # sizes, per author.
    sample_vocab_sizes = [VOCAB_SIZES[0], VOCAB_SIZES[-1]]

    for vocab_size in sample_vocab_sizes:
        tokenizer = tokenizers[vocab_size]
        for book_name in book_names:
            print(f"\n-- vocab{vocab_size} / {book_name} --")
            for n in (2, 3):
                counter = all_counts[vocab_size][book_name][n]
                print(f"  {ORDER_NAMES[n]}s:")
                for id_tuple, count in counter.most_common(TOP_K):
                    tokens = tuple(tokenizer.id_to_token(i) for i in id_tuple)
                    print(f"    {tokens}  x{count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading and cleaning books...")
    book_texts = {
        "hobbit": load_and_clean(HOBBIT_PATH),
        "lost_world": load_and_clean(LOST_WORLD_PATH),
    }
    book_names = list(book_texts.keys())

    print("Building n-gram counts for all vocab sizes and authors...")
    all_counts, token_ids = compute_all_counts(book_texts)

    print_summary_table(all_counts, token_ids, book_names)

    tokenizers = {v: load_tokenizer(v) for v in VOCAB_SIZES}
    print_sample_ngrams(all_counts, tokenizers, book_names)

    with open(COUNTS_PATH, "wb") as f:
        pickle.dump(all_counts, f)

    print_section("STEP 2 COMPLETE")
    print(f"N-gram counts saved to: {COUNTS_PATH}")
    print("Structure: all_counts[vocab_size][book_name][n] -> Counter[id-tuple -> count]")


if __name__ == "__main__":
    main()
