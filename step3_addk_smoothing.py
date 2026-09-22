"""
Step 3 — Add-k Smoothed N-gram Probabilities

Loads the raw n-gram counts from Step 2 (ngram_counts.pkl) and converts
them into Add-k smoothed language model probabilities, for every
n-gram order (unigram/bigram/trigram), every vocab size, and each book
separately.

Pipeline position:
    ... -> Token IDs -> N-gram counts (Step 2) -> [Add-k smoothing]  <- here
                                                    -> perplexity (later)
                                                    -> author ID (later)

This step does NOT compute perplexity, negative log likelihood, do
train/test splitting, or author identification -- those are later
steps. It only converts counts into smoothed probabilities.

Add-k smoothing formula:
    P(w | context) = (count(context, w) + k) / (count(context) + k * V)

  - count(context, w) is the raw n-gram count from Step 2.
  - count(context) is the raw (n-1)-gram count (the empty context's
    "count" is just the total number of tokens, for unigrams).
  - V is the tokenizer's actual BPE vocabulary size for that vocab
    setting (e.g. 500), NOT the number of distinct tokens observed in
    the book -- per the assignment spec, unseen vocab entries must
    still be accounted for in the denominator.
"""

import pickle
from pathlib import Path

from step1_bpe_tokenizer import VOCAB_SIZES, load_tokenizer
from step2_ngram_counts import COUNTS_PATH, NGRAM_ORDERS, ORDER_NAMES, print_section

DEFAULT_K = 1.0
PROBS_PATH = Path(__file__).parent / "lm_probs.pkl"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ngram_counts(path: Path = COUNTS_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def compute_smoothed_probs(
    all_counts: dict,
    vocab_sizes_actual: dict[int, int],
    k: float = DEFAULT_K,
) -> dict:
    """Add-k smoothed probabilities, indexed the same way as `all_counts`.

    Returns:
      lm_probs[vocab_size][book_name][n] = {
          "k": k,
          "vocab_size": V,                     # actual BPE vocab size used
          "probs": {ngram_id_tuple: prob},      # observed n-grams only
          "context_counts": {context_tuple: count},
      }

    "probs" only holds entries for n-grams actually observed in the
    corpus (storing every possible n-gram, including unseen ones, is
    combinatorially infeasible -- e.g. 5000^3 possible trigrams at
    vocab5000). "context_counts" is kept alongside so a later step can
    compute the smoothed probability of an UNSEEN n-gram too, via the
    same formula:
        context_count = context_counts.get(context, 0)
        prob = k / (context_count + k * V)
    """
    lm_probs: dict = {}

    for vocab_size, counts_by_book in all_counts.items():
        V = vocab_sizes_actual[vocab_size]
        lm_probs[vocab_size] = {}

        for book_name, counts_by_order in counts_by_book.items():
            total_tokens = sum(counts_by_order[1].values())
            lm_probs[vocab_size][book_name] = {}

            for n in NGRAM_ORDERS:
                ngram_counts = counts_by_order[n]
                context_counts = (
                    {(): total_tokens} if n == 1 else counts_by_order[n - 1]
                )

                probs = {
                    ngram: (count + k) / (context_counts.get(ngram[:-1], 0) + k * V)
                    for ngram, count in ngram_counts.items()
                }

                lm_probs[vocab_size][book_name][n] = {
                    "k": k,
                    "vocab_size": V,
                    "probs": probs,
                    "context_counts": context_counts,
                }

    return lm_probs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def sanity_check_normalization(lm_probs: dict, tokenizers: dict) -> None:
    """For one representative context per order, confirm P(. | context)
    summed over the FULL vocabulary (seen and unseen tokens alike) is
    ~1.0, as Add-k smoothing guarantees. This is the standard check
    that the formula (not just the code) was applied correctly.
    """
    print_section("SANITY CHECK - Smoothed Probabilities Normalize to ~1.0")

    header = f"{'Vocab':<8}{'Author':<14}{'Order':<10}{'Context':<26}{'Sum P(.|context)':<18}"
    print(header)
    print("-" * len(header))

    for vocab_size in VOCAB_SIZES:
        tokenizer = tokenizers[vocab_size]
        V = tokenizer.get_vocab_size()

        for book_name, by_order in lm_probs[vocab_size].items():
            for n in NGRAM_ORDERS:
                entry = by_order[n]
                k, probs, context_counts = entry["k"], entry["probs"], entry["context_counts"]

                if n == 1:
                    context = ()
                    label = "<all tokens>"
                else:
                    context = max(context_counts, key=context_counts.get)
                    label = str(tuple(tokenizer.id_to_token(i) for i in context))
                context_count = context_counts.get(context, 0)

                unseen_prob = k / (context_count + k * V)
                total = sum(
                    probs.get(context + (token_id,), unseen_prob)
                    for token_id in range(V)
                )

                print(f"{vocab_size:<8}{book_name:<14}{ORDER_NAMES[n]:<10}{label:<26}{total:<18.6f}")


TOP_K_EXAMPLES = 5


def print_example_probs(lm_probs: dict, tokenizers: dict, book_names: list[str]) -> None:
    print_section(f"TOP {TOP_K_EXAMPLES} HIGHEST-PROBABILITY N-GRAMS (illustrative sample)")

    sample_vocab_sizes = [VOCAB_SIZES[0], VOCAB_SIZES[-1]]

    for vocab_size in sample_vocab_sizes:
        tokenizer = tokenizers[vocab_size]
        for book_name in book_names:
            print(f"\n-- vocab{vocab_size} / {book_name} (k={DEFAULT_K}) --")
            for n in (2, 3):
                probs = lm_probs[vocab_size][book_name][n]["probs"]
                top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_EXAMPLES]
                print(f"  {ORDER_NAMES[n]}s:")
                for ngram, prob in top:
                    tokens = tuple(tokenizer.id_to_token(i) for i in ngram)
                    print(f"    {tokens}  P={prob:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(k: float = DEFAULT_K) -> None:
    print(f"Loading n-gram counts from: {COUNTS_PATH}")
    all_counts = load_ngram_counts()
    book_names = list(next(iter(all_counts.values())).keys())

    print("Loading tokenizers to get authoritative BPE vocab sizes...")
    tokenizers = {v: load_tokenizer(v) for v in VOCAB_SIZES}
    vocab_sizes_actual = {v: t.get_vocab_size() for v, t in tokenizers.items()}
    for v in VOCAB_SIZES:
        print(f"  vocab{v} -> actual tokenizer size {vocab_sizes_actual[v]}")

    print(f"\nApplying Add-k smoothing (k={k}) across all vocab sizes, books, and orders...")
    lm_probs = compute_smoothed_probs(all_counts, vocab_sizes_actual, k=k)

    sanity_check_normalization(lm_probs, tokenizers)
    print_example_probs(lm_probs, tokenizers, book_names)

    with open(PROBS_PATH, "wb") as f:
        pickle.dump(lm_probs, f)

    print_section("STEP 3 COMPLETE")
    print(f"Smoothed LM probabilities saved to: {PROBS_PATH}")
    print(
        "Structure: lm_probs[vocab_size][book_name][n] -> "
        "{'k', 'vocab_size', 'probs': {ngram: prob}, 'context_counts': {context: count}}"
    )


if __name__ == "__main__":
    main()
