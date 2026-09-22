"""
Step 4 — Negative Log-Likelihood + Perplexity Evaluation

Evaluates how well each author's Add-k smoothed language model (from
Step 3) predicts each book's token sequence, for every vocab size and
every n-gram order. Every book is scored under BOTH author models, so
later steps can compare which model fits a text better.

Pipeline position:
    ... -> Add-k smoothed probabilities (Step 3) -> [NLL + Perplexity]  <- here
                                                      -> author ID (later)

This step does NOT retrain tokenizers, rebuild n-gram counts, change
the Step 3 smoothing implementation, or implement a final
classification system -- those are other steps. It only scores.

Definitions:
    NLL(tokens, n, model) = -sum( log P(token_i | context_i) )
        summed over every predicted n-gram in the sequence, using
        natural log and the model's Add-k smoothed probabilities.

    Perplexity = exp(NLL / num_predictions)

    num_predictions:
        unigram -> len(tokens)
        bigram  -> len(tokens) - 1
        trigram -> len(tokens) - 2

Unseen n-grams (not in lm_probs[...]["probs"]) are handled with the
same Add-k formula Step 3 already established:
    P = k / (context_count + k * vocab_size)
using the "k", "vocab_size", and "context_counts" stored in lm_probs.pkl.
"""

import csv
import math
import pickle
from pathlib import Path

from step1_bpe_tokenizer import (
    HOBBIT_PATH,
    LOST_WORLD_PATH,
    VOCAB_SIZES,
    load_and_clean,
    load_tokenizer,
)
from step2_ngram_counts import NGRAM_ORDERS, ORDER_NAMES, print_section
from step3_addk_smoothing import PROBS_PATH

RESULTS_PKL_PATH = Path(__file__).parent / "step4_results.pkl"
RESULTS_CSV_PATH = Path(__file__).parent / "step4_results.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_language_models(path: Path = PROBS_PATH) -> dict:
    """Load lm_probs[vocab_size][book_name][n] -> smoothed-LM entry."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokenized_books(tokenizers: dict) -> dict:
    """Encode each book with each tokenizer.

    Returns token_ids[(book_name, vocab_size)] -> list[int]
    """
    book_texts = {
        "hobbit": load_and_clean(HOBBIT_PATH),
        "lost_world": load_and_clean(LOST_WORLD_PATH),
    }
    token_ids = {}
    for vocab_size, tokenizer in tokenizers.items():
        for book_name, text in book_texts.items():
            token_ids[(book_name, vocab_size)] = tokenizer.encode(text).ids
    return token_ids


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def iter_ngrams(token_ids: list[int], n: int):
    """Yield every length-n sliding-window tuple over `token_ids`.

    Same windowing Step 2 used to build n-gram counts, so these tuples
    match the keys stored in lm_probs[...]["probs"].
    """
    if len(token_ids) < n:
        return
    yield from zip(*(token_ids[offset:] for offset in range(n)))


def get_ngram_probability(ngram: tuple, entry: dict) -> float:
    """Smoothed P(ngram) using Step 3's Add-k model `entry`.

    Looks up the observed probability first; falls back to the Add-k
    formula for unseen n-grams using the entry's stored k, vocab_size,
    and context_counts.
    """
    probs = entry["probs"]
    if ngram in probs:
        return probs[ngram]

    k = entry["k"]
    vocab_size = entry["vocab_size"]
    context = ngram[:-1]
    context_count = entry["context_counts"].get(context, 0)
    return k / (context_count + k * vocab_size)


def calculate_nll(token_ids: list[int], n: int, entry: dict) -> tuple[float, int]:
    """Total negative log-likelihood and number of predictions made."""
    total_nll = 0.0
    num_predictions = 0
    for ngram in iter_ngrams(token_ids, n):
        p = get_ngram_probability(ngram, entry)
        assert p > 0, f"Add-k smoothing produced a non-positive probability for {ngram}"
        total_nll += -math.log(p)
        num_predictions += 1
    return total_nll, num_predictions


def calculate_perplexity(nll: float, num_predictions: int) -> float:
    return math.exp(nll / num_predictions)


def evaluate_text(token_ids: list[int], entry: dict, n: int) -> dict:
    """Score one token sequence under one model at one n-gram order."""
    nll, num_predictions = calculate_nll(token_ids, n, entry)
    average_nll = nll / num_predictions
    perplexity = calculate_perplexity(nll, num_predictions)
    return {
        "nll": nll,
        "num_predictions": num_predictions,
        "average_nll": average_nll,
        "perplexity": perplexity,
    }


# ---------------------------------------------------------------------------
# Full evaluation sweep
# ---------------------------------------------------------------------------

def evaluate_all(
    lm_probs: dict,
    token_ids_by_book: dict,
    book_names: list[str],
    vocab_sizes=VOCAB_SIZES,
    orders=NGRAM_ORDERS,
) -> dict:
    """results[vocab_size][text_book][model_author][n] = evaluate_text(...)"""
    results: dict = {}

    for vocab_size in vocab_sizes:
        print_section(f"EVALUATING VOCAB SIZE {vocab_size}")
        results[vocab_size] = {}

        for text_book in book_names:
            token_ids = token_ids_by_book[(text_book, vocab_size)]
            results[vocab_size][text_book] = {}

            for model_author in book_names:
                results[vocab_size][text_book][model_author] = {}

                for n in orders:
                    entry = lm_probs[vocab_size][model_author][n]
                    outcome = evaluate_text(token_ids, entry, n)
                    results[vocab_size][text_book][model_author][n] = outcome

                    print(
                        f"  [{ORDER_NAMES[n]:<8}] text={text_book:<11} "
                        f"model={model_author:<11} "
                        f"NLL={outcome['nll']:>10.3f}  "
                        f"PPL={outcome['perplexity']:>10.3f}"
                    )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def flatten_results(results: dict, book_names: list[str], vocab_sizes=VOCAB_SIZES, orders=NGRAM_ORDERS) -> list[dict]:
    rows = []
    for vocab_size in vocab_sizes:
        for text_book in book_names:
            for model_author in book_names:
                for n in orders:
                    outcome = results[vocab_size][text_book][model_author][n]
                    rows.append({
                        "vocab_size": vocab_size,
                        "text_book": text_book,
                        "model_author": model_author,
                        "ngram_order": ORDER_NAMES[n],
                        "nll": outcome["nll"],
                        "num_predictions": outcome["num_predictions"],
                        "average_nll": outcome["average_nll"],
                        "perplexity": outcome["perplexity"],
                    })
    return rows


def print_results_table(rows: list[dict]) -> None:
    print_section("RESULTS TABLE")
    header = (
        f"{'Vocab':<7}{'Text':<12}{'Model':<12}{'Order':<9}"
        f"{'NLL':>12}{'NumPred':>10}{'AvgNLL':>10}{'PPL':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['vocab_size']:<7}{row['text_book']:<12}{row['model_author']:<12}"
            f"{row['ngram_order']:<9}{row['nll']:>12.3f}{row['num_predictions']:>10}"
            f"{row['average_nll']:>10.4f}{row['perplexity']:>12.3f}"
        )


def print_comparison(results: dict, book_names: list[str], vocab_sizes=VOCAB_SIZES, orders=NGRAM_ORDERS) -> None:
    """For each text, show NLL/perplexity under both author models side by side."""
    print_section("MODEL COMPARISON — Which author model fits each text better?")

    header = (
        f"{'Vocab':<7}{'Text':<12}{'Order':<9}"
        f"{'NLL(hobbit)':>13}{'NLL(lost_world)':>17}"
        f"{'PPL(hobbit)':>13}{'PPL(lost_world)':>17}  {'Lower PPL':<12}"
    )
    print(header)
    print("-" * len(header))

    for vocab_size in vocab_sizes:
        for text_book in book_names:
            for n in orders:
                hobbit_r = results[vocab_size][text_book]["hobbit"][n]
                lost_r = results[vocab_size][text_book]["lost_world"][n]
                better = "hobbit" if hobbit_r["perplexity"] < lost_r["perplexity"] else "lost_world"
                print(
                    f"{vocab_size:<7}{text_book:<12}{ORDER_NAMES[n]:<9}"
                    f"{hobbit_r['nll']:>13.3f}{lost_r['nll']:>17.3f}"
                    f"{hobbit_r['perplexity']:>13.3f}{lost_r['perplexity']:>17.3f}  {better:<12}"
                )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    results: dict,
    token_ids_by_book: dict,
    book_names: list[str],
    vocab_sizes=VOCAB_SIZES,
    orders=NGRAM_ORDERS,
) -> list[str]:
    """Returns a list of failure messages (empty list = everything passed)."""
    print_section("SANITY CHECKS")
    failures: list[str] = []

    for vocab_size in vocab_sizes:
        for text_book in book_names:
            token_len = len(token_ids_by_book[(text_book, vocab_size)])
            for model_author in book_names:
                for n in orders:
                    outcome = results[vocab_size][text_book][model_author][n]
                    tag = f"vocab{vocab_size}/{text_book}/{model_author}/{ORDER_NAMES[n]}"

                    if not (outcome["nll"] >= 0):
                        failures.append(f"{tag}: NLL < 0 ({outcome['nll']})")
                    if not math.isfinite(outcome["nll"]):
                        failures.append(f"{tag}: NLL is not finite ({outcome['nll']})")
                    if not (outcome["perplexity"] >= 1):
                        failures.append(f"{tag}: perplexity < 1 ({outcome['perplexity']})")
                    if not math.isfinite(outcome["perplexity"]):
                        failures.append(f"{tag}: perplexity is not finite ({outcome['perplexity']})")

                    expected_predictions = token_len - n + 1
                    if outcome["num_predictions"] != expected_predictions:
                        failures.append(
                            f"{tag}: num_predictions={outcome['num_predictions']} "
                            f"!= expected {expected_predictions}"
                        )

    total_checks = len(vocab_sizes) * len(book_names) * len(book_names) * len(orders)
    if failures:
        print(f"FAILED: {len(failures)} issue(s) found across {total_checks} evaluations:")
        for msg in failures:
            print(f"  - {msg}")
    else:
        print(f"PASSED: all checks OK across {total_checks} evaluations "
              f"(NLL >= 0, perplexity >= 1, finite values, correct num_predictions).")
        print("Zero-probability lookups are additionally prevented at the source: "
              "get_ngram_probability() asserts p > 0 on every call, guaranteed by "
              "Add-k smoothing's k / (context_count + k * V) floor for unseen n-grams.")

    return failures


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results(results: dict, rows: list[dict]) -> None:
    with open(RESULTS_PKL_PATH, "wb") as f:
        pickle.dump(results, f)

    with open(RESULTS_CSV_PATH, "w", newline="") as f:
        fieldnames = [
            "vocab_size", "text_book", "model_author", "ngram_order",
            "nll", "num_predictions", "average_nll", "perplexity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading smoothed language models from: {PROBS_PATH}")
    lm_probs = load_language_models()
    book_names = list(next(iter(lm_probs.values())).keys())

    print("Loading tokenizers and encoding both books at every vocab size...")
    tokenizers = {v: load_tokenizer(v) for v in VOCAB_SIZES}
    token_ids_by_book = load_tokenized_books(tokenizers)
    for (book_name, vocab_size), ids in sorted(token_ids_by_book.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  {book_name:<12} @ vocab{vocab_size}: {len(ids):,} tokens")

    results = evaluate_all(lm_probs, token_ids_by_book, book_names)

    rows = flatten_results(results, book_names)
    print_results_table(rows)
    print_comparison(results, book_names)

    failures = run_sanity_checks(results, token_ids_by_book, book_names)

    save_results(results, rows)

    print_section("STEP 4 COMPLETE")
    print(f"Pickle results saved to: {RESULTS_PKL_PATH}")
    print(f"CSV results saved to:    {RESULTS_CSV_PATH}")
    print(f"Sanity checks: {'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")


if __name__ == "__main__":
    main()
