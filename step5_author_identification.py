"""
Step 5 — Author Identification

Uses the NLL/perplexity results already computed in Step 4 to predict
which author wrote each text: for a given text, whichever author's
language model gives it the LOWER perplexity (= assigns it higher
probability) is the predicted author.

Pipeline position:
    ... -> NLL + Perplexity (Step 4) -> [Author Identification]  <- here

This step does NOT re-tokenize anything, re-count n-grams, re-fit
smoothing, or recompute NLL/perplexity -- it only reads step4_results.pkl
and turns those scores into a prediction + accuracy report. It also
does NOT introduce a machine-learning classifier (logistic regression,
SVM, etc.) -- the "classifier" here is simply "lower perplexity wins",
using the existing statistical language models.

Method (per the assignment's worked example):
    For a text, compare its perplexity under the Hobbit model and
    under the Lost World model.
        predicted_author = argmin(hobbit_perplexity, lost_world_perplexity)

This is done independently for every (vocab_size, ngram_order) pair,
so the effect of vocab size and n-gram order on identification
accuracy can be inspected -- NOT to declare any one of them "best".
"""

import csv
import math
import pickle
from pathlib import Path

from step1_bpe_tokenizer import VOCAB_SIZES
from step2_ngram_counts import NGRAM_ORDERS, ORDER_NAMES, print_section
from step4_evaluation import RESULTS_PKL_PATH

BOOK_NAMES = ("hobbit", "lost_world")

RESULTS_PKL_OUT_PATH = Path(__file__).parent / "step5_results.pkl"
RESULTS_CSV_OUT_PATH = Path(__file__).parent / "step5_results.csv"

CSV_FIELDNAMES = [
    "vocab_size", "text_book", "ngram_order",
    "hobbit_perplexity", "lost_world_perplexity",
    "predicted_author", "actual_author", "correct",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_step4_results(path: Path = RESULTS_PKL_PATH) -> dict:
    """Load results[vocab_size][text_book][model_author][n] from Step 4."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Prediction: "lower perplexity wins"
# ---------------------------------------------------------------------------

def predict_author(hobbit_perplexity: float, lost_world_perplexity: float) -> str:
    """The author whose model assigns this text the lower perplexity.

    Ties (hobbit_perplexity == lost_world_perplexity) are effectively
    impossible with real floating-point perplexities from two
    different corpora, but if it ever happens we deterministically
    fall back to "lost_world" via the else-branch below, rather than
    returning some third "tie" value.
    """
    if hobbit_perplexity < lost_world_perplexity:
        return "hobbit"
    return "lost_world"


def identify_authors(
    step4_results: dict,
    vocab_sizes=VOCAB_SIZES,
    orders=NGRAM_ORDERS,
    book_names=BOOK_NAMES,
) -> list[dict]:
    """Build one prediction row per (vocab_size, text_book, ngram_order)."""
    rows = []

    for vocab_size in vocab_sizes:
        for text_book in book_names:
            # Every text was scored under BOTH author models in Step 4.
            scores_for_text = step4_results[vocab_size][text_book]
            assert set(scores_for_text.keys()) >= set(book_names), (
                f"vocab{vocab_size}/{text_book} is missing a model score"
            )

            for n in orders:
                hobbit_ppl = scores_for_text["hobbit"][n]["perplexity"]
                lost_world_ppl = scores_for_text["lost_world"][n]["perplexity"]

                predicted_author = predict_author(hobbit_ppl, lost_world_ppl)
                actual_author = text_book  # each book's true author is itself

                rows.append({
                    "vocab_size": vocab_size,
                    "text_book": text_book,
                    "ngram_order": ORDER_NAMES[n],
                    "hobbit_perplexity": hobbit_ppl,
                    "lost_world_perplexity": lost_world_ppl,
                    "predicted_author": predicted_author,
                    "actual_author": actual_author,
                    "correct": predicted_author == actual_author,
                })

    return rows


# ---------------------------------------------------------------------------
# Accuracy breakdowns
# ---------------------------------------------------------------------------

def accuracy_by(rows: list[dict], key: str) -> dict:
    """Classification accuracy grouped by `key` (e.g. "vocab_size")."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    return {
        value: {
            "correct": sum(r["correct"] for r in group),
            "total": len(group),
            "accuracy": sum(r["correct"] for r in group) / len(group),
        }
        for value, group in groups.items()
    }


def overall_accuracy(rows: list[dict]) -> dict:
    correct = sum(r["correct"] for r in rows)
    total = len(rows)
    return {"correct": correct, "total": total, "accuracy": correct / total}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_predictions(rows: list[dict]) -> None:
    print_section("PREDICTIONS PER TEXT")
    header = (
        f"{'Vocab':<7}{'Text':<12}{'Order':<9}"
        f"{'PPL(hobbit)':>13}{'PPL(lost_world)':>17}  "
        f"{'Predicted':<13}{'Actual':<13}{'Correct':<8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['vocab_size']:<7}{row['text_book']:<12}{row['ngram_order']:<9}"
            f"{row['hobbit_perplexity']:>13.3f}{row['lost_world_perplexity']:>17.3f}  "
            f"{row['predicted_author']:<13}{row['actual_author']:<13}"
            f"{'YES' if row['correct'] else 'NO':<8}"
        )


def print_accuracy_table(title: str, breakdown: dict) -> None:
    print_section(title)
    print(f"{'Group':<15}{'Correct':<10}{'Total':<10}{'Accuracy':<10}")
    print("-" * 45)
    for value, stats in breakdown.items():
        print(f"{str(value):<15}{stats['correct']:<10}{stats['total']:<10}{stats['accuracy']:<10.2%}")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(rows: list[dict], step4_results: dict) -> list[str]:
    """Returns a list of failure messages (empty list = everything passed)."""
    print_section("SANITY CHECKS")
    failures: list[str] = []

    expected_total_rows = len(VOCAB_SIZES) * len(BOOK_NAMES) * len(NGRAM_ORDERS)
    if len(rows) != expected_total_rows:
        failures.append(f"expected {expected_total_rows} rows, got {len(rows)}")

    for row in rows:
        tag = f"vocab{row['vocab_size']}/{row['text_book']}/{row['ngram_order']}"

        # Every text has exactly two author-model scores (hobbit + lost_world),
        # both present and both finite (no missing/NaN/inf).
        for ppl_key in ("hobbit_perplexity", "lost_world_perplexity"):
            ppl = row[ppl_key]
            if ppl is None or not math.isfinite(ppl):
                failures.append(f"{tag}: {ppl_key} is missing/NaN/inf ({ppl})")

        # predicted_author must be one of the two known authors.
        if row["predicted_author"] not in BOOK_NAMES:
            failures.append(f"{tag}: predicted_author={row['predicted_author']!r} not in {BOOK_NAMES}")

        # actual_author must correctly reflect the text's true source.
        if row["actual_author"] != row["text_book"]:
            failures.append(f"{tag}: actual_author={row['actual_author']!r} != text_book={row['text_book']!r}")

        # The prediction must match the "lower perplexity wins" metric.
        expected_prediction = predict_author(row["hobbit_perplexity"], row["lost_world_perplexity"])
        if row["predicted_author"] != expected_prediction:
            failures.append(
                f"{tag}: predicted_author={row['predicted_author']!r} "
                f"!= recomputed {expected_prediction!r}"
            )

    # Accuracy bookkeeping: group totals must add back up to the full row count.
    by_vocab = accuracy_by(rows, "vocab_size")
    if sum(stats["total"] for stats in by_vocab.values()) != len(rows):
        failures.append("accuracy_by_vocab_size totals do not sum to len(rows)")

    by_order = accuracy_by(rows, "ngram_order")
    if sum(stats["total"] for stats in by_order.values()) != len(rows):
        failures.append("accuracy_by_ngram_order totals do not sum to len(rows)")

    overall = overall_accuracy(rows)
    if overall["total"] != len(rows):
        failures.append("overall accuracy denominator does not equal len(rows)")
    if not (0 <= overall["accuracy"] <= 1):
        failures.append(f"overall accuracy out of [0, 1] range: {overall['accuracy']}")

    if failures:
        print(f"FAILED: {len(failures)} issue(s) found:")
        for msg in failures:
            print(f"  - {msg}")
    else:
        print(f"PASSED: all checks OK across {len(rows)} predictions "
              f"({expected_total_rows} expected = {len(VOCAB_SIZES)} vocab sizes "
              f"x {len(BOOK_NAMES)} texts x {len(NGRAM_ORDERS)} n-gram orders).")

    return failures


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results(rows: list[dict], accuracy_summary: dict) -> None:
    with open(RESULTS_PKL_OUT_PATH, "wb") as f:
        pickle.dump({"rows": rows, **accuracy_summary}, f)

    with open(RESULTS_CSV_OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading Step 4 evaluation results from: {RESULTS_PKL_PATH}")
    step4_results = load_step4_results()

    print("Identifying predicted author for each text (lower perplexity wins)...")
    rows = identify_authors(step4_results)

    print_predictions(rows)

    accuracy_by_vocab_size = accuracy_by(rows, "vocab_size")
    accuracy_by_ngram_order = accuracy_by(rows, "ngram_order")
    accuracy_overall = overall_accuracy(rows)

    print_accuracy_table("ACCURACY BY VOCAB SIZE", accuracy_by_vocab_size)
    print_accuracy_table("ACCURACY BY N-GRAM ORDER", accuracy_by_ngram_order)

    print_section("OVERALL ACCURACY")
    print(f"{accuracy_overall['correct']} / {accuracy_overall['total']} correct "
          f"({accuracy_overall['accuracy']:.2%})")

    failures = run_sanity_checks(rows, step4_results)

    accuracy_summary = {
        "accuracy_by_vocab_size": accuracy_by_vocab_size,
        "accuracy_by_ngram_order": accuracy_by_ngram_order,
        "accuracy_overall": accuracy_overall,
    }
    save_results(rows, accuracy_summary)

    print_section("STEP 5 COMPLETE")
    print(f"Pickle results saved to: {RESULTS_PKL_OUT_PATH}")
    print(f"CSV results saved to:    {RESULTS_CSV_OUT_PATH}")
    print(f"Sanity checks: {'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")


if __name__ == "__main__":
    main()
