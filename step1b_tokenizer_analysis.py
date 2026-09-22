"""
Step 1B — Tokenizer Analysis + Decoder Fix

Deeper analysis of the four BPE tokenizers trained in Step 1
(vocab sizes 500 / 1000 / 2000 / 5000), plus a fix for the decoder
spacing bug discovered while sanity-checking Step 1's output.

This step does NOT retrain anything: no changes to vocab sizes,
training logic, or the Whitespace pre-tokenization strategy. It only
loads the already-saved tokenizers and analyzes/decodes with them.

Sections:
  - Decoder Fix   : diagnoses and fixes the spacing bug in decode()
  - Experiment 1  : tokenization efficiency across the full corpus
  - Experiment 2  : expanded word list across vocab sizes
  - Experiment 3  : one real sentence per author, across vocab sizes
"""

import re

from tokenizers import Tokenizer, decoders

from step1_bpe_tokenizer import (
    HOBBIT_PATH,
    LOST_WORLD_PATH,
    VOCAB_SIZES,
    build_combined_corpus,
    load_and_clean,
    load_tokenizer,
)

SECTION_RULE = "=" * 78


def print_section(title: str) -> None:
    print(f"\n{SECTION_RULE}")
    print(title)
    print(SECTION_RULE)


# ---------------------------------------------------------------------------
# Decoder fix
# ---------------------------------------------------------------------------
#
# Bug: Tokenizer.decode() with no decoder configured just joins every
# token with a single space. That's wrong here in two ways:
#   1. BPE subword pieces of the SAME word (e.g. "fore" + "st") get a
#      stray space between them  -> "fore st" instead of "forest".
#   2. Punctuation tokens (e.g. ".") get a stray space before them
#      -> "st ." instead of "st.".
#
# HF's built-in decoders (WordPiece, BPEDecoder, Metaspace) don't fix
# this cleanly for our setup: they all assume a marker token (like
# WordPiece's "##" or ByteLevel's "Ġ") distinguishes a word-initial
# piece from a continuation piece. Our tokenizer uses a plain
# Whitespace pre-tokenizer, which never adds such a marker, so those
# decoders can't tell continuations apart from new words -- they just
# fall back to concatenating EVERY token with no space at all (shown
# below), which is wrong in the opposite direction.
#
# Fix: Encoding.word_ids already tells us, per token, which original
# pre-tokenized "word" it came from (this info comes from
# pre-tokenization, not from any marker on the token string itself).
# Two tokens sharing a word_id are subword pieces of one word -> no
# space between them. Tokens from different word_ids are separate
# pretokens -> a space belongs between them, UNLESS the next token is
# pure punctuation, in which case it should hug the previous token.
# That punctuation check is the "minimal regex-based cleanup" called
# for when the built-in decoders fall short.

_PUNCT_ONLY = re.compile(r"^[^\w\s]+$")


def clean_decode(tokenizer: Tokenizer, text: str) -> str:
    """Encode `text` and reconstruct it with correct spacing.

    Requires re-encoding `text` (rather than working from raw ids alone)
    because the word-boundary information this fix relies on
    (Encoding.word_ids) only exists on a fresh Encoding, not on a bare
    id sequence -- our tokenizer's vocabulary carries no marker that
    would let us recover word boundaries from ids alone.
    """
    encoding = tokenizer.encode(text)
    pieces: list[str] = []
    prev_word_id = None

    for token, word_id in zip(encoding.tokens, encoding.word_ids):
        is_continuation = word_id is not None and word_id == prev_word_id
        is_punct = bool(_PUNCT_ONLY.match(token))
        if pieces and not is_continuation and not is_punct:
            pieces.append(" ")
        pieces.append(token)
        prev_word_id = word_id

    return "".join(pieces)


def demo_decoder_fix() -> None:
    print_section("DECODER FIX")

    test_sentence = "Bilbo went into the forest."
    print(f"Test sentence: {test_sentence!r}")

    # Show why a built-in decoder doesn't cleanly solve this, using the
    # worst-case tokenizer (vocab500, where subword splitting is heaviest).
    worst_case = load_tokenizer(500)
    worst_case.decoder = decoders.WordPiece(prefix="", cleanup=True)
    worst_encoding = worst_case.encode(test_sentence)
    print("\nBuilt-in decoder attempt (WordPiece decoder, vocab500):")
    print(f"  tokens -> {worst_encoding.tokens}")
    print(f"  result -> {worst_case.decode(worst_encoding.ids)!r}")
    print("  (no continuation marker exists to guide it, so it collapses")
    print("   every token together with NO spaces at all -- also wrong.)")

    print("\nApplying clean_decode() (word_ids + punctuation regex) to all")
    print("four tokenizers:\n")

    for vocab_size in VOCAB_SIZES:
        tokenizer = load_tokenizer(vocab_size)
        encoding = tokenizer.encode(test_sentence)

        buggy = tokenizer.decode(encoding.ids)  # old, unconfigured decode()
        fixed = clean_decode(tokenizer, test_sentence)
        matches = fixed == test_sentence

        print(f"vocab{vocab_size}:")
        print(f"  tokens          : {encoding.tokens}")
        print(f"  old decode()    : {buggy!r}")
        print(f"  clean_decode()  : {fixed!r}")
        print(f"  exact match?    : {matches}")
        print()


# ---------------------------------------------------------------------------
# Experiment 1 — Tokenization efficiency across the full corpus
# ---------------------------------------------------------------------------

def experiment1_efficiency(combined_text: str) -> None:
    print_section("EXPERIMENT 1 — Tokenization Efficiency Across Full Corpus")

    n_chars = len(combined_text)
    print(f"Combined corpus size: {n_chars:,} characters\n")

    print(f"{'Vocab size':<12}{'Total tokens':<16}{'Chars / token':<16}")
    print("-" * 44)
    for vocab_size in VOCAB_SIZES:
        tokenizer = load_tokenizer(vocab_size)
        n_tokens = len(tokenizer.encode(combined_text).ids)
        chars_per_token = n_chars / n_tokens
        print(f"{vocab_size:<12}{n_tokens:<16,}{chars_per_token:<16.2f}")


# ---------------------------------------------------------------------------
# Experiment 2 — Expanded word list test
# ---------------------------------------------------------------------------

EXPANDED_WORDS = [
    "walking",
    "walked",
    "forest",
    "forests",
    "adventure",
    "adventures",
    "Bilbo",
    "Gandalf",
    "creature",
    "mountains",
    "underground",
    "unexpected",
]


def experiment2_word_list() -> None:
    print_section("EXPERIMENT 2 — Expanded Word List Across Vocab Sizes")

    tokenizers = {v: load_tokenizer(v) for v in VOCAB_SIZES}

    col_width = 22
    header = f"{'Word':<14}" + "".join(
        f"vocab{v:<{col_width - 5}}" for v in VOCAB_SIZES
    )
    print(header)
    print("-" * len(header))

    for word in EXPANDED_WORDS:
        row = f"{word:<14}"
        for v in VOCAB_SIZES:
            pieces = "+".join(tokenizers[v].encode(word).tokens)
            row += f"{pieces:<{col_width}}"
        print(row)


# ---------------------------------------------------------------------------
# Experiment 3 — Real sentence comparison across both authors
# ---------------------------------------------------------------------------

HOBBIT_SENTENCE = (
    "Then they went back, and found Thorin with his feet on the fender "
    "smoking a pipe."
)
LOST_WORLD_SENTENCE = (
    "The old man nodded as I entered the room, and he pushed his "
    "spectacles far up on his bald forehead."
)


def _assert_sentence_is_real(sentence: str, source_path, label: str) -> None:
    """Confirm `sentence` actually appears in the cleaned source text."""
    normalized_source = re.sub(r"\s+", " ", load_and_clean(source_path))
    if sentence not in normalized_source:
        raise AssertionError(f"{label} sentence not found verbatim in source text")


def experiment3_sentence_comparison() -> None:
    print_section("EXPERIMENT 3 — Real Sentence Comparison Across Authors")

    _assert_sentence_is_real(HOBBIT_SENTENCE, HOBBIT_PATH, "Hobbit")
    _assert_sentence_is_real(LOST_WORLD_SENTENCE, LOST_WORLD_PATH, "Lost World")

    sentences = [
        ("The Hobbit (Tolkien)", HOBBIT_SENTENCE),
        ("The Lost World (Doyle)", LOST_WORLD_SENTENCE),
    ]

    for label, sentence in sentences:
        print(f"\n{label}:")
        print(f"  {sentence!r}\n")
        for vocab_size in VOCAB_SIZES:
            tokenizer = load_tokenizer(vocab_size)
            tokens = tokenizer.encode(sentence).tokens
            print(f"  vocab{vocab_size:<5} ({len(tokens):>2} tokens): {tokens}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _, _, combined_text = build_combined_corpus()

    demo_decoder_fix()
    experiment1_efficiency(combined_text)
    experiment2_word_list()
    experiment3_sentence_comparison()

    print_section("STEP 1B COMPLETE")


if __name__ == "__main__":
    main()
