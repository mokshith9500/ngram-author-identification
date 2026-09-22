# N-Gram Author Identification

## Overview

This project looks at author identification using classical statistical language models, not neural networks or LLMs. The idea is simple: train a separate language model on each author's writing, then see which model is less "surprised" by a given piece of text. Whichever model is less surprised (lower perplexity) is predicted as the author.

The two texts used are:

- J.R.R. Tolkien's *The Hobbit*
- Arthur Conan Doyle's *The Lost World*

The project has five steps:

1. BPE Tokenization
2. N-gram Counts
3. Probabilities and Add-k Smoothing
4. NLL and Perplexity Evaluation
5. Author Identification

```
Raw Books
   ↓
BPE Tokenization
   ↓
N-gram Counts
   ↓
Probabilities + Add-k Smoothing
   ↓
NLL + Perplexity
   ↓
Author Identification
```

## Step 1: BPE Tokenization

I trained one shared BPE tokenizer on the combined text of both books. Using a single shared tokenizer keeps things fair since both books end up using the exact same vocabulary and token IDs.

I tested four vocabulary sizes: 500, 1,000, 2,000, and 5,000. The combined corpus was about 928,569 characters.

| Vocabulary Size | Combined Token Count |
|---|---|
| 500 | 342,711 |
| 1,000 | 292,964 |
| 2,000 | 256,397 |
| 5,000 | 223,904 |

As the vocabulary got bigger, the tokenizer could represent more words as complete units instead of breaking them into smaller pieces. For example, "walking" gets split into multiple subword tokens at a small vocab size, but becomes a single token at a larger one.

All four vocabulary sizes were kept for the rest of the project so their effects could be compared later, rather than picking one too early.

Tokenizer files:
```
tokenizer_vocab500.json
tokenizer_vocab1000.json
tokenizer_vocab2000.json
tokenizer_vocab5000.json
```

## Step 2: N-gram Counts

After tokenizing, I built raw N-gram counts separately for each book. Three orders were used:

- **Unigram** — `P(w)`, no context
- **Bigram** — `P(w_i | w_{i-1})`, one token of context
- **Trigram** — `P(w_i | w_{i-2}, w_{i-1})`, two tokens of context

Counts were generated separately for *The Hobbit* and *The Lost World*, and no N-grams were allowed to cross the boundary between the two books.

I used sparse `Counter` structures instead of dense matrices, since most possible N-gram combinations never actually occur.

Saved as: `ngram_counts.pkl`

Structure: `vocab_size → book → N-gram order → counts`

## Step 3: Probabilities and Add-k Smoothing

Raw counts on their own don't answer the real question: how likely is the next token given the previous context? So Step 3 converts counts into probabilities, e.g. `P(the | of)` instead of just a raw count of how often "of the" appeared.

**The problem:** some N-grams never show up in training. Without smoothing, those get probability zero, which breaks log-probability calculations later.

**The fix:** Add-k smoothing.

```
P(w | context) = (count(context, w) + k) / (count(context) + kV)
```

where `k` is the smoothing parameter and `V` is the vocabulary size.

For the main experiment I used `k = 1` (Add-1 / Laplace smoothing).

Saved as: `lm_probs.pkl`, also using sparse storage. I checked that probabilities were properly normalized, and this passed for every vocabulary size, both books, and all three N-gram orders.

## Step 4: NLL and Perplexity Evaluation

Each book was scored under both author models, at every vocab size and every N-gram order:

```
4 vocab sizes × 2 texts × 2 author models × 3 N-gram orders = 48 evaluations
```

**Negative Log-Likelihood (NLL)** measures how well a model explains a sequence:

```
NLL = -Σ log P(token_i | context_i)
```

Lower NLL means the model assigned higher probability to the actual text.

**Perplexity** normalizes NLL into something easier to compare:

```
Perplexity = exp(NLL / number of predictions)
```

Lower perplexity means the model was less surprised. As one example, for the Hobbit text:

```
Hobbit model PPL:      69.14
Lost World model PPL: 128.93
```

Since the Hobbit model scored lower, the text gets predicted as Tolkien's.

Saved as: `step4_results.csv`, `step4_results.pkl`

All sanity checks passed: NLL values were non-negative, perplexity was always ≥ 1, everything was finite, prediction counts matched expected N-gram lengths, and probabilities stayed above zero thanks to smoothing.

## Step 5: Author Identification

For every text, vocab size, and N-gram order, I compared perplexity under the Hobbit model vs the Lost World model and predicted whichever was lower. No separate classifier, just a direct comparison.

```
4 vocab sizes × 2 texts × 3 N-gram orders = 24 decisions
```

**Result: 24 / 24 correct (100% accuracy)**

| Vocabulary Size | Correct |
|---|---|
| 500 | 6 / 6 |
| 1,000 | 6 / 6 |
| 2,000 | 6 / 6 |
| 5,000 | 6 / 6 |

| N-gram Order | Correct |
|---|---|
| Unigram | 8 / 8 |
| Bigram | 8 / 8 |
| Trigram | 8 / 8 |

Saved as: `step5_results.csv`, `step5_results.pkl`

## Observations

**Vocabulary size changes representation, but bigger isn't automatically better.** Smaller vocabularies split words into more pieces, larger ones keep more words whole. I kept all four sizes rather than assuming the largest would perform best.

**More context isn't automatically better either.** Trigrams use more context than bigrams, but that also means more possible combinations, which makes the counts sparser.

**Smoothing mattered.** Without it, unseen N-grams would have zero probability and break the log-probability math. Add-1 smoothing gave them a small non-zero probability instead.

**NLL and perplexity are two views of the same thing**, perplexity is just `exp(average NLL)`, so it's easier to interpret while NLL stays closer to the raw math.

**Each book consistently scored lower perplexity under its own author's model**, across every vocab size and N-gram order tested, which is what let this simple approach actually separate the two authors.

## Challenges

- **BPE decoding** wasn't cleanly readable out of the box because of whitespace/BPE config. This didn't affect the modeling itself since everything runs on token IDs, but I added a small cleanup function to make manual inspection easier.
- **Large N-gram spaces**: as vocab size grew, the number of possible N-gram combinations grew too. Dense matrices would've wasted a ton of memory, so I stuck with sparse `Counter`/dict storage.
- **Unseen N-grams** needed smoothing, covered in Step 3.
- **Smoothing values**: I considered testing different `k` values beyond 1, but kept the main results at `k = 1` for one consistent setting.

## Project Structure

```
author-identification/
│
├── data/
│   ├── hobbit.txt
│   ├── lost_world.txt
│   └── combined_corpus.txt
│
├── step1_bpe_tokenizer.py
├── step1b_tokenizer_analysis.py
├── step2_ngram_counts.py
├── step3_addk_smoothing.py
├── step4_evaluation.py
├── step5_author_identification.py
│
├── tokenizer_vocab500.json
├── tokenizer_vocab1000.json
├── tokenizer_vocab2000.json
├── tokenizer_vocab5000.json
│
├── ngram_counts.pkl
├── lm_probs.pkl
│
├── step4_results.csv
├── step4_results.pkl
├── step5_results.csv
└── step5_results.pkl
```

The `.venv` environment and Python cache files are not included.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tokenizers
```

Then run each step in order:

```bash
python step1_bpe_tokenizer.py
python step1b_tokenizer_analysis.py
python step2_ngram_counts.py
python step3_addk_smoothing.py
python step4_evaluation.py
python step5_author_identification.py
```

Each script produces files that the next step depends on.

## Limitation

The 100% accuracy here is specific to these two books and this setup. The models were trained and evaluated on the same books, so this shouldn't be read as a general claim about author identification accuracy on new, unseen writing.

A stronger test would use held-out passages or other works by the same authors that weren't part of training.

## Final Summary

```
STEP 1: BPE Tokenization
      ↓
STEP 2: Unigram / Bigram / Trigram Counts
      ↓
STEP 3: Probabilities + Add-1 Smoothing
      ↓
STEP 4: NLL + Perplexity
      ↓
STEP 5: Author Identification
```

The final experiment covered 4 vocabulary sizes × 3 N-gram orders × 2 books, producing 24 author identification decisions with 24 correct predictions in this experiment.

The main takeaway: even simple statistical N-gram models can pick up on differences in how different authors use words and short sequences, without needing neural networks or large language models.

---

## AI Usage Disclosure

I used Claude Code (Anthropic) as a coding assistant while working on this assignment.

- **Development support:** I used Claude Code to help implement parts of the Python code based on the approach, requirements, and step-by-step instructions I provided. I also used it to help debug issues and explain parts of the implementation when needed.
- **My contribution:** I made the main decisions about the project workflow, BPE vocabulary sizes, tokenizer setup, N-gram structure, smoothing approach, evaluation process, and how the final author identification was performed. I also decided what to test and how to interpret the results.
- **Review and verification:** I reviewed the generated and modified code, ran the scripts myself, checked the intermediate outputs, and verified that the results matched the expected behavior at each stage.
- **Understanding:** I used Claude Code as a programming aid, but I worked through the concepts and logic behind each step and made changes when the implementation did not match what I wanted or what the assignment required.
