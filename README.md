# N-Gram Author Identification

A classical NLP project that uses BPE tokenization and statistical N-gram language models to identify the author of a text.

The project compares J.R.R. Tolkien's *The Hobbit* with Arthur Conan Doyle's *The Lost World*. The main idea is simple: train a language model on each book, measure how well each model predicts a given text, and use perplexity to determine which author's writing style the text is closer to.

## Project Overview

The project follows this pipeline:

Raw text → BPE tokenization → N-gram counts → probabilities + smoothing → NLL/perplexity → author identification

This is a statistical language modeling project. It does not use an LLM, neural network, or RAG system.

The main questions explored were:

- How does BPE vocabulary size affect tokenization?
- How do unigram, bigram, and trigram models represent text?
- Why is smoothing needed for N-gram models?
- How does vocabulary size affect perplexity?
- Can a language model trained on one author's writing distinguish between two authors?

## Dataset

Two books were used:

- J.R.R. Tolkien, *The Hobbit*
- Arthur Conan Doyle, *The Lost World*

The books were kept as separate corpora when building the N-gram models. A combined version of the corpus was used only when training the shared BPE tokenizers.

The combined corpus contained approximately 928,569 characters.

## 1. BPE Tokenization

The first step was to create a shared Byte Pair Encoding (BPE) tokenizer using the combined text from both books.

Four vocabulary sizes were tested:

- 500
- 1,000
- 2,000
- 5,000

Using a shared tokenizer is important because both books need to use the same token vocabulary and token IDs. For example, if `"the"` is represented by token ID `86`, it should have the same ID in both books.

### Token counts

| Vocabulary Size | Combined Tokens |
|---:|---:|
| 500 | 342,711 |
| 1,000 | 292,964 |
| 2,000 | 256,397 |
| 5,000 | 223,904 |

As the vocabulary size increases, more words can be represented as complete tokens instead of being split into smaller pieces.

For example, with a smaller vocabulary, a word such as `walking` may be split into several subword tokens. With a larger vocabulary, the tokenizer may be able to represent the whole word as one token.

The larger vocabulary is not automatically considered better. All four settings were kept for later comparison.

## 2. N-Gram Counts

After tokenization, separate N-gram statistics were created for each book.

Three types of N-grams were used:

### Unigram

A unigram looks at one token at a time.

```text
P(w)
