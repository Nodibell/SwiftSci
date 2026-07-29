# Text Normalization & Tokenization

Clean text, strip punctuation, remove English and Ukrainian stopwords, and tokenize using BPETokenizer and NGramTokenizer.

## Overview

Prepare raw unstructured text for machine learning pipelines.

### 1. Text Normalization

```swift
import SwiftNLP

let text = "SwiftSci 2.2 — Це потужний фреймворк для аналізу даних!"
let normalized = TextNormalizer.normalize(text, lowercase: true, stripPunctuation: true)
print("Normalized: \(normalized)")
```

### 2. Byte-Pair Encoding (BPE)

```swift
let bpe = BPETokenizer(vocabSize: 500)
bpe.train(corpus: [text])
let tokens = bpe.encode(text)
print("BPE Tokens: \(tokens)")
```
