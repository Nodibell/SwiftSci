# ``SwiftNLP``

Natural Language Processing & Text Feature Extraction.

## Overview

`SwiftNLP` provides Ukrainian and English text normalization, tokenizers, stopword filtering, and n-gram vectorization pipelines.

### Key Capabilities

- **Text Normalization**: `TextNormalizer` with lowercasing, punctuation stripping, and stopword removal.
- **Subword Tokenization**: `BPETokenizer` (Byte-Pair Encoding) and `NGramTokenizer`.
- **Vectorization Pipelines**: `CountVectorizer`, `HashingVectorizer`, and `TFIDFVectorizer`.
- **Multilingual Datasets**: Pre-built Ukrainian (`ua-news`) and English stopword lexicons.

### Example Usage

```swift
import SwiftNLP

let vectorizer = TFIDFVectorizer(ngramRange: (1, 2))
let X_text = try vectorizer.fitTransform(corpus)
```

## Topics

### Guides & Tutorials
- <doc:TextNormalizationAndTokenization>
- <doc:VectorizationAndEmbeddings>
