# Text Feature Vectorization

Convert text corpora into numerical feature matrices using CountVectorizer, HashingVectorizer, and TFIDFVectorizer.

## Overview

Extract term frequency and inverse document frequency matrices for downstream classification.

### 1. TF-IDF Vectorizer

```swift
import SwiftNLP

let corpus = [
    "SwiftSci is fast and native",
    "Machine learning in Swift for Apple Silicon"
]

let vectorizer = TFIDFVectorizer(ngramRange: (1, 2))
let matrix = try vectorizer.fitTransform(corpus)
```
