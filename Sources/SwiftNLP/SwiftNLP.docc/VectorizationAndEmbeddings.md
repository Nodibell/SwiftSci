# Text Feature Vectorization, Embeddings & Naive Bayes Classification

Convert text corpora into numerical feature matrices using `CountVectorizer`, `HashingVectorizer`, and `TFIDFVectorizer`, query `WordEmbeddings` and `AppleNLEmbedding`, and train Naive Bayes classifiers with `TextPipeline`.

## Overview

Extract term frequency and inverse document frequency matrices for downstream classification and clustering.

### 1. TF-IDF Vectorizer & Naive Bayes Classification

```swift
import SwiftNLP

let corpus = [
    "soccer match goal score penalty stadium",
    "football player tournament champion world cup",
    "stock market economy investment finance bank",
    "trading stocks profit dividend inflation interest"
]

// Extract TF-IDF features
let vectorizer = TFIDFVectorizer()
try await vectorizer.fit(corpus)
let X = try await vectorizer.transform(corpus)

// Train actor-based NaiveBayesClassifier
let labels: [Double] = [0.0, 0.0, 1.0, 1.0] // 0: sports, 1: finance
let nb = NaiveBayesClassifier(alpha: 1.0)
try await nb.fit(features: X, targets: labels)

let testDoc = try await vectorizer.transform(["goal score match"])
let prediction = try await nb.predict(features: testDoc)
print("Prediction: \(prediction.first ?? 0)") // 0
```

### 2. Word Embeddings (Accelerated vDSP Cosine Similarity)

```swift
let embeddings = try WordEmbeddings(fromTextFile: "embeddings.txt")
let similarity = embeddings.similarity("king", "queen")

let nearest = embeddings.topK(vector: queryVector, k: 5)
```

### 3. Fluent DataFrame Text Processing Extensions

```swift
import SwiftDataFrame

var df = try DataFrame(columns: [
    TypedColumn<String>(name: "text", values: ["I love SwiftSci!", "This error is bad."])
])

// Chain NLP operations directly on DataFrames
let processedDF = try df
    .tokenizeColumn(column: "text", targetColumn: "tokens")
    .stemColumn(column: "tokens", targetColumn: "stemmed")
    .analyzeSentiment(column: "text", targetColumn: "sentiment")
```
