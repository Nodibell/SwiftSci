# Text Feature Vectorization, Sparse TF-IDF & Local Embeddings

Convert text corpora into numerical and sparse feature matrices using `TFIDFVectorizer`, `CountVectorizer`, `HashingVectorizer`, `SparseVector`, and `LocalEmbeddingEngine`.

## Overview

Natural language processing begins with translating text tokens into numerical vectors. `SwiftNLP` provides dense and sparse vectorization pipelines with min/max document frequency filtering, sublinear TF scaling, custom stop-words, and hardware-accelerated cosine similarities via Apple Accelerate `vDSP`.

---

## 1. Memory-Efficient Sparse TF-IDF Vectorization

When processing large text corpora with vocabularies of $100,000+$ words, dense matrices waste gigabytes of RAM storing zeros. `SparseVector` stores only non-zero index-value pairs:

```swift
import Foundation
import SwiftNLP

// 1. Text corpus
let corpus = [
    "SwiftSci provides high performance scientific computing on Apple Silicon",
    "Machine learning models execute natively on unified memory architecture",
    "Vectorized statistical analysis and linear algebra with Apple Accelerate",
    "Natural language processing and tokenization in pure Swift 6"
]

// 2. Configure TFIDFVectorizer with custom stop words and sparsity constraints
let customStopWords: Set<String> = ["and", "with", "on", "in", "the"]
let vectorizer = TFIDFVectorizer(
    minDF: 1,
    maxDF: 0.9,
    sublinearTF: true,
    stopWords: customStopWords
)

// 3. Fit vocabulary and transform corpus into SparseVectors
try await vectorizer.fit(corpus)
let sparseMatrix: [SparseVector] = try await vectorizer.transformSparse(corpus)

print("Vocabulary Size: \(await vectorizer.vocabulary.count) terms.")
for (docIdx, sparseDoc) in sparseMatrix.enumerated() {
    print("Doc \(docIdx + 1): \(sparseDoc.indices.count) non-zero features out of \(sparseDoc.dimension) dimensions.")
}

// 4. Convert to dense vector on-demand when required
let denseDoc0: [Double] = sparseMatrix[0].toDense()
print("First 5 dense weights: \(denseDoc0.prefix(5))")
```

---

## 2. Naive Bayes Text Classification

Train an actor-isolated Naive Bayes classifier on vectorized text features:

```swift
import SwiftNLP

let trainingDocs = [
    "soccer match goal penalty referee stadium",
    "basketball tournament points rebound court",
    "stock market earnings investment dividend treasury",
    "inflation interest rates economy monetary policy"
]
let targets = [0.0, 0.0, 1.0, 1.0] // 0: Sports, 1: Finance

let docVectorizer = TFIDFVectorizer()
try await docVectorizer.fit(trainingDocs)
let trainFeatures = try await docVectorizer.transform(trainingDocs)

let classifier = NaiveBayesClassifier(alpha: 1.0)
try await classifier.fit(features: trainFeatures, targets: targets)

// Predict topic of unseen text
let testSample = try await docVectorizer.transform(["stocks and market economy"])
let prediction = try await classifier.predict(features: testSample)

print("Predicted Topic: \(prediction.first == 1.0 ? "Finance" : "Sports")")
```

---

## 3. Local On-Device Embeddings (`LocalEmbeddingEngine`)

Generate dense, 100% offline text embeddings for Retrieval-Augmented Generation (RAG) and semantic similarity search:

```swift
let embeddingEngine = LocalEmbeddingEngine(dimension: 128)

let queryVector = embeddingEngine.embed("Apple Silicon M-Series GPU computing")
let candidateVectors = embeddingEngine.embedBatch([
    "Unified Memory Architecture acceleration",
    "Cooking recipes and culinary arts",
    "Neural Engine transformer token generation"
])

print("Generated \(candidateVectors.count) dense vectors of dimension \(queryVector.count).")
```
