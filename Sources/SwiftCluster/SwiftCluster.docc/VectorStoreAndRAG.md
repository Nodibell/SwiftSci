# In-Memory VectorStore & On-Device RAG

Build low-latency semantic search, embedding retrieval, and Retrieval-Augmented Generation (RAG) pipelines on Apple Silicon using SIMD Accelerate.

## Overview

Vector databases are essential for modern semantic search and AI agent knowledge retrieval. `SwiftCluster` provides `VectorStore`, a thread-safe in-memory vector index optimized with Apple Accelerate SIMD operations (`vDSP_dotprD`, `vDSP_svesqD`, `vDSP_distancesqD`).

## 1. Creating a VectorStore

```swift
import SwiftCluster
import SwiftNLP

// Initialize vector store with Cosine Similarity
let store = VectorStore(metric: .cosineSimilarity)

// Generate dense text embeddings
let engine = LocalEmbeddingEngine(dimension: 128)

let articles = [
    (id: "art_1", title: "Apple Silicon Architecture", text: "Unified memory and neural engine overview"),
    (id: "art_2", title: "Swift Strict Concurrency", text: "Data-race safety and Sendable types in Swift 6"),
    (id: "art_3", title: "Quantum Computing Basics", text: "Qubits and superposition principles")
]

for art in articles {
    let vec = engine.embed(art.text)
    store.add(id: art.id, vector: vec, metadata: ["title": art.title])
}

print("Total indexed vectors: \(store.count)")
```

## 2. Top-K Semantic Search

```swift
let queryVec = engine.embed("How does memory sharing work on M-series chips?")
let results = store.search(query: queryVec, topK: 2)

for hit in results {
    print("Found ID: \(hit.id), Score: \(hit.score), Title: \(hit.metadata["title"] ?? "")")
}
```

## 3. Supported Metrics

* **`.cosineSimilarity`**: Ideal for normalized text embeddings (returns `[-1.0, 1.0]`).
* **`.dotProduct`**: Direct inner product for unnormalized projections.
* **`.euclideanDistance`**: Geometric L2 distance for spatial clustering.

## Topics

### Vector Store Types
- ``VectorStore``
- ``VectorSearchResult``
- ``VectorMetric``
