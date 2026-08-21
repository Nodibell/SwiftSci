# Dense Local Text Embeddings with Accelerate

Generate high-dimensional, unit-normalized semantic embeddings 100% offline on Apple Silicon.

## Overview

Modern Retrieval-Augmented Generation (RAG) and semantic search systems require dense vector representations of text. Traditional approaches depend on remote APIs (e.g. OpenAI) or large multi-gigabyte models that exhaust memory. `SwiftNLP` introduces `LocalEmbeddingEngine`, a subword N-gram character hashing projection engine that produces deterministic, unit-normalized dense embeddings in sub-millisecond execution times.

## 1. Algorithmic Foundation

The `LocalEmbeddingEngine` operates through a three-stage mathematical pipeline:

1. **Subword N-gram Hashing**: Text is tokenized into word and character n-grams (sizes 3 to 6). Each n-gram is hashed into a fixed integer space using FNV-1a hash functions.
2. **Dimension Projection**: Hashes are projected onto a $D$-dimensional vector space (e.g., $D=128$ or $D=256$) with deterministic sign flips to prevent feature collision bias.
3. **Unit $L_2$ Normalization**: The resulting dense vector is projected onto the unit hypersphere:

$$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^{D} v_i^2}}$$

Because all embeddings have $\|v\|_2 = 1.0$, the **Cosine Similarity** between two embeddings is equivalent to their direct **Dot Product**, allowing single-instruction vector evaluation via Apple Accelerate `vDSP_dotprD`:

$$\text{CosineSimilarity}(u, v) = u \cdot v$$

## 2. Generating Embeddings

```swift
import SwiftNLP

// Initialize a 128-dimensional local embedding engine
let engine = LocalEmbeddingEngine(dimension: 128)

// Generate embedding for a single text snippet
let vector = engine.embed("Apple Silicon M3 Max neural network processing")
print("Vector dimension: \(vector.count)") // 128

// Batch embedding generation
let documents = [
    "High-performance computing on macOS",
    "Relational database wire protocol drivers",
    "Deep convolutional neural networks for vision"
]
let batchEmbeddings = engine.embedBatch(documents)
```

## 3. Integration with SwiftCluster.VectorStore

```swift
import SwiftCluster

let store = VectorStore(metric: .cosineSimilarity)

for (idx, doc) in documents.enumerated() {
    let vec = engine.embed(doc)
    store.add(id: "doc_\(idx)", vector: vec, metadata: ["text": doc])
}

// Perform instant semantic similarity search
let query = engine.embed("Mac hardware acceleration")
let topHits = store.search(query: query, topK: 1)
print("Top match: \(topHits.first?.metadata["text"] ?? "")")
```

## Topics

### Embedding APIs
- ``LocalEmbeddingEngine``
