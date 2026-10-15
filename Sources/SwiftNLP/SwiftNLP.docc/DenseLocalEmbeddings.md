# Dense Local Text Embeddings & Zero-Shot Classification

Generate high-dimensional, unit-normalized semantic embeddings 100% offline on Apple Silicon and perform zero-shot classification without model training.

## Overview

Modern Retrieval-Augmented Generation (RAG), semantic vector search, and automated data categorization require fast, high-quality vector representations of text. `SwiftNLP` provides a unified architecture via ``SentenceEmbeddingProvider``, offering two distinct engine implementations:

1. **``AppleSentenceEmbeddingEngine``**: Native 512-dimensional contextual semantic embeddings powered by Apple Natural Language and Core ML foundation models with Accelerate $L_2$ unit normalization.
2. **``LocalEmbeddingEngine``**: Deterministic, lightweight subword N-gram hashing projection producing customizable $D$-dimensional embeddings in microseconds with zero disk weight.

Both engines integrate seamlessly with ``ZeroShotTextClassifier`` for zero-shot text categorization and ``SwiftCluster.VectorStore`` for real-time similarity search.

```mermaid
graph TD
    Text[Raw Text / Document] --> Provider[SentenceEmbeddingProvider]
    Provider --> Apple[AppleSentenceEmbeddingEngine: 512-D Core ML]
    Provider --> Hash[LocalEmbeddingEngine: Subword N-Gram Projection]
    Apple --> SIMDNorm[Accelerate vDSP L2 Unit Normalization]
    Hash --> SIMDNorm
    SIMDNorm --> NormalizedVec[Unit Vector: ||v|| = 1.0]
    NormalizedVec --> VectorStore[(SwiftCluster VectorStore / HNSW)]
    NormalizedVec --> ZeroShot[ZeroShotTextClassifier Softmax Probabilities]
```

---

## 1. Choosing an Embedding Engine

| Feature | `AppleSentenceEmbeddingEngine` | `LocalEmbeddingEngine` |
|:---|:---|:---|
| **Underlying Mechanism** | Apple Natural Language / Core ML | Subword N-Gram Hashing (FNV-1a + Signs) |
| **Embedding Dimension** | Fixed 512-D | Configurable (e.g., 64, 128, 256, 512) |
| **Weights on Disk** | 0 MB (pre-installed system foundation model) | 0 MB (algorithmic hash projection) |
| **Semantic Awareness** | High (contextual, synonym & paraphrase aware) | Lexical / Syntactic (spelling, morphology, keyword) |
| **Typical Latency** | ~0.5 – 1.5 ms / text | ~0.05 – 0.15 ms / text |
| **Best Used For** | RAG retrieval, zero-shot classification, semantic search | High-throughput deduplication, fuzzy search, microcontrollers |

---

## 2. Using `AppleSentenceEmbeddingEngine`

The ``AppleSentenceEmbeddingEngine`` leverages Apple Silicon's Neural Engine to produce 512-dimensional contextual embeddings. All output vectors are automatically normalized to unit length ($\|v\|_2 = 1.0$) using Accelerate SIMD operations (`vDSP_svesqD` and `vDSP_vsdivD`):

```swift
import SwiftNLP

// Initialize native Apple sentence embedder
let engine = AppleSentenceEmbeddingEngine()

// Verify availability for current device and locale
guard engine.isAvailable(for: .english) else {
    fatalError("Apple sentence embedding model not available for English on this system.")
}

// Generate a 512-dimensional embedding
let embedding = try await engine.embed("Retrieval-Augmented Generation with SwiftSci")
print("Vector length: \(embedding.count)") // 512

// Batch asynchronous embedding generation
let documents = [
    "Apple Accelerate provides vectorized SIMD routines.",
    "Metal Performance Shaders accelerate tensor operations.",
    "SwiftData provides declarative SQLite persistence."
]
let batchVectors = try await engine.embedBatch(documents)
```

---

## 3. Using `LocalEmbeddingEngine`

For memory-constrained environments or extreme throughput requirements, ``LocalEmbeddingEngine`` computes embeddings via subword character n-gram projections without model weights:

```swift
import SwiftNLP

// Initialize with custom dimensionality (default: 128)
let fastEngine = LocalEmbeddingEngine(dimension: 128)

// Synchronous and asynchronous embedding
let vector = fastEngine.embed("Fast in-memory hashing projection")
let batch = try await fastEngine.embedBatch(["Document A", "Document B"])
```

---

## 4. Measuring Semantic Similarity

Because all embeddings produced by `SentenceEmbeddingProvider` are unit-normalized, the **Cosine Similarity** between two vectors $u$ and $v$ is mathematically identical to their direct **Dot Product**:

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$

Calculate cosine similarity at native memory speeds using Apple Accelerate:

```swift
import Accelerate

func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return 0.0 }
    var dot: Double = 0.0
    vDSP_dotprD(a, 1, b, 1, &dot, vDSP_Length(a.count))
    return dot
}

let queryVec = try await engine.embed("Vector mathematics on Mac")
let docVec = try await engine.embed("Apple Accelerate linear algebra")
let similarity = cosineSimilarity(queryVec, docVec)
print("Semantic similarity score: \(similarity)")
```

---

## 5. Zero-Shot Text Classification

Classify arbitrary text documents into dynamic candidate categories on the fly without training a classifier or labeling training sets:

```swift
import SwiftNLP

// Initialize classifier with Apple embeddings
let classifier = ZeroShotTextClassifier(
    embeddingEngine: AppleSentenceEmbeddingEngine(),
    hypothesisTemplate: "This text is about {}."
)

// Single-label classification (Softmax distribution)
let result = try await classifier.classify(
    text: "The patient presented with elevated systolic pressure and sinus tachycardia.",
    candidateLabels: ["Cardiology", "Astronomy", "Sports", "Politics"],
    multiLabel: false
)

print("Top label: \(result.topLabel ?? "") (confidence: \(result.topScore))")
for (label, score) in result.scores {
    print("  • \(label): \(String(format: "%.3f", score))")
}
```

### Multi-Label Classification

For documents that can belong to multiple categories simultaneously, set `multiLabel: true` to compute independent sigmoid activations:

```swift
let multiResult = try await classifier.classify(
    text: "Quarterly corporate earnings and federal interest rate decisions.",
    candidateLabels: ["Finance", "Government", "Culinary", "Gaming"],
    multiLabel: true
)
```

---

## 6. DataFrame Zero-Shot Categorization

Classify entire columns of tabular data directly within a ``SwiftDataFrame`` using `zeroShotClassify`:

```swift
import SwiftDataFrame
import SwiftNLP

var feedbackDF = DataFrame(columns: [
    TypedColumn(name: "FeedbackID", values: [1, 2, 3]),
    TypedColumn(name: "ReviewText", values: [
        "The app crashed when opening the PDF export screen.",
        "Monthly subscription pricing is too expensive.",
        "Smooth navigation and beautiful dark mode aesthetics."
    ])
])

// Classify ReviewText column into sentiment and topic categories
let classifiedDF = try await feedbackDF.zeroShotClassify(
    textColumn: "ReviewText",
    candidateLabels: ["Bug Report", "Pricing", "User Interface"],
    outputColumn: "Category"
)

print(classifiedDF.summary())
```

---

## Topics

### Embedding Engine Types
- ``SentenceEmbeddingProvider``
- ``AppleSentenceEmbeddingEngine``
- ``LocalEmbeddingEngine``

### Zero-Shot Classification
- ``ZeroShotTextClassifier``
- ``ZeroShotClassificationResult``
