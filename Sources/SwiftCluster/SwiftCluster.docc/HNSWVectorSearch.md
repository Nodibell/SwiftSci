# High-Dimensional Vector Search with HNSW

Scale approximate nearest neighbor (ANN) retrieval logarithmically using Hierarchical Navigable Small World graphs.

## Overview

When vector collections expand beyond tens of thousands of items, exact \(O(N)\) linear scans become a major latency bottleneck for real-time recommendation and retrieval-augmented generation (RAG) systems.

The `HNSWIndex` actor provides a hierarchical graph-based spatial partitioning index with sub-millisecond query latency and logarithmic \(O(\log N)\) search scaling.

## Hierarchical Navigable Small World Architecture

HNSW constructs a hierarchy of graph layers:
- **Top Layers (Sparse)**: Contain widely spaced long-range highway edges for rapid coarse geometric routing across distant clusters.
- **Intermediate Layers**: Progressively decrease edge length while increasing vertex density.
- **Bottom Layer (Layer 0, Dense)**: Contains all indexed vertices connected by dense short-range edges for high-accuracy localized nearest-neighbor refinement.

```
Layer 2 (Express):   [A] ───────────────────────────► [F]
                      │                                │
                      ▼                                ▼
Layer 1 (Regional):  [A] ──────────► [C] ────────────► [F]
                      │               │                │
                      ▼               ▼                ▼
Layer 0 (Base):      [A] ──► [B] ──► [C] ──► [D] ──► [E] ──► [F]
```

## Key Configuration Parameters

| Parameter | Recommended Default | Impact on Search Quality & Index Build Time |
| :--- | :--- | :--- |
| `M` | `16` | Maximum outgoing connections per node. Higher values improve recall at the cost of higher memory and index construction time. |
| `efConstruction` | `200` | Size of dynamic candidate evaluation queue during insertion. Larger values produce higher-quality graph connectivity. |
| `efSearch` | `50` | Dynamic queue size during search traversal. Higher values increase search recall towards 100% exact match. |
| `metric` | `.euclideanDistance` | Choice of `.euclideanDistance`, `.cosineSimilarity`, or `.dotProduct`. |

## Example Usage

```swift
import SwiftCluster

// 1. Initialize the actor-isolated index
let hnsw = HNSWIndex(metric: .cosineSimilarity, M: 16, efConstruction: 200, efSearch: 50)

// 2. Add embeddings asynchronously
await hnsw.add(id: "doc_news_1", vector: [0.12, 0.94, -0.31], metadata: ["category": "news"])
await hnsw.add(id: "doc_tech_2", vector: [-0.85, 0.11, 0.52], metadata: ["category": "technology"])

// 3. Query Top-K nearest neighbors
let queryEmbedding = [0.10, 0.90, -0.30]
let matches = await hnsw.search(query: queryEmbedding, topK: 5)

for match in matches {
    print("Matched: \(match.id) with score: \(match.score)")
}
```

## Thread Safety & Swift 6 Strict Concurrency

`HNSWIndex` is implemented as an isolated Swift `actor`, guaranteeing thread safety and eliminating race conditions when multiple background tasks insert items or execute queries concurrently.

## Topics

### Spatial Indexing
- ``HNSWIndex``
- ``VectorMetric``
- ``VectorSearchResult``
