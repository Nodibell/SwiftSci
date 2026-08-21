# Clustering & Anomaly Detection

Cluster data points and detect outliers using KMeans++, DBSCAN, IsolationForest, and LocalOutlierFactor.

## Overview

Unsupervised algorithms for discovery of spatial structures and anomalous patterns.

### 1. KMeans++ Clustering

```swift
import SwiftCluster

let kmeans = KMeans(k: 3, maxIterations: 100)
try await kmeans.fit(X)

let labels = try await kmeans.predict(X)
print("Cluster Centers: \(kmeans.clusterCenters)")
```

### 2. Isolation Forest Anomaly Detection

```swift
let isoForest = IsolationForest(nEstimators: 100)
try await isoForest.fit(X)
let anomalyScores = try await isoForest.scoreSamples(X)
```

### 3. In-Memory VectorStore (Semantic Search & RAG)

```swift
let store = VectorStore(metric: .cosineSimilarity)
store.add(id: "doc_1", vector: embedding1, metadata: ["title": "Neural Networks"])
store.add(id: "doc_2", vector: embedding2, metadata: ["title": "Optimization Algorithms"])

let topMatches = store.search(query: queryEmbedding, topK: 5)
```
