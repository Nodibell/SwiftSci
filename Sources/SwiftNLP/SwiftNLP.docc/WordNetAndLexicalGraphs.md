# WordNet Semantic Hierarchy & Lexical Graphs

Navigate semantic relationships, compute synset distances, and explore English hypernym/hyponym trees.

## Overview

`WordNet` is a large lexical database of English where nouns, verbs, adjectives, and adverbs are organized into sets of cognitive synonyms called **Synsets**, interlinked by conceptual-semantic and lexical relations.

## 1. Looking Up Synsets & Definitions

```swift
import SwiftNLP

let wordnet = WordNet.shared

// Look up all noun synsets for "bank"
let synsets = wordnet.synsets(for: "bank", pos: .noun)
for s in synsets {
    print("ID: \(s.id), Definition: \(s.definition)")
}
```

## 2. Semantic Similarity Metrics

SwiftSci implements standard lexical similarity algorithms:

* **Path Similarity**: Shortest path distance in the hypernym taxonomy:
  > **Formula:** `Sim_path(s1, s2) = 1 / (1 + distance(s1, s2))`

* **Wu-Palmer Similarity (WUP)**: Measures depth of least common subsumer (LCS) relative to synset depths:
  > **Formula:** `Sim_wup(s1, s2) = (2 · depth(LCS)) / (depth(s1) + depth(s2))`

```swift
let dog = wordnet.synset("dog.n.01")!
let cat = wordnet.synset("cat.n.01")!

let pathSim = wordnet.pathSimilarity(dog, cat)
let wupSim = wordnet.wuPalmerSimilarity(dog, cat)

print("Path Similarity: \(pathSim), Wu-Palmer: \(wupSim)")
```

## Topics

### WordNet Types
- ``WordNet``
- ``Synset``
- ``POS``
