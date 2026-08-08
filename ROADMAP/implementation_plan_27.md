# Implementation Plan 27 — SwiftSci 2.7 Final Consolidation & Feature Maturity

This plan completes release **2.7.0** before freezing the SwiftSci 3.0 API, addressing the remaining `SwiftPreprocessing` concurrency refactoring (`MinMaxScaler`/`StandardScaler` value-semantics), bringing `WordNet` into `SwiftNLP`, upgrading `ONNXExporter` to Protobuf binary serialization, and updating release documentation.

## Proposed Changes

### 1. `SwiftPreprocessing` Value Semantics Refactoring
- **MinMaxScaler.swift**: Refactor from `final class: PreprocessingTransformer, @unchecked Sendable` to `struct MinMaxScaler: PreprocessingTransformer, Sendable`.
- **StandardScaler.swift**: Refactor from `final class: PreprocessingTransformer, @unchecked Sendable` to `struct StandardScaler: PreprocessingTransformer, Sendable`.
- **PreprocessingTests.swift**: Update test calls to use `var scaler = MinMaxScaler()`.

### 2. `SwiftNLP` WordNet Synset & Similarity Engine
- **WordNet.swift**: Add synset lookup, hypernyms/hyponyms tree, and path/Wu-Palmer similarity algorithms.
- **WordNetTests.swift**: Unit tests for WordNet.

### 3. `SwiftML` ONNX Protobuf Binary Serialization
- **ONNXExporter.swift**: Upgrade exporter to generate valid ONNX binary ModelProto payloads.

### 4. Release Documentation
- **CHANGELOG.md**: Add `## [2.7.0] - 2026-08-07` entry summarizing 2.7 release.
