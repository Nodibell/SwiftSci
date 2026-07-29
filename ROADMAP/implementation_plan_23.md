# SwiftSci 2.3.0 Implementation Plan: SIMD Acceleration, FFT Spectral Engines, Vision Deep Learning & PostgreSQL Driver

---

## 🎯 Strategic Goals & Architecture

**SwiftSci 2.3.0** focuses on closing remaining performance gaps against C/Fortran primitives in Python (Pandas/NumPy/Statsmodels), expanding deep learning inference capabilities in `SwiftVision`, and delivering native enterprise database integration in `SwiftDatabase`.

---

## 📅 Version 2.3.0 Phases Overview

| Phase | Target Module | Scope & Objectives | Key Performance Goal |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `SwiftDataFrame` | **SIMD Bitmask Boolean Filtering**: Bit-packed UInt64 mask indexing & vector gathers | Close **0.02× gap** → target **>1.0× vs Pandas** |
| **Phase 2** | `SwiftDataFrame` | **vDSP Radix Sort Indexing**: Primitive array `vDSP.sort` for `sortBy` | Close **0.10× gap** → target **>1.0× vs Pandas** |
| **Phase 3** | `SwiftForecast` | **vDSP FFT & 1D Convolution**: Fast Fourier Transform additive/multiplicative decomposition | Close **0.09× gap** → target **>1.0× vs Statsmodels** |
| **Phase 4** | `SwiftVision` | **Real U-Net Segmentation & YOLOv8 Detector**: Complete Metal/vDSP forward pass & NMS | Real segmentation masks & bounding box predictions |
| **Phase 5** | `SwiftDatabase` | **PostgreSQL Native Driver**: Binary protocol connection & DataFrame ingestion | `PostgreSQLConnection.execute` zero-copy ingestion |
| **Phase 6** | `SwiftML` & `SwiftCluster` | **Accelerate SIMD Tree Split Evaluation & GPU Offloading**: Parallel GBDT split search | 2–3× speedup on large tree ensembles |
| **Phase 7** | `SwiftSci` | **Comprehensive Verification, Benchmarks & DocC Update**: 100% test suite & release notes | Verified zero-regression release |

---

## 🔍 Detailed Plan by Phase

### Phase 1: SIMD Bitmask Boolean Filtering (`SwiftDataFrame`)
- **Target File**: [DataFrame.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Sources/SwiftDataFrame/Core/DataFrame.swift)
- **Problem**: `DataFrame.filter` evaluates row closures item-by-item, introducing boundary checks and closure call overhead.
- **Solution**:
  - Implement `TypedColumn.evaluatedIndices(where:)` producing contiguous `[Int32]` array using SIMD vector comparisons (`vDSP_vcmpeqD` / `vDSP_vcmpgtD`).
  - Introduce `DataFrame.gathered(atIndices: [Int32])` leveraging raw pointer copy semantics (`UnsafeBufferPointer`).

---

### Phase 2: Primitive Array vDSP Radix Sort Indexing (`SwiftDataFrame`)
- **Target File**: [DataFrame.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Sources/SwiftDataFrame/Core/DataFrame.swift)
- **Problem**: `sortBy` uses indirect array sorting with Swift's generic `sort(by:)`.
- **Solution**:
  - Implement `vDSP.sort` permutation index generation (`vDSP_vsortD` / `vDSP_vsortiD`).
  - Re-order contiguous Arrow column buffers in parallel using `DispatchQueue.concurrentPerform`.

---

### Phase 3: vDSP FFT & 1D FIR Convolution (`SwiftForecast`)
- **Target File**: `Sources/SwiftForecast/Core/Decomposition.swift`
- **Problem**: `TSDecomposition` uses naive sliding window loops for moving average trends.
- **Solution**:
  - Implement `vDSP_convD` 1D FIR filtering for moving averages.
  - Implement Accelerate `vDSP_fft_zripD` (Real Fast Fourier Transform) for seasonal frequency extraction.

---

### Phase 4: U-Net Segmentation & YOLOv8 Object Detection (`SwiftVision`)
- **Target Files**:
  - [ImageDataset.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Sources/SwiftVision/ImageDataset.swift)
  - `Sources/SwiftVision/Core/UNetSegmentationModel.swift`
  - `Sources/SwiftVision/Core/YOLOv8Detector.swift`
- **Solution**:
  - Build U-Net forward pipeline with Conv2D, BatchNorm, ReLU, and Skip Connections using `MLX` / `Accelerate`.
  - Implement YOLOv8 Anchor-Free Head with Non-Maximum Suppression (NMS) for overlapping bounding box filtering.

---

### Phase 5: PostgreSQL Native Binary Connection Driver (`SwiftDatabase`)
- **Target File**: [DatabaseConnection.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Sources/SwiftDatabase/DatabaseConnection.swift)
- **Solution**:
  - Replace `PostgreSQLConnection.notImplemented` with native socket binary protocol query parser.
  - Convert PostgreSQL binary row tuples directly into `TypedColumn` instances.

---

### Phase 6: Tree Split Evaluation SIMD Vectorization (`SwiftML`)
- **Target File**: [DecisionTree.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Sources/SwiftML/Core/DecisionTree.swift)
- **Solution**:
  - Vectorize gain evaluation across pre-sorted split points using `vDSP_vsumD` prefix sums.
  - Reduce decision tree split finding complexity by another **2.5×**.

---

### Phase 7: Benchmarks, Verification & DocC Updates
- Run release build: `swift build -c release`
- Execute test suite across all 16 workspace modules: `swift test`
- Run native benchmark comparison suite: `python3 Benchmarks/Python/compare.py`
- Generate DocC archives: `swift package generate-documentation`
