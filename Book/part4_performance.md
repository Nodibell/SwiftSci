# Part IV — Performance

Performance is a fundamental design goal of SwiftAnalytics. We target **bare-metal efficiency** by tailoring our software architecture to the unique hardware traits of Apple Silicon M-series chips.

---

## Why Accelerate & MLX?

Rather than writing custom low-level math kernels, SwiftAnalytics delegates matrix and vector calculations to two primary engines:

```
                  ┌───────────────────────────────┐
                  │        SwiftAnalytics         │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      ┌───────────────────────┐       ┌───────────────────────┐
      │  Accelerate (vDSP)    │       │     MLX Framework     │
      ├───────────────────────┤       ├───────────────────────┤
      │ • CPU execution       │       │ • GPU execution       │
      │ • Low branch overhead │       │ • Lazy evaluation     │
      │ • AMX optimization    │       │ • Metal compilation   │
      └───────────────────────┘       └───────────────────────┘
```

### Apple Accelerate
The `Accelerate` framework contains highly optimized functions for vector processing (`vDSP`), signal processing (`vForce`), and linear algebra (`LAPACK` / `BLAS`). Apple maintains Accelerate, ensuring that its functions are custom-tailored to the registers, caches, and instruction sets of every Apple Silicon chip generation.

### MLX
Developed by Apple's machine learning research division, `MLX` is an array framework designed specifically for Apple Silicon. It supports:
* **Lazy Evaluation**: Computations are recorded as a graph and only evaluated when needed. This allows MLX to merge multiple operations (e.g., scale + add + activation) into a single Metal GPU shader, reducing memory round-trips.
* **GPU Vectorization**: Operations run directly on the Unified Memory GPU, taking full advantage of massive parallel execution pipelines.

---

## Why LAPACK?

We use standard Linear Algebra Package (`LAPACK`) interfaces (e.g., `dgesvd` for SVD, `dgelsd` for least squares regression) because they are the industry standard for stable, high-performance matrix decompositions. In the Apple ecosystem, LAPACK is accelerated by Apple's CPU-based mathematical accelerators, guaranteeing optimal thread scheduling and cache line alignments.

---

## Why Not Eigen?

C++ implementations like Eigen are popular in Unix/Windows ecosystems. However, we choose not to include Eigen in SwiftAnalytics for several reasons:

1. **Compilation Overhead**: Eigen is a template-heavy C++ library. Incorporating it would require C++ integration inside the Swift Package, which increases compile times and complicates SPM target dependencies.
2. **Bridging Costs**: Passing data between Swift arrays and C++ Eigen matrices requires writing bridging headers and can introduce copying overhead, which violates our **Zero-Copy** principle.
3. **Optimized Alternatives**: Apple's native Accelerate framework matches or outperforms Eigen on M-series chips without the C++ baggage.

---

## Why Not Write BLAS Manually?

Writing custom Basic Linear Algebra Subprograms (BLAS) or matrix multiplication kernels in raw Swift is discouraged:
* **Hardware Co-processors**: Modern Apple Silicon chips contain a proprietary **Apple Matrix Coprocessor (AMX)**. The instruction set for AMX is private and not exposed to standard compiler pipelines.
* **Apple Optimization**: The only public, supported way to run code on the AMX is through Apple's `Accelerate` and `CoreML` frameworks. By using Accelerate, we automatically run on the AMX without needing to reverse-engineer Apple's hardware.

---

## Apple Silicon Architecture

To understand the speed of SwiftAnalytics, it is important to look at the hardware features of Apple Silicon:

### Unified Memory Architecture (UMA)
In a traditional computer, a CPU has its memory (RAM), and a GPU has its own memory (VRAM). When transferring a DataFrame from a CSV file (CPU) to a Machine Learning model (GPU), the system must copy the entire table over the PCIe bus:

```text
[CPU Memory] ──► (PCIe Bus: 16-64 GB/s Bottleneck) ──► [GPU Memory]
```

Under Apple's Unified Memory Architecture, the CPU and GPU share a single physical memory pool with bandwidths reaching up to **800 GB/s** (on M-series Max/Ultra chips). This allows for **zero-copy execution**:

```text
       ┌──────────────────────────────────────────────┐
       │           Unified Memory Pool                │
       │   (Shares the same physical memory space)    │
       ├──────────────────────┬───────────────────────┤
       │     CPU access       │      GPU access       │
       └──────────────────────┴───────────────────────┘
```

SwiftAnalytics takes full advantage of this. A DataFrame created in `SwiftDataFrame` (backed by Apache Arrow) can be read directly by an `MLXArray` tensor and processed on the GPU without a single memory copy.

### Apple Matrix Coprocessor (AMX)
The AMX is an accelerator block situated next to each CPU core. It is designed to perform large matrix multiplication, outer products, and vector operations at high speeds. By using Accelerate APIs like `cblas_dgemm`, SwiftAnalytics automatically targets the AMX, freeing up the CPU vector units (NEON) and the GPU for other concurrent workloads.
