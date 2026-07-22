# Part V — Coding Standards

All code written for SwiftAnalytics must follow these standards to ensure safety, performance, and readability across the package.

---

## Naming & API Guidelines

We adhere strictly to the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).

### 1. Clarity at the Point of Use
All methods and functions must read like grammatical English phrases when invoked:
* **Good**: `df.filter(where: "age > 30")` or `pca.fit(matrix)`
* **Bad**: `df.runFilter(col: "age", val: 30)` or `pca.computePCA(X: matrix)`

### 2. Parameter Labels
Omit parameter labels when the arguments are clear from context. Use descriptive labels otherwise:
* **Good**: `linearRegression.predict(features)`
* **Good**: `KMeans(nClusters: 5, device: .gpu)`

### 3. Swift 6 Sendable Conformance
Public API structures and classes must be `Sendable` where appropriate:
* Use `struct` for immutable data representation (which are implicitly `Sendable`).
* Use `actor` for classes that manage mutable state (e.g., training states, model instances).
* Never mark a class as `@unchecked Sendable` unless you encapsulate all operations inside synchronization blocks (like `OSAllocatedUnfairLock`).

---

## DocC Documentation

Every public symbol must be documented using Swift's **DocC** format.

### DocC Comment Standards
* **Summary Line**: A single, concise sentence explaining the symbol's purpose.
* **Parameters List**: Document every parameter using the `- Parameter [name]:` markup.
* **Returns**: Document the return type and meaning using `- Returns:`.
* **Throws**: List all possible error types thrown using `- Throws:`.

Example:
```swift
/// Fits the Principal Component Analysis (PCA) model on the given dataset.
///
/// - Parameter X: A 2D array of shape [samples, features] representing the input dataset.
/// - Throws: `ClusterError.emptyInput` if the dataset is empty, or `ClusterError.invalidParameter` if components exceed dimensions.
public func fit(_ X: [[Double]]) async throws {
    // Implementation
}
```

---

## Actors & Protocols

### Actor Isolation Patterns
Algorithms that maintain state during training (such as `PCA`, `KMeans`, and `ARIMAModel`) must be implemented as Swift `actor` types. This ensures that concurrent tasks cannot mutate internal weights, variables, or resolution parameters during fitting operations.

### Standard Protocols
To maintain compatibility across modules (such as `SwiftOptimize`'s cross-validators), models must conform to our standard estimator protocols:

```swift
/// Protocol for supervised classification models.
public protocol ClassifierEstimator: Actor {
    /// Fits the classifier on the training features and labels.
    func fit(_ X: [[Double]], y: [Int]) async throws
    
    /// Predicts the labels for the given dataset.
    func predict(_ X: [[Double]]) async throws -> [Int]
}

/// Protocol for supervised regression models.
public protocol RegressorEstimator: Actor {
    /// Fits the regressor on the training features and labels.
    func fit(_ X: [[Double]], y: [Double]) async throws
    
    /// Predicts numerical targets for the given dataset.
    func predict(_ X: [[Double]]) async throws -> [Double]
}
```

---

## Testing & Benchmarking

### Writing Unit Tests
* Every new feature or bug fix must be accompanied by comprehensive tests in the corresponding test target (e.g., `Tests/SwiftMLTests`).
* Tests targeting concurrency must be run with the Swift 6 compiler checks active.
* Use `async/await` in tests when interacting with actors:
```swift
func testPCAFit() async throws {
    let pca = try PCA(nComponents: 2)
    let X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    try await pca.fit(X)
    let resolved = await pca.resolvedDevice
    XCTAssertNotNil(resolved)
}
```

### Adding Benchmarks
* If you implement a new algorithm, you must add a corresponding benchmark class to `Benchmarks/Swift`.
* Ensure that the benchmark uses the same data generator and setup as the Python baseline suite to make comparisons accurate.

---

## Generics & Unsafe Code

### Generics
Use generics when writing algorithms that operate on multiple numeric types (such as `Float` and `Double`). Constrain generic parameters to `BinaryFloatingPoint` or `Real` from the Swift Numerics library:
```swift
public func computeMean<T: BinaryFloatingPoint>(_ array: [T]) -> T {
    // Implementation
}
```

### Unsafe Code Boundaries
Unsafe code (e.g., `UnsafePointer`, `UnsafeMutableRawPointer`) must be localized and encapsulated. It is primarily permitted when passing contiguous arrays to Accelerate LAPACK functions:
```swift
var info: Int32 = 0
var lwork = Int32(-1)
var workQuery = [Double](repeating: 0.0, count: 1)

// Localizing unsafe access
vt.withUnsafeMutableBufferPointer { vtPtr in
    s.withUnsafeMutableBufferPointer { sPtr in
        // Call LAPACK function
    }
}
```
* **Rule**: Never expose unsafe pointers in the public API. Keep all unsafe interfaces `fileprivate` or `internal`.

---

## SIMD & Memory Management

### SIMD
Where vectorization is required on the CPU and `vDSP` does not provide a direct function, use Swift's native `SIMD` vectors (such as `SIMD4`, `SIMD8`, `SIMD16`):
```swift
// Vectorized dot product using SIMD
func dotProduct(_ a: SIMD4<Double>, _ b: SIMD4<Double>) -> Double {
    return (a * b).wrappedSum()
}
```

### Memory Management
Avoid frequent allocations inside tight loops. Use `inout` parameters to modify buffers in-place:
```swift
// Good: Mutates buffer in-place
func scaleBuffer(_ buffer: inout [Double], factor: Double) {
    vDSP_vsmulD(buffer, 1, &factor, &buffer, 1, vDSP_Length(buffer.count))
}

// Bad: Allocates a new array on every call
func scaleBuffer(_ buffer: [Double], factor: Double) -> [Double] {
    return buffer.map { $0 * factor }
}
```
