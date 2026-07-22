# Part VII — Contribution Guide

We welcome contributions to SwiftAnalytics. This guide outlines the steps needed to add new algorithms, write matching tests/benchmarks, and format your code for pull requests.

---

## Adding Algorithms

When implementing a new algorithm (e.g., a classifier, clustering method, or preprocessor):

### 1. Identify the Correct Module
Ensure your code is added to the correct target directory:
* `SwiftStats`: General statistical distributions, hypothesis tests.
* `SwiftPreprocessing`: Transformers that scale, encode, or impute features.
* `SwiftML`: Supervised classification or regression models.
* `SwiftCluster`: Unsupervised clustering or dimensionality reduction.
* `SwiftForecast`: Time series processing, filtering, or forecasting models.

### 2. Follow State Isolation Rules
If the algorithm maintains state (such as fitted weights, means, or coefficients), implement it as an `actor`:
```swift
public actor MyNewClassifier: ClassifierEstimator {
    // Model parameters
    public private(set) var weights: [Double]?
    
    public init() {}
    
    public func fit(_ X: [[Double]], y: [Int]) async throws {
        // Compute and assign weights safely
    }
}
```

### 3. Handle Resource Limits
If your algorithm makes heavy use of GPU memory or allocates large MLX arrays, run it through the memory ticket system:
```swift
let result = try await WiredMemoryManager.shared.withTicket {
    // Heavy MLX GPU operations here
}
```

---

## Writing Tests

Every new algorithm must have matching unit tests.

### 1. Test Location
Add your test class to the appropriate test suite under the `Tests/` directory:
* Example: `Tests/SwiftMLTests/MyNewClassifierTests.swift`

### 2. Implementation Rules
* **Test Boundary Cases**: Test with empty arrays, NaN values, mismatched input dimensions, and negative values. Ensure the code throws the correct error type rather than crashing.
* **Verify Concurrency Safety**: Ensure that multiple asynchronous calls to the actor do not cause deadlocks or data corruption.
* **Verify against Reference Implementation**: Compare your algorithm's output against a reference implementation (such as Scikit-Learn or NumPy) using a small test dataset.

Example test method:
```swift
func testMyClassifierDivergence() async {
    let clf = MyNewClassifier()
    // Intentionally pass mismatched inputs
    do {
        try await clf.fit([[1.0]], y: [1, 2]) // Mismatch
        XCTFail("Should have thrown dimension mismatch error")
    } catch let error as MLError {
        XCTAssertEqual(error, .dimensionMismatch)
    } catch {
        XCTFail("Unexpected error thrown: \(error)")
    }
}
```

---

## Writing Benchmarks

If you add a major algorithm, you should include a corresponding test script in our benchmark target:

### 1. Swift Benchmark
Open `Benchmarks/Swift/main.swift` (or the appropriate benchmark suite) and add your algorithm. Record the execution time using the standard time measurement block:
```swift
let start = ContinuousClock.now
try await myAlgorithm.run()
let duration = start.duration(to: .now)
print("MyAlgorithm duration: \(duration)")
```

### 2. Python Baseline
Add the equivalent implementation in the corresponding Python benchmark file. Verify that the parameters (e.g., number of trees, iterations, dataset dimensions) match the Swift test exactly.

---

## PR Checklist

Before submitting a Pull Request, verify that you have completed the following steps:

- [ ] **Strict Concurrency Compilation**: Code compiles without warnings under strict concurrency checks. Run `swift build -c release` to verify.
- [ ] **All Tests Pass**: Run `swift test` locally and verify that all unit tests complete successfully.
- [ ] **DocC Cover**: Run `swift package generate-documentation` and ensure that all new public types and methods have complete DocC comments.
- [ ] **Formatting Guidelines**: Check that variable names and function signatures conform to the Swift API Design Guidelines.
- [ ] **Changelog Update**: Add a brief summary of your changes to `CHANGELOG.md` under the `Unreleased` or current minor version header.
