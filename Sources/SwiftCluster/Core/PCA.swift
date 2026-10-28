#if os(macOS)
import Foundation
import Accelerate
import MLX
import SwiftPreprocessing
@_exported import SwiftDataFrame

/// SVD solver strategy for PCA on CPU.
public enum SVDSolver: Sendable {
    /// Full LAPACK divide-and-conquer SVD.
    case full
    /// Randomized SVD with the existing deterministic sketch and power iterations.
    case randomized
    /// Randomized SVD when k < 0.8 * min(rows, columns); full SVD otherwise.
    case auto
}

/// Principal component analysis with explicit CPU solver and device choices.
/// Prepared inputs retain feature identity and return named, provenance-preserving scores.
public actor PCA {
    /// Complete immutable projection state, including variance omitted by the selected components.
    public struct Parameters: Sendable {
        public let mean: [Double]
        public let components: [[Double]]
        public let explainedVariance: [Double]
        public let totalVariance: Double
        public let featureNames: [String]?

        /// Validates supplied coefficients without recomputing or normalizing them.
        /// Selected variance may exceed the total only within 1e-12 relative rounding.
        public init(mean: [Double], components: [[Double]], explainedVariance: [Double],
                    totalVariance: Double, featureNames: [String]? = nil) throws {
            guard !mean.isEmpty, mean.allSatisfy(\.isFinite), !components.isEmpty,
                  components.count <= mean.count,
                  components.allSatisfy({ $0.count == mean.count && $0.allSatisfy(\.isFinite) }),
                  explainedVariance.count == components.count,
                  explainedVariance.allSatisfy({ $0.isFinite && $0 >= 0 }),
                  totalVariance.isFinite, totalVariance >= 0,
                  featureNames == nil || featureNames?.count == mean.count else {
                throw ClusterError.invalidParameter("PCA parameters must have matching finite dimensions and nonnegative variances.")
            }
            let varianceMatchesTotal: Bool
            if totalVariance == 0 {
                varianceMatchesTotal = explainedVariance.allSatisfy { $0 == 0 }
            } else {
                let retainedFraction = explainedVariance.reduce(0) { $0 + $1 / totalVariance }
                varianceMatchesTotal = retainedFraction <= 1 + 1e-12
            }
            guard varianceMatchesTotal else {
                throw ClusterError.invalidParameter("Selected PCA variances cannot exceed total input variance.")
            }
            self.mean = mean
            self.components = components
            self.explainedVariance = explainedVariance
            self.totalVariance = totalVariance
            self.featureNames = featureNames
        }
    }

    /// Number of components to retain.
    public let nComponents: Int
    /// Solver used when fitting on the CPU.
    public let svdSolver: SVDSolver
    /// Requested fitting device. Projection uses Double CPU arithmetic.
    public let requestedDevice: ExecutionDevice

    private struct FittedState: Sendable {
        let result: PCAFitResult
        let device: ExecutionDevice
        let featureNames: [String]?
    }
    private var fitted: FittedState?

    /// Restores a CPU projection. A subsequent fit uses the full CPU solver.
    public init(parameters: Parameters) {
        nComponents = parameters.components.count
        svdSolver = .full
        requestedDevice = .cpu
        fitted = FittedState(result: PCAFitResult(mean: parameters.mean,
            components: parameters.components, variance: parameters.explainedVariance,
            totalVariance: parameters.totalVariance), device: .cpu,
            featureNames: parameters.featureNames)
    }

    /// Captures one successful fit atomically, including its ordered prepared schema.
    public func parameters() throws -> Parameters {
        guard let fitted else { throw ClusterError.fittingRequired }
        return try Parameters(mean: fitted.result.mean, components: fitted.result.components,
            explainedVariance: fitted.result.variance, totalVariance: fitted.result.totalVariance,
            featureNames: fitted.featureNames)
    }

    /// Device resolved for the last successful fit.
    public var resolvedDevice: ExecutionDevice? { fitted?.device }
    /// Fitted feature means.
    public var mean: [Double]? { fitted?.result.mean }
    /// Principal components, shaped [components, features].
    public var components: [[Double]]? { fitted?.result.components }
    /// Selected sample variances, normalized by max(1, samples - 1).
    public var explainedVariance: [Double]? { fitted?.result.variance }
    /// Selected variances divided by total input variance, including omitted components.
    public var explainedVarianceRatio: [Double]? {
        guard let result = fitted?.result else { return nil }
        return result.totalVariance > 0 ? result.variance.map { $0 / result.totalVariance }
            : result.variance.map { _ in 0 }
    }

    /// Creates PCA with automatic CPU solver selection.
    public init(nComponents: Int, device: ExecutionDevice = .auto) throws {
        guard nComponents > 0 else {
            throw ClusterError.invalidParameter("nComponents must be greater than 0.")
        }
        self.nComponents = nComponents
        self.svdSolver = .auto
        self.requestedDevice = device
    }

    /// Creates PCA with an explicit CPU solver. GPU fitting retains its Float MLX path.
    public init(nComponents: Int, svdSolver: SVDSolver, device: ExecutionDevice = .auto) throws {
        guard nComponents > 0 else {
            throw ClusterError.invalidParameter("nComponents must be greater than 0.")
        }
        self.nComponents = nComponents
        self.svdSolver = svdSolver
        self.requestedDevice = device
    }

    /// Fits finite, rectangular nested rows. A successful fit clears prepared feature names.
    /// Invalid input or a failed solver leaves the previous fit unchanged.
    public func fit(_ X: [[Double]]) async throws {
        try validateRows(X)
        try await fitInput(PCARows(values: X), featureNames: nil)
    }

    /// Fits prepared columns without exporting nested rows. Missing and nonfinite values throw.
    /// Records exact feature names and order, including duplicate names.
    public func fit(_ input: PreparedNumericBatch) async throws {
        try requirePresent(input)
        try await fitInput(PCAColumns(batch: input), featureNames: input.columnNames)
    }

    /// Fits either prepared matrix layout while preserving the caller's snapshot.
    /// NaNs, including exported missing entries, and infinities throw.
    public func fit(_ input: PreparedNumericMatrix) async throws {
        try await fitInput(input, featureNames: input.columnNames)
    }

    private func fitInput<Input: PCANumericInput>(_ input: Input, featureNames: [String]?) async throws {
        guard input.rowCount > 0, input.columnCount > 0 else { throw ClusterError.emptyInput }
        guard nComponents <= min(input.rowCount, input.columnCount) else {
            throw ClusterError.invalidParameter("nComponents cannot exceed min(samples, features).")
        }
        guard input.rowCount <= Int32.max, input.columnCount <= Int32.max,
              !input.rowCount.multipliedReportingOverflow(by: input.columnCount).overflow else {
            throw ClusterError.invalidParameter("PCA dimensions exceed backend limits.")
        }
        try requireFinite(input)
        let device = await HardwareRouter.shared.resolveDevice(
            for: "PCA", sampleCount: input.rowCount, featureCount: input.columnCount,
            requestedDevice: requestedDevice)
        let result = try PCAComputation.fit(input, components: nComponents, solver: svdSolver, device: device)
        guard result.mean.allSatisfy(\.isFinite), result.variance.allSatisfy(\.isFinite),
              result.components.allSatisfy({ $0.allSatisfy(\.isFinite) }), result.totalVariance.isFinite else {
            throw ClusterError.invalidParameter("PCA fitting produced nonfinite results.")
        }
        fitted = FittedState(result: result, device: device, featureNames: featureNames)
    }

    /// Projects nested rows with the existing ascending-feature Double accumulation.
    /// Names are positional on this compatibility interface; NaNs propagate.
    public func transform(_ X: [[Double]]) throws -> [[Double]] {
        guard let fitted else { throw ClusterError.fittingRequired }
        try validateRows(X)
        try requireWidth(X[0].count, fitted: fitted)
        let input = PCARows(values: X)
        return (0..<input.rowCount).map { row in
            (0..<nComponents).map { component in
                PCAComputation.score(input, row: row, component: component, fitted: fitted.result)
            }
        }
    }

    /// Projects finite prepared columns into PC1...PCk, retaining selected row provenance.
    /// Prepared fits require identical ordered feature names. Empty matching inputs are supported.
    public func transform(_ input: PreparedNumericBatch) throws -> PreparedNumericBatch {
        guard let fitted else { throw ClusterError.fittingRequired }
        try requireSchema(input.columnNames, fitted: fitted)
        try requirePresent(input)
        let source = PCAColumns(batch: input)
        try requireFinite(source)
        let columns = (0..<nComponents).map { component in
            (0..<input.rowCount).map { row in
                PCAComputation.score(source, row: row, component: component, fitted: fitted.result)
            }
        }
        guard columns.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw ClusterError.invalidParameter("PCA projection overflowed.")
        }
        return input.replacingNumericColumns(columns, columnNames: scoreNames)
    }

    /// Projects a finite matrix into PC1...PCk while preserving layout and row provenance.
    /// Prepared fits require identical ordered feature names. Empty matching inputs are supported.
    public func transform(_ input: PreparedNumericMatrix) throws -> PreparedNumericMatrix {
        guard let fitted else { throw ClusterError.fittingRequired }
        try requireSchema(input.columnNames, fitted: fitted)
        try requireFinite(input)
        let (count, overflow) = input.rowCount.multipliedReportingOverflow(by: nComponents)
        guard !overflow else { throw ClusterError.invalidParameter("PCA score dimensions overflow Int.") }
        var values = [Double](repeating: 0, count: count)
        for row in 0..<input.rowCount {
            for component in 0..<nComponents {
                let index = input.order == .rowMajor ? row * nComponents + component : component * input.rowCount + row
                values[index] = PCAComputation.score(input, row: row, component: component, fitted: fitted.result)
            }
        }
        guard values.allSatisfy(\.isFinite) else { throw ClusterError.invalidParameter("PCA projection overflowed.") }
        return input.replacingValues(values, columnNames: scoreNames)
    }

    /// Fits nested rows and projects them through the ordinary transform path.
    public func fitTransform(_ X: [[Double]]) async throws -> [[Double]] {
        try await fit(X)
        return try transform(X)
    }

    /// Fits prepared columns and projects them, preserving source snapshots and row identity.
    public func fitTransform(_ input: PreparedNumericBatch) async throws -> PreparedNumericBatch {
        try await fit(input)
        return try transform(input)
    }

    /// Fits a prepared matrix and projects it in the same layout with preserved row identity.
    public func fitTransform(_ input: PreparedNumericMatrix) async throws -> PreparedNumericMatrix {
        try await fit(input)
        return try transform(input)
    }

    private var scoreNames: [String] { (1...nComponents).map { "PC\($0)" } }

    private func validateRows(_ rows: [[Double]]) throws {
        guard let first = rows.first, !first.isEmpty else { throw ClusterError.emptyInput }
        for row in rows where row.count != first.count {
            throw ClusterError.dimensionMismatch(expected: first.count, got: row.count)
        }
    }

    private func requireWidth(_ width: Int, fitted: FittedState) throws {
        guard width == fitted.result.mean.count else {
            throw ClusterError.dimensionMismatch(expected: fitted.result.mean.count, got: width)
        }
    }

    private func requireSchema(_ names: [String], fitted: FittedState) throws {
        try requireWidth(names.count, fitted: fitted)
        if let expected = fitted.featureNames, names != expected {
            throw ClusterError.invalidParameter("PCA feature names and order must match fitting.")
        }
    }

    private func requirePresent(_ input: PreparedNumericBatch) throws {
        guard input.columns.allSatisfy({ $0.nullCount == 0 }) else {
            throw ClusterError.invalidParameter("PCA input contains missing values.")
        }
    }

    private func requireFinite<Input: PCANumericInput>(_ input: Input) throws {
        for row in 0..<input.rowCount {
            for column in 0..<input.columnCount where !input[row, column].isFinite {
                throw ClusterError.invalidParameter("PCA input must contain finite values.")
            }
        }
    }

    /// Normalizes SVD signs using the existing largest-magnitude convention.
    public static func svdFlip(u: inout [[Double]]?, vt: inout [[Double]], uBasedDecision: Bool = false) {
        RandomizedSVD.svdFlip(u: &u, vt: &vt, uBasedDecision: uBasedDecision)
    }
}
#endif
