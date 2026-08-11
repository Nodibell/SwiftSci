import Foundation
import Accelerate

/// Represents observable dictionary basis functions for Koopman Operator EDMD.
public enum ObservableDictionary: Sendable {
    /// Standard polynomial basis up to specified degree.
    case polynomial(degree: Int, includeCrossTerms: Bool = true)
    /// Radial Basis Functions with specified centers and scale gamma.
    case rbf(centers: [[Double]], gamma: Double)
    /// Trigonometric Fourier basis functions with given frequencies.
    case fourier(frequencies: [Double])
    /// Linear identity basis (no lifting).
    case identity
    /// Combination of multiple observable dictionaries.
    case combined([ObservableDictionary])

    /// Evaluates the dictionary feature vector for a given state vector x.
    /// - Parameter x: Input state vector [x_1, x_2, ..., x_d].
    /// - Returns: Lifted observable feature vector [psi_1, psi_2, ..., psi_K].
    public func evaluate(x: [Double]) -> [Double] {
        switch self {
        case .identity:
            return x

        case .polynomial(let degree, let includeCrossTerms):
            var features = x
            let d = x.count
            if degree >= 2 {
                // Pure powers
                for p in 2...degree {
                    for i in 0..<d {
                        features.append(pow(x[i], Double(p)))
                    }
                }
                // Pairwise cross terms
                if includeCrossTerms && d > 1 {
                    for i in 0..<d {
                        for j in (i + 1)..<d {
                            features.append(x[i] * x[j])
                        }
                    }
                }
            }
            return features

        case .rbf(let centers, let gamma):
            var features = x
            for center in centers {
                var distSq = 0.0
                let count = min(x.count, center.count)
                for i in 0..<count {
                    let diff = x[i] - center[i]
                    distSq += diff * diff
                }
                features.append(exp(-gamma * distSq))
            }
            return features

        case .fourier(let frequencies):
            var features = x
            for freq in frequencies {
                for xi in x {
                    features.append(sin(freq * xi))
                    features.append(cos(freq * xi))
                }
            }
            return features

        case .combined(let dicts):
            var features = [Double]()
            for dict in dicts {
                features.append(contentsOf: dict.evaluate(x: x))
            }
            return features
        }
    }
}

/// Koopman Operator time-series forecasting and dynamical system modeling actor.
/// Uses Extended Dynamic Mode Decomposition (EDMD) to approximate non-linear dynamics
/// via linear operators in lifted observable feature spaces.
public actor KoopmanOperator {
    /// The chosen observable dictionary basis functions.
    public let dictionary: ObservableDictionary
    /// Tikhonov Ridge regularization parameter (lambda >= 0).
    public let regularization: Double
    /// Number of embedding lags for time-delay Hankel reconstruction (>= 1).
    public let embeddingLags: Int

    private var fittedKoopmanMatrix: [[Double]]?
    private var fittedReconstructionMatrix: [[Double]]?
    private var lastFittedState: [Double]?
    private var stateDimension: Int = 0
    private var observableDimension: Int = 0
    private var isFitted: Bool = false

    /// Creates a new KoopmanOperator instance.
    /// - Parameters:
    ///   - dictionary: Observable basis function dictionary (default: polynomial of degree 2).
    ///   - regularization: Tikhonov regularization factor (default: 1e-5).
    ///   - embeddingLags: Time-delay embedding lags count for 1D series (default: 1).
    public init(
        dictionary: ObservableDictionary = .polynomial(degree: 2),
        regularization: Double = 1e-5,
        embeddingLags: Int = 1
    ) {
        self.dictionary = dictionary
        self.regularization = max(0.0, regularization)
        self.embeddingLags = max(1, embeddingLags)
    }

    /// Fits the Koopman Operator on a 1D scalar time series using Hankel time-delay embedding.
    /// - Parameter series: Array of 1D time series values.
    /// - Throws: `ForecastError` if series is empty, invalid, or too short.
    public func fit(series: [Double]) async throws {
        guard !series.isEmpty else {
            throw ForecastError.emptyTimeSeries
        }
        if series.contains(where: { $0.isNaN }) {
            throw ForecastError.containsNaN
        }
        if series.contains(where: { $0.isInfinite }) {
            throw ForecastError.containsInfinity
        }

        let minLen = embeddingLags + 2
        guard series.count >= minLen else {
            throw ForecastError.insufficientLength(minimum: minLen, got: series.count)
        }

        // Construct Hankel delay embedding trajectory
        var trajectory = [[Double]]()
        for t in (embeddingLags - 1)..<series.count {
            var state = [Double]()
            for lag in 0..<embeddingLags {
                state.append(series[t - lag])
            }
            trajectory.append(state)
        }

        try await fit(trajectory: trajectory)
    }

    /// Fits the Koopman Operator on a multi-dimensional state trajectory.
    /// - Parameter trajectory: Array of multi-dimensional state vectors [[x_0], [x_1], ...].
    /// - Throws: `ForecastError` if trajectory is empty or matrix solve fails.
    public func fit(trajectory: [[Double]]) async throws {
        let N = trajectory.count
        guard N >= 2 else {
            throw ForecastError.insufficientLength(minimum: 2, got: N)
        }
        let d = trajectory[0].count
        guard d > 0 else {
            throw ForecastError.emptyTimeSeries
        }

        for state in trajectory {
            guard state.count == d else {
                throw ForecastError.matrixDimensionMismatch(
                    expectedRows: d, expectedCols: 1, gotRows: state.count, gotCols: 1
                )
            }
            if state.contains(where: { $0.isNaN }) { throw ForecastError.containsNaN }
            if state.contains(where: { $0.isInfinite }) { throw ForecastError.containsInfinity }
        }

        let numSnapshots = N - 1
        let samplePsi = dictionary.evaluate(x: trajectory[0])
        let K_dim = samplePsi.count
        guard K_dim > 0 else {
            throw ForecastError.emptyTimeSeries
        }

        // Build feature snapshot matrices PsiX and PsiY
        var PsiX = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: numSnapshots)
        var PsiY = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: numSnapshots)
        var XState = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: numSnapshots)

        for i in 0..<numSnapshots {
            let psiX = dictionary.evaluate(x: trajectory[i])
            let psiY = dictionary.evaluate(x: trajectory[i + 1])
            PsiX[i] = psiX
            PsiY[i] = psiY
            XState[i] = trajectory[i]
        }

        // Compute Covariance G = PsiX^T * PsiX  (K_dim x K_dim)
        // and Cross-Covariance A_XY = PsiX^T * PsiY (K_dim x K_dim)
        var G = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: K_dim)
        var A_XY = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: K_dim)
        var B_state = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: d)

        for i in 0..<numSnapshots {
            let px = PsiX[i]
            let py = PsiY[i]
            let xs = XState[i]

            for r in 0..<K_dim {
                for c in 0..<K_dim {
                    G[r][c] += px[r] * px[c]
                    A_XY[r][c] += px[r] * py[c]
                }
                for c in 0..<d {
                    B_state[c][r] += xs[c] * px[r]
                }
            }
        }

        // Add Ridge Regularization lambda * I to G
        for r in 0..<K_dim {
            G[r][r] += regularization
        }

        // Solve G * Z = A_XY for Z (K_dim x K_dim), then Koopman Matrix = Z^T
        let Z = try Self.solveLinearSystem(A: G, B: A_XY)

        var KMat = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: K_dim)
        for r in 0..<K_dim {
            for c in 0..<K_dim {
                KMat[r][c] = Z[c][r]
            }
        }

        // Solve G * W = B_state^T for W (K_dim x d), then C = W^T (d x K_dim)
        var B_stateT = [[Double]](repeating: [Double](repeating: 0.0, count: d), count: K_dim)
        for r in 0..<K_dim {
            for c in 0..<d {
                B_stateT[r][c] = B_state[c][r]
            }
        }
        let W = try Self.solveLinearSystem(A: G, B: B_stateT)

        var CMat = [[Double]](repeating: [Double](repeating: 0.0, count: K_dim), count: d)
        for r in 0..<d {
            for c in 0..<K_dim {
                CMat[r][c] = W[c][r]
            }
        }

        self.fittedKoopmanMatrix = KMat
        self.fittedReconstructionMatrix = CMat
        self.lastFittedState = trajectory.last
        self.stateDimension = d
        self.observableDimension = K_dim
        self.isFitted = true
    }

    /// Forecasts multi-dimensional states for a specified horizon ahead from the last fitted state.
    /// - Parameter horizon: Number of steps to forecast ahead (h >= 1).
    /// - Returns: Forecasted multi-dimensional state trajectory.
    /// - Throws: `ForecastError` if not fitted or horizon is invalid.
    public func predict(horizon: Int) async throws -> [[Double]] {
        guard isFitted, let initialState = lastFittedState else {
            throw ForecastError.notFitted
        }
        return try await predict(from: initialState, horizon: horizon)
    }

    /// Forecasts 1D time-series values for a specified horizon ahead.
    /// - Parameter horizon: Number of steps to forecast ahead (h >= 1).
    /// - Returns: Array of forecasted 1D scalar time series values.
    /// - Throws: `ForecastError` if not fitted or horizon is invalid.
    public func predict1D(horizon: Int) async throws -> [Double] {
        let trajectory = try await predict(horizon: horizon)
        return trajectory.map { $0[0] }
    }

    /// Forecasts multi-dimensional states starting from a specified initial state vector.
    /// - Parameters:
    ///   - initialState: Starting state vector [x_1, x_2, ..., x_d].
    ///   - horizon: Number of steps to forecast ahead (h >= 1).
    /// - Returns: Forecasted multi-dimensional state trajectory.
    /// - Throws: `ForecastError` if not fitted, horizon invalid, or state size mismatches.
    public func predict(from initialState: [Double], horizon: Int) async throws -> [[Double]] {
        guard isFitted,
              let KMat = fittedKoopmanMatrix,
              let CMat = fittedReconstructionMatrix else {
            throw ForecastError.notFitted
        }
        guard horizon >= 1 else {
            throw ForecastError.invalidHorizon(horizon)
        }
        guard initialState.count == stateDimension else {
            throw ForecastError.matrixDimensionMismatch(
                expectedRows: stateDimension, expectedCols: 1, gotRows: initialState.count, gotCols: 1
            )
        }

        var results = [[Double]]()
        var currentState = initialState

        for _ in 0..<horizon {
            let psi = dictionary.evaluate(x: currentState)

            // Multiply KMat * psi
            var nextPsi = [Double](repeating: 0.0, count: observableDimension)
            for r in 0..<observableDimension {
                var sum = 0.0
                for c in 0..<observableDimension {
                    sum += KMat[r][c] * psi[c]
                }
                nextPsi[r] = sum
            }

            // Reconstruct nextState = CMat * nextPsi
            var nextState = [Double](repeating: 0.0, count: stateDimension)
            for r in 0..<stateDimension {
                var sum = 0.0
                for c in 0..<observableDimension {
                    sum += CMat[r][c] * nextPsi[c]
                }
                nextState[r] = sum
            }

            results.append(nextState)
            currentState = nextState
        }

        return results
    }

    /// Computes the complex eigenvalues of the fitted Koopman matrix.
    /// Eigenvalues characterize the growth rates, decay, and discrete frequency spectra of the underlying dynamical system.
    /// - Returns: Array of complex eigenvalues (real: Double, imag: Double).
    /// - Throws: `ForecastError` if model is not fitted or LAPACK fails.
    public func eigenvalues() async throws -> [(real: Double, imag: Double)] {
        guard isFitted, let KMat = fittedKoopmanMatrix else {
            throw ForecastError.notFitted
        }
        let k = KMat.count
        guard k > 0 else { return [] }

        var jobvl = Int8(78) // 'N'
        var jobvr = Int8(78) // 'N'
        var n = LAPACKInteger(k)
        var lda = n
        var ldvl = LAPACKInteger(1)
        var ldvr = LAPACKInteger(1)
        var info = LAPACKInteger(0)

        var aColMajor = [Double](repeating: 0.0, count: k * k)
        for r in 0..<k {
            for c in 0..<k {
                aColMajor[c * k + r] = KMat[r][c]
            }
        }

        var wr = [Double](repeating: 0.0, count: k)
        var wi = [Double](repeating: 0.0, count: k)
        var dummyVL = [Double](repeating: 0.0, count: 1)
        var dummyVR = [Double](repeating: 0.0, count: 1)

        var lwork = LAPACKInteger(-1)
        var workQuery = [Double](repeating: 0.0, count: 1)
        dgeev_wrapper(&jobvl, &jobvr, &n, &aColMajor, &lda, &wr, &wi, &dummyVL, &ldvl, &dummyVR, &ldvr, &workQuery, &lwork, &info)

        lwork = LAPACKInteger(workQuery[0])
        var work = [Double](repeating: 0.0, count: Int(lwork))
        dgeev_wrapper(&jobvl, &jobvr, &n, &aColMajor, &lda, &wr, &wi, &dummyVL, &ldvl, &dummyVR, &ldvr, &work, &lwork, &info)

        guard info == 0 else {
            throw ForecastError.singularMatrix
        }

        var result = [(real: Double, imag: Double)]()
        for i in 0..<k {
            result.append((real: wr[i], imag: wi[i]))
        }
        return result
    }

    // MARK: - Private Matrix Solver Helper

    private static func solveLinearSystem(A: [[Double]], B: [[Double]]) throws -> [[Double]] {
        let k = A.count
        guard k > 0 && A[0].count == k else { throw ForecastError.singularMatrix }
        let m = B[0].count
        guard B.count == k else {
            throw ForecastError.matrixDimensionMismatch(expectedRows: k, expectedCols: m, gotRows: B.count, gotCols: m)
        }

        var aColMajor = [Double](repeating: 0.0, count: k * k)
        for r in 0..<k {
            for c in 0..<k {
                aColMajor[c * k + r] = A[r][c]
            }
        }

        var bColMajor = [Double](repeating: 0.0, count: k * m)
        for r in 0..<k {
            for c in 0..<m {
                bColMajor[c * k + r] = B[r][c]
            }
        }

        var n = LAPACKInteger(k)
        var nrhs = LAPACKInteger(m)
        var lda = n
        var ldb = n
        var ipiv = [LAPACKInteger](repeating: 0, count: k)
        var info = LAPACKInteger(0)

        dgesv_wrapper(&n, &nrhs, &aColMajor, &lda, &ipiv, &bColMajor, &ldb, &info)

        guard info == 0 else {
            throw ForecastError.singularMatrix
        }

        var X = [[Double]](repeating: [Double](repeating: 0.0, count: m), count: k)
        for r in 0..<k {
            for c in 0..<m {
                X[r][c] = bColMajor[c * k + r]
            }
        }
        return X
    }
}
