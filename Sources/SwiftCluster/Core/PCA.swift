#if os(macOS)
import Foundation
import Accelerate
import MLX
import SwiftPreprocessing
@_exported import SwiftDataFrame

#if ACCELERATE_LAPACK_ILP64
typealias LAPACKInteger = Int
#else
typealias LAPACKInteger = Int32
#endif

private func dgesvd_wrapper(
    _ jobu: UnsafeMutablePointer<Int8>,
    _ jobvt: UnsafeMutablePointer<Int8>,
    _ m: UnsafeMutablePointer<LAPACKInteger>,
    _ n: UnsafeMutablePointer<LAPACKInteger>,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: UnsafeMutablePointer<LAPACKInteger>,
    _ s: UnsafeMutablePointer<Double>,
    _ u: UnsafeMutablePointer<Double>,
    _ ldu: UnsafeMutablePointer<LAPACKInteger>,
    _ vt: UnsafeMutablePointer<Double>,
    _ ldvt: UnsafeMutablePointer<LAPACKInteger>,
    _ work: UnsafeMutablePointer<Double>,
    _ lwork: UnsafeMutablePointer<LAPACKInteger>,
    _ info: UnsafeMutablePointer<LAPACKInteger>
) {
    dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
}

/// SVD solver strategy for PCA on CPU.
public enum SVDSolver: Sendable {
    /// Use full LAPACK `dgesdd_` (exact, best for k ≥ 80% of min dimension).
    case full
    /// Use Randomized SVD (Halko 2011, best for k ≪ min(M,N)). O(M·N·k) complexity.
    case randomized
    /// Automatically select: Randomized SVD when k < 0.8·min(M,N), else full SVD.
    case auto
}

/// Principal Component Analysis (PCA) for dimensionality reduction using Accelerate LAPACK SVD or MLX SVD.
public actor PCA {
    /// Number of components to keep.
    public let nComponents: Int

    /// SVD solver to use on the CPU path.
    public let svdSolver: SVDSolver

    /// Target compute device for PCA execution.
    public let requestedDevice: ExecutionDevice
    
    /// Device actually used for the last fit operation.
    public private(set) var resolvedDevice: ExecutionDevice?
    
    /// Mean values of features, computed during fit. Shape: [nFeatures]
    public private(set) var mean: [Double]?
    
    /// Principal components (eigenvectors). Shape: [nComponents, nFeatures]
    public private(set) var components: [[Double]]?
    
    /// Explained variance of the selected components. Shape: [nComponents]
    public private(set) var explainedVariance: [Double]?

    /// Explained variance ratio of each selected component (percentage of total variance).
    public var explainedVarianceRatio: [Double]? {
        guard let explainedVariance = explainedVariance else { return nil }
        let totalVar = explainedVariance.reduce(0.0, +)
        guard totalVar > 0 else { return explainedVariance.map { _ in 0.0 } }
        return explainedVariance.map { $0 / totalVar }
    }

    /// Initializes PCA with the number of components to keep and the target device.
    public init(nComponents: Int, device: ExecutionDevice = .auto) throws {
        guard nComponents > 0 else {
            throw ClusterError.invalidParameter("nComponents must be greater than 0.")
        }
        self.nComponents = nComponents
        self.requestedDevice = device
        self.svdSolver = .auto
    }

    /// Initializes PCA with an explicit SVD solver strategy.
    ///
    /// Use `svdSolver: .randomized` when `nComponents ≪ min(rows, cols)` for an
    /// order-of-magnitude speedup via the Halko (2011) Randomized SVD algorithm.
    ///
    /// ```swift
    /// // 10 components from a 1000×100 matrix — Randomized SVD is ~4× faster.
    /// var pca = try PCA(nComponents: 10, svdSolver: .randomized)
    /// let reduced = try await pca.fitTransform(X)
    /// ```
    public init(nComponents: Int, svdSolver: SVDSolver, device: ExecutionDevice = .auto) throws {
        guard nComponents > 0 else {
            throw ClusterError.invalidParameter("nComponents must be greater than 0.")
        }
        self.nComponents = nComponents
        self.requestedDevice = device
        self.svdSolver = svdSolver
    }
    
    /// Fits the PCA model on the given dataset X.
    /// - Parameter X: A 2D array of shape [samples, features].
    public func fit(_ X: [[Double]]) async throws {
        guard !X.isEmpty, !X[0].isEmpty else {
            throw ClusterError.emptyInput
        }
        
        let numSamples = X.count
        let numFeatures = X[0].count
        
        guard nComponents <= min(numSamples, numFeatures) else {
            throw ClusterError.invalidParameter("nComponents (\(nComponents)) cannot be greater than min(samples: \(numSamples), features: \(numFeatures)).")
        }
        
        let device = await HardwareRouter.shared.resolveDevice(
            for: "PCA",
            sampleCount: numSamples,
            featureCount: numFeatures,
            requestedDevice: requestedDevice
        )
        self.resolvedDevice = device
        
        switch device {
        case .cpu:
            try fitCPU(X)
        case .gpu, .ane, .auto:
            let nComp = self.nComponents
            let results = Device.withDefaultDevice(.gpu) {
                Self.runFitGPU(X: X, nComponents: nComp)
            }
            self.mean = results.mean
            self.components = results.components
            self.explainedVariance = results.explainedVariance
        }
    }
    
    // MARK: - CPU Backend (LAPACK SVD / Randomized SVD)

    private func fitCPU(_ X: [[Double]]) throws {
        let numSamples = X.count
        let numFeatures = X[0].count
        let minDim = min(numSamples, numFeatures)

        // Choose solver: use Randomized SVD when k < 80% of min dimension.
        let useRandomized: Bool
        switch svdSolver {
        case .randomized: useRandomized = true
        case .full:       useRandomized = false
        case .auto:       useRandomized = Double(nComponents) < 0.8 * Double(minDim)
        }

        if useRandomized {
            try fitCPURandomized(X)
        } else {
            try fitCPUSVD(X)
        }
    }

    /// CPU path using Randomized SVD (Halko 2011). O(M·N·k) complexity.
    private func fitCPURandomized(_ X: [[Double]]) throws {
        let numFeatures = X[0].count

        // 1. Column means for centering.
        var colMeans = [Double](repeating: 0.0, count: numFeatures)
        for row in X {
            for c in 0..<numFeatures { colMeans[c] += row[c] }
        }
        var inv = 1.0 / Double(X.count)
        vDSP_vsmulD(colMeans, 1, &inv, &colMeans, 1, vDSP_Length(numFeatures))

        // 2. Build centered matrix.
        let numSamples = X.count
        var Xc = X
        for r in 0..<numSamples {
            for c in 0..<numFeatures { Xc[r][c] -= colMeans[c] }
        }

        // 3. Randomized SVD.
        let rsvd = try RandomizedSVD.compute(X: Xc, nComponents: nComponents)

        // 4. Extract principal components from Vt rows (shape [k, N]).
        var comp = [[Double]]()
        comp.reserveCapacity(nComponents)
        var expVar = [Double]()
        expVar.reserveCapacity(nComponents)
        let scaleFactor = 1.0 / Double(max(1, numSamples - 1))
        for i in 0..<nComponents {
            comp.append(rsvd.Vt[i])
            let sv = rsvd.S[i]
            expVar.append(sv * sv * scaleFactor)
        }

        self.mean = colMeans
        self.components = comp
        self.explainedVariance = expVar
    }

    private func fitCPUCov(_ X: [[Double]]) throws {
        let numSamples = X.count
        let numFeatures = X[0].count

        var colMeans = [Double](repeating: 0.0, count: numFeatures)
        for row in X {
            guard row.count == numFeatures else {
                throw ClusterError.dimensionMismatch(expected: numFeatures, got: row.count)
            }
            for c in 0..<numFeatures { colMeans[c] += row[c] }
        }
        for c in 0..<numFeatures { colMeans[c] /= Double(numSamples) }

        var xCent = [Double](repeating: 0.0, count: numSamples * numFeatures)
        for r in 0..<numSamples {
            for c in 0..<numFeatures {
                xCent[r * numFeatures + c] = X[r][c] - colMeans[c]
            }
        }

        var cov = [Double](repeating: 0.0, count: numFeatures * numFeatures)
        let alpha = 1.0 / Double(max(1, numSamples - 1))
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                    Int32(numFeatures), Int32(numSamples),
                    alpha, xCent, Int32(numFeatures),
                    0.0, &cov, Int32(numFeatures))

        var jobz: Int8 = 86 // 'V'
        var uplo: Int8 = 85 // 'U'
        var n = LAPACKInteger(numFeatures)
        var lda = n
        var w = [Double](repeating: 0.0, count: numFeatures)
        var info = LAPACKInteger(0)

        var workQuery = [Double](repeating: 0.0, count: 1)
        var lwork = LAPACKInteger(-1)
        var iworkQuery = [LAPACKInteger](repeating: 0, count: 1)
        var liwork = LAPACKInteger(-1)

        dsyevd_(&jobz, &uplo, &n, &cov, &lda, &w, &workQuery, &lwork, &iworkQuery, &liwork, &info)
        lwork = LAPACKInteger(workQuery[0])
        liwork = iworkQuery[0]
        var work = [Double](repeating: 0.0, count: Int(lwork))
        var iwork = [LAPACKInteger](repeating: 0, count: Int(liwork))

        dsyevd_(&jobz, &uplo, &n, &cov, &lda, &w, &work, &lwork, &iwork, &liwork, &info)
        guard info == 0 else {
            throw ClusterError.svdFailed(info: Int32(info))
        }

        var comp = [[Double]]()
        comp.reserveCapacity(nComponents)
        var expVar = [Double]()
        expVar.reserveCapacity(nComponents)

        for k in 0..<nComponents {
            let idx = numFeatures - 1 - k
            expVar.append(max(0.0, w[idx]))
            var row = [Double](repeating: 0.0, count: numFeatures)
            for c in 0..<numFeatures {
                row[c] = cov[c * numFeatures + idx]
            }
            comp.append(row)
        }

        self.mean = colMeans
        self.components = comp
        self.explainedVariance = expVar
    }

    private func fitCPUSVD(_ X: [[Double]]) throws {
        let numSamples = X.count
        let numFeatures = X[0].count
        
        // 1. Calculate column means
        var colMeans = [Double](repeating: 0.0, count: numFeatures)
        for row in X {
            guard row.count == numFeatures else {
                throw ClusterError.dimensionMismatch(expected: numFeatures, got: row.count)
            }
            for c in 0..<numFeatures {
                colMeans[c] += row[c]
            }
        }
        for c in 0..<numFeatures {
            colMeans[c] /= Double(numSamples)
        }
        
        // 2. Prepare centered column-major matrix A for LAPACK SVD
        var a = [Double](repeating: 0.0, count: numSamples * numFeatures)
        for r in 0..<numSamples {
            for c in 0..<numFeatures {
                a[c * numSamples + r] = X[r][c] - colMeans[c]
            }
        }
        
        // 3. Setup SVD parameters for dgesdd_ (divide-and-conquer fast SVD)
        var jobz: Int8 = 83 // 'S' (compute first min(M, N) singular vectors of V^T)
        
        var m = LAPACKInteger(numSamples)
        var n = LAPACKInteger(numFeatures)
        var lda = m
        var ldu = m
        
        let minDim = min(numSamples, numFeatures)
        var ldvt = LAPACKInteger(minDim)
        
        var s = [Double](repeating: 0.0, count: minDim)
        var u = [Double](repeating: 0.0, count: numSamples * minDim)
        var vt = [Double](repeating: 0.0, count: minDim * numFeatures)
        
        var info = LAPACKInteger(0)
        var workQuery = [Double](repeating: 0.0, count: 1)
        var lwork = LAPACKInteger(-1)
        var iwork = [LAPACKInteger](repeating: 0, count: 8 * minDim)
        
        // Workspace query
        dgesdd_(&jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &workQuery, &lwork, &iwork, &info)
        guard info == 0 else {
            throw ClusterError.svdFailed(info: Int32(info))
        }
        
        lwork = LAPACKInteger(workQuery[0])
        var work = [Double](repeating: 0.0, count: Int(lwork))
        
        // Run actual divide-and-conquer SVD
        dgesdd_(&jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &iwork, &info)
        guard info == 0 else {
            throw ClusterError.svdFailed(info: Int32(info))
        }
        
        // 4. Extract principal components (first nComponents rows of V^T)
        var comp = [[Double]]()
        comp.reserveCapacity(nComponents)
        for k in 0..<nComponents {
            var row = [Double]()
            row.reserveCapacity(numFeatures)
            for c in 0..<numFeatures {
                row.append(vt[c * minDim + k])
            }
            comp.append(row)
        }
        
        // 5. Calculate explained variance: singular_values^2 / (N - 1)
        let df = Double(max(1, numSamples - 1))
        var expVar = [Double]()
        expVar.reserveCapacity(nComponents)
        for k in 0..<nComponents {
            expVar.append((s[k] * s[k]) / df)
        }
        
        self.mean = colMeans
        self.components = comp
        self.explainedVariance = expVar
    }
    
    // MARK: - GPU Backend (MLX SVD)
    
    private static func runFitGPU(
        X: [[Double]],
        nComponents: Int
    ) -> (mean: [Double], components: [[Double]], explainedVariance: [Double]) {
        let numSamples = X.count
        let numFeatures = X[0].count
        
        let flat = X.flatMap { $0.map { Float($0) } }
        let X_arr = MLXArray(flat).reshaped([numSamples, numFeatures])
        
        let colMeans_arr = X_arr.mean(axis: 0)
        let XCent = X_arr - colMeans_arr
        
        let (_, s_arr, vt_arr) = svd(XCent, stream: .cpu)
        
        eval(colMeans_arr, s_arr, vt_arr)
        
        let colMeans = colMeans_arr.asArray(Float.self).map { Double($0) }
        let s = s_arr.asArray(Float.self).map { Double($0) }
        let vt = vt_arr.asArray(Float.self)
        
        let vtCols = vt_arr.shape[1]
        
        var comp = [[Double]]()
        comp.reserveCapacity(nComponents)
        for k in 0..<nComponents {
            var row = [Double]()
            row.reserveCapacity(numFeatures)
            for c in 0..<numFeatures {
                row.append(Double(vt[k * vtCols + c]))
            }
            comp.append(row)
        }
        
        let df = Double(max(1, numSamples - 1))
        var expVar = [Double]()
        expVar.reserveCapacity(nComponents)
        for k in 0..<nComponents {
            expVar.append((s[k] * s[k]) / df)
        }
        
        return (colMeans, comp, expVar)
    }
    
    /// Projects the given dataset X onto the principal components.
    /// - Parameter X: A 2D array of shape [samples, features].
    /// - Returns: Projected dataset of shape [samples, nComponents].
    public func transform(_ X: [[Double]]) throws -> [[Double]] {
        guard let mean = self.mean, let components = self.components else {
            throw ClusterError.fittingRequired
        }
        guard !X.isEmpty, !X[0].isEmpty else {
            throw ClusterError.emptyInput
        }
        
        let numFeatures = mean.count
        var result = [[Double]]()
        result.reserveCapacity(X.count)
        
        for row in X {
            guard row.count == numFeatures else {
                throw ClusterError.dimensionMismatch(expected: numFeatures, got: row.count)
            }
            
            var projected = [Double](repeating: 0.0, count: nComponents)
            for k in 0..<nComponents {
                var sum = 0.0
                for c in 0..<numFeatures {
                    sum += (row[c] - mean[c]) * components[k][c]
                }
                projected[k] = sum
            }
            result.append(projected)
        }
        
        return result
    }
    
    /// Fits the model on X and returns the projected data.
    /// - Parameter X: A 2D array of shape [samples, features].
    /// - Returns: Projected dataset of shape [samples, nComponents].
    public func fitTransform(_ X: [[Double]]) async throws -> [[Double]] {
        try await fit(X)
        return try transform(X)
    }
}
#endif // os(macOS)
