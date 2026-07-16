import Foundation
import Accelerate

#if ACCELERATE_NEW_LAPACK
  #if ACCELERATE_LAPACK_ILP64
  typealias LAPACKInteger = Int
  #else
  typealias LAPACKInteger = Int32
  #endif
#else
typealias LAPACKInteger = __CLPK_integer
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

/// Principal Component Analysis (PCA) for dimensionality reduction using Accelerate LAPACK SVD.
public actor PCA {
    /// Number of components to keep.
    public let nComponents: Int
    
    /// Mean values of features, computed during fit. Shape: [nFeatures]
    public private(set) var mean: [Double]?
    
    /// Principal components (eigenvectors). Shape: [nComponents, nFeatures]
    public private(set) var components: [[Double]]?
    
    /// Explained variance of the selected components. Shape: [nComponents]
    public private(set) var explainedVariance: [Double]?
    
    /// Initializes PCA with the number of components to keep.
    /// - Parameter nComponents: Number of principal components.
    public init(nComponents: Int) throws {
        guard nComponents > 0 else {
            throw ClusterError.invalidParameter("nComponents must be greater than 0.")
        }
        self.nComponents = nComponents
    }
    
    /// Fits the PCA model on the given dataset X.
    /// - Parameter X: A 2D array of shape [samples, features].
    public func fit(_ X: [[Double]]) throws {
        guard !X.isEmpty, !X[0].isEmpty else {
            throw ClusterError.emptyInput
        }
        
        let numSamples = X.count
        let numFeatures = X[0].count
        
        guard nComponents <= min(numSamples, numFeatures) else {
            throw ClusterError.invalidParameter("nComponents (\(nComponents)) cannot be greater than min(samples: \(numSamples), features: \(numFeatures)).")
        }
        
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
        
        // 3. Setup SVD parameters for dgesvd_
        var jobu: Int8 = 78  // 'N' (do not compute U)
        var jobvt: Int8 = 83 // 'S' (compute first min(M, N) singular vectors of V^T)
        
        var m = LAPACKInteger(numSamples)
        var n = LAPACKInteger(numFeatures)
        var lda = m
        var ldu = LAPACKInteger(1)
        
        let minDim = min(numSamples, numFeatures)
        var ldvt = LAPACKInteger(minDim)
        
        var s = [Double](repeating: 0.0, count: minDim)
        var u = [Double](repeating: 0.0, count: 1) // Dummy
        var vt = [Double](repeating: 0.0, count: minDim * numFeatures)
        
        var info = LAPACKInteger(0)
        var workQuery = [Double](repeating: 0.0, count: 1)
        var lwork = LAPACKInteger(-1)
        
        // Workspace query
        dgesvd_wrapper(&jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &workQuery, &lwork, &info)
        guard info == 0 else {
            throw ClusterError.svdFailed(info: info)
        }
        
        lwork = LAPACKInteger(workQuery[0])
        var work = [Double](repeating: 0.0, count: Int(lwork))
        
        // Run actual SVD
        dgesvd_wrapper(&jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        guard info == 0 else {
            throw ClusterError.svdFailed(info: info)
        }
        
        // 4. Extract principal components (first nComponents rows of V^T)
        // vt is column-major V^T of shape [minDim, numFeatures]
        var comp = [[Double]]()
        for k in 0..<nComponents {
            var row = [Double]()
            for c in 0..<numFeatures {
                row.append(vt[c * minDim + k])
            }
            comp.append(row)
        }
        
        // 5. Calculate explained variance: singular_values^2 / (N - 1)
        let df = Double(max(1, numSamples - 1))
        var expVar = [Double]()
        for k in 0..<nComponents {
            expVar.append((s[k] * s[k]) / df)
        }
        
        self.mean = colMeans
        self.components = comp
        self.explainedVariance = expVar
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
    public func fitTransform(_ X: [[Double]]) throws -> [[Double]] {
        try fit(X)
        return try transform(X)
    }
}


