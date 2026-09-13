#if os(macOS)
import Foundation
import Accelerate
import SwiftPreprocessing
@_exported import SwiftDataFrame

/// t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear dimensionality reduction and manifold learning.
///
/// Converts similarities between data points into joint probabilities and minimizes the Kullback-Leibler (KL)
/// divergence between the low-dimensional embedding and the high-dimensional data.
///
/// Reference: Laurens van der Maaten and Geoffrey Hinton (2008). Visualizing Data using t-SNE. JMLR 9:2579-2605.
public actor TSNE: Sendable {
    
    /// Target dimensionality of the embedded space (typically 2 or 3).
    public let nComponents: Int
    
    /// Perplexity relates to the number of nearest neighbors that is used in other manifold learning algorithms.
    /// Larger datasets usually require larger perplexities (typical values between 5 and 50).
    public let perplexity: Double
    
    /// The learning rate for gradient descent (typical values between 10.0 and 1000.0).
    public let learningRate: Double
    
    /// Maximum number of iterations for the optimization.
    public let maxIterations: Int
    
    /// Factor by which the joint probabilities in high-dimensional space are exaggerated in the initial optimization stage.
    public let earlyExaggeration: Double
    
    /// Number of iterations to apply early exaggeration.
    public let earlyExaggerationIter: Int
    
    /// Optional random seed for reproducible initialization and updates.
    public let seed: UInt64?
    
    /// Fitted low-dimensional embedding coordinates [N, nComponents].
    public private(set) var embedding: [[Double]]?
    
    /// Final Kullback-Leibler divergence after optimization.
    public private(set) var klDivergence: Double?
    
    /// Initializes a t-SNE manifold model.
    /// - Parameters:
    ///   - nComponents: Number of coordinates in the projected low-dimensional space (default: 2).
    ///   - perplexity: The balance between preserving local vs global structure (default: 30.0).
    ///   - learningRate: Gradient step size during embedding updates (default: 200.0).
    ///   - maxIterations: Total optimization iterations (default: 1000).
    ///   - earlyExaggeration: Scaling applied to high-dimensional similarities early in optimization (default: 12.0).
    ///   - earlyExaggerationIter: Number of iterations early exaggeration is applied (default: 250).
    ///   - seed: Optional deterministic pseudo-random seed.
    public init(
        nComponents: Int = 2,
        perplexity: Double = 30.0,
        learningRate: Double = 200.0,
        maxIterations: Int = 1000,
        earlyExaggeration: Double = 12.0,
        earlyExaggerationIter: Int = 250,
        seed: UInt64? = nil
    ) {
        self.nComponents = max(1, nComponents)
        self.perplexity = max(1.0, perplexity)
        self.learningRate = max(1.0, learningRate)
        self.maxIterations = max(10, maxIterations)
        self.earlyExaggeration = max(1.0, earlyExaggeration)
        self.earlyExaggerationIter = max(1, earlyExaggerationIter)
        self.seed = seed
    }
    
    /// Fits t-SNE on high-dimensional features and transforms them into low-dimensional coordinates.
    /// - Parameter X: Feature matrix of shape `[n_samples, n_features]`.
    /// - Returns: Embedding coordinates of shape `[n_samples, n_components]`.
    /// - Throws: `SwiftMLError` if inputs are invalid or insufficient.
    public func fitTransform(_ X: [[Double]]) throws -> [[Double]] {
        let n = X.count
        guard n >= 4 else {
            throw SwiftMLError.insufficientData(minimum: 4, got: n)
        }
        let d = X[0].count
        guard d >= 1 else {
            throw SwiftMLError.invalidInput("Feature matrix must contain at least 1 feature dimension.")
        }
        for row in X {
            if row.count != d {
                throw SwiftMLError.dimensionMismatch(expected: d, got: row.count)
            }
        }
        
        let effectivePerplexity = min(perplexity, Double(n - 1) / 3.0)
        
        // 1. Compute pairwise squared Euclidean distances: D[i][j] = ||x_i - x_j||^2
        var D = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                var distSq = 0.0
                for col in 0..<d {
                    let diff = X[i][col] - X[j][col]
                    distSq += diff * diff
                }
                D[i][j] = distSq
                D[j][i] = distSq
            }
        }
        
        // 2. Binary search for beta_i = 1 / (2 * sigma_i^2) to achieve desired Shannon entropy
        let targetEntropy = log(effectivePerplexity)
        var P = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        
        for i in 0..<n {
            var betaMin = -Double.infinity
            var betaMax = Double.infinity
            var beta = 1.0
            
            var pRow = [Double](repeating: 0.0, count: n)
            
            for _ in 0..<50 {
                var sumP = 0.0
                for j in 0..<n {
                    if i != j {
                        let pj = exp(-beta * D[i][j])
                        pRow[j] = pj
                        sumP += pj
                    } else {
                        pRow[j] = 0.0
                    }
                }
                
                if sumP < 1e-15 {
                    sumP = 1e-15
                }
                
                var h = 0.0
                for j in 0..<n {
                    pRow[j] /= sumP
                    if pRow[j] > 1e-15 {
                        h -= pRow[j] * log(pRow[j])
                    }
                }
                
                let hDiff = h - targetEntropy
                if abs(hDiff) < 1e-5 {
                    break
                }
                
                if hDiff > 0 {
                    betaMin = beta
                    if betaMax.isInfinite {
                        beta *= 2.0
                    } else {
                        beta = (beta + betaMax) / 2.0
                    }
                } else {
                    betaMax = beta
                    if betaMin.isInfinite {
                        beta /= 2.0
                    } else {
                        beta = (beta + betaMin) / 2.0
                    }
                }
            }
            
            for j in 0..<n {
                P[i][j] = pRow[j]
            }
        }
        
        // 3. Symmetrize joint probabilities: P_ij = (p_{j|i} + p_{i|j}) / (2n)
        let twoN = 2.0 * Double(n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                let sym = (P[i][j] + P[j][i]) / twoN
                let clamped = max(sym, 1e-12)
                P[i][j] = clamped
                P[j][i] = clamped
            }
            P[i][i] = 1e-12
        }
        
        // 4. Initialize low-dimensional embedding Y ~ N(0, 1e-4)
        var rng: any RandomNumberGenerator
        if let s = seed {
            rng = SeedableRandomNumberGenerator(seed: Int(s))
        } else {
            rng = SystemRandomNumberGenerator()
        }
        
        func sampleNormal(using generator: inout any RandomNumberGenerator) -> Double {
            let u1 = max(1e-15, Double.random(in: 0.0..<1.0, using: &generator))
            let u2 = Double.random(in: 0.0..<1.0, using: &generator)
            return sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2) * 1e-4
        }
        
        var Y = [[Double]](repeating: [Double](repeating: 0.0, count: nComponents), count: n)
        for i in 0..<n {
            for c in 0..<nComponents {
                Y[i][c] = sampleNormal(using: &rng)
            }
        }
        
        var iY = [[Double]](repeating: [Double](repeating: 0.0, count: nComponents), count: n)
        var gains = [[Double]](repeating: [Double](repeating: 1.0, count: nComponents), count: n)
        
        // 5. Optimization loop with early exaggeration and momentum
        for iter in 0..<maxIterations {
            let momentum: Double = iter < 250 ? 0.5 : 0.8
            let exaggeration: Double = iter < earlyExaggerationIter ? earlyExaggeration : 1.0
            
            // Student-t similarities in low-dimensional space: w_ij = 1 / (1 + ||y_i - y_j||^2)
            var sumW = 0.0
            var W = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
            
            for i in 0..<n {
                for j in (i + 1)..<n {
                    var distSq = 0.0
                    for c in 0..<nComponents {
                        let diff = Y[i][c] - Y[j][c]
                        distSq += diff * diff
                    }
                    let w = 1.0 / (1.0 + distSq)
                    W[i][j] = w
                    W[j][i] = w
                    sumW += 2.0 * w
                }
            }
            
            let safeSumW = max(sumW, 1e-15)
            
            // Gradients computation: dY_i = 4 * sum_j (exaggeration * P_ij - q_ij) * w_ij * (y_i - y_j)
            var dY = [[Double]](repeating: [Double](repeating: 0.0, count: nComponents), count: n)
            for i in 0..<n {
                for j in 0..<n {
                    if i == j { continue }
                    let q_ij = W[i][j] / safeSumW
                    let mult = 4.0 * (exaggeration * P[i][j] - q_ij) * W[i][j]
                    for c in 0..<nComponents {
                        dY[i][c] += mult * (Y[i][c] - Y[j][c])
                    }
                }
            }
            
            // Adaptive learning rate and momentum updates (Jacob's update)
            for i in 0..<n {
                for c in 0..<nComponents {
                    let grad = dY[i][c]
                    let step = iY[i][c]
                    
                    if (grad > 0.0) != (step > 0.0) {
                        gains[i][c] += 0.2
                    } else {
                        gains[i][c] = max(0.01, gains[i][c] * 0.8)
                    }
                    
                    iY[i][c] = momentum * step - learningRate * (gains[i][c] * grad)
                    Y[i][c] += iY[i][c]
                }
            }
            
            // Center embedding
            for c in 0..<nComponents {
                var meanC = 0.0
                for i in 0..<n {
                    meanC += Y[i][c]
                }
                meanC /= Double(n)
                for i in 0..<n {
                    Y[i][c] -= meanC
                }
            }
        }
        
        // Final KL divergence calculation
        var finalSumW = 0.0
        var finalW = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                var distSq = 0.0
                for c in 0..<nComponents {
                    let diff = Y[i][c] - Y[j][c]
                    distSq += diff * diff
                }
                let w = 1.0 / (1.0 + distSq)
                finalW[i][j] = w
                finalW[j][i] = w
                finalSumW += 2.0 * w
            }
        }
        let safeSumW = max(finalSumW, 1e-15)
        var kl = 0.0
        for i in 0..<n {
            for j in 0..<n {
                if i != j {
                    let p = P[i][j]
                    let q = max(finalW[i][j] / safeSumW, 1e-15)
                    kl += p * log(p / q)
                }
            }
        }
        
        self.embedding = Y
        self.klDivergence = kl
        return Y
    }
}

// MARK: - DataFrame Extension for t-SNE

extension DataFrame {
    /// Applies t-SNE non-linear dimensionality reduction on selected numeric columns of the DataFrame.
    /// - Parameters:
    ///   - columns: List of numeric column names to project. If nil, all numeric columns are selected.
    ///   - nComponents: Number of embedding dimensions (default: 2).
    ///   - perplexity: Perplexity parameter balancing local vs global manifold topology (default: 30.0).
    ///   - learningRate: Gradient step size during optimization (default: 200.0).
    ///   - maxIterations: Maximum optimization iterations (default: 500).
    ///   - seed: Deterministic seed for reproducible projections.
    /// - Returns: A new DataFrame containing the projected coordinates (e.g. `tsne_1`, `tsne_2`).
    public func tsne(
        columns: [String]? = nil,
        nComponents: Int = 2,
        perplexity: Double = 30.0,
        learningRate: Double = 200.0,
        maxIterations: Int = 500,
        seed: UInt64? = nil
    ) async throws -> DataFrame {
        let featureNames = columns ?? columnNames.filter { colName in
            self[colName]?.dtype.isNumeric == true
        }
        guard !featureNames.isEmpty else {
            throw SwiftMLError.invalidInput("No valid numeric feature columns found for t-SNE.")
        }
        
        let nRows = shape.rows
        guard nRows >= 4 else {
            throw SwiftMLError.insufficientData(minimum: 4, got: nRows)
        }
        
        var matrix = [[Double]](repeating: [Double](repeating: 0.0, count: featureNames.count), count: nRows)
        for (colIdx, name) in featureNames.enumerated() {
            guard let col = self[name] else { continue }
            if let doubles = col.toDoubles(), doubles.count == nRows {
                for r in 0..<nRows {
                    matrix[r][colIdx] = doubles[r]
                }
            } else {
                for r in 0..<nRows {
                    if let val = col.value(at: r) {
                        if let d = val as? Double {
                            matrix[r][colIdx] = d
                        } else if let i = val as? Int {
                            matrix[r][colIdx] = Double(i)
                        }
                    }
                }
            }
        }
        
        let tsneModel = TSNE(
            nComponents: nComponents,
            perplexity: perplexity,
            learningRate: learningRate,
            maxIterations: maxIterations,
            seed: seed
        )
        let coords = try await tsneModel.fitTransform(matrix)
        
        var resultColumns: [any AnyColumn] = []
        for c in 0..<nComponents {
            let colName = "tsne_\(c + 1)"
            var colVals = [Double](repeating: 0.0, count: nRows)
            for r in 0..<nRows {
                colVals[r] = coords[r][c]
            }
            resultColumns.append(TypedColumn(name: colName, values: colVals))
        }
        
        return try DataFrame(columns: resultColumns)
    }
}
#endif
