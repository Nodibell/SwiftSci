import Foundation
import Accelerate

/// Local Interpretable Model-agnostic Explanations (LIME).
///
/// LIME approximates complex black-box predictive models locally around a specific
/// target instance using an interpretable, weighted ridge linear surrogate model.
public actor LIMEExplainer {
    
    /// Result structure produced by a LIME explanation.
    public struct Explanation: Sendable, Codable {
        /// Feature importance weights (linear regression coefficients) for each feature dimension.
        public let featureWeights: [Double]
        /// Local intercept of the linear surrogate model.
        public let intercept: Double
        /// Local goodness-of-fit ($R^2$) score of the surrogate model around the perturbed neighborhood.
        public let localR2: Double
        /// Target predicted value at the original unperturbed instance.
        public let prediction: Double
    }
    
    /// Kernel width parameter $\sigma$ for exponential distance weighting.
    public let kernelWidth: Double
    /// Regularization strength $\lambda$ for the local ridge regression surrogate.
    public let regularization: Double
    
    /// Initializes a new LIME explainer.
    /// - Parameters:
    ///   - kernelWidth: Exponential kernel bandwidth (defaults to 0.75).
    ///   - regularization: Ridge regression penalty $\lambda$ (defaults to 1.0).
    public init(kernelWidth: Double = 0.75, regularization: Double = 1.0) {
        self.kernelWidth = max(1e-5, kernelWidth)
        self.regularization = max(0.0, regularization)
    }
    
    /// Explains a black-box model prediction for the target instance.
    /// - Parameters:
    ///   - model: Prediction closure mapping a feature vector `[Double]` to a scalar prediction.
    ///   - instance: The target feature vector to explain locally.
    ///   - numSamples: Number of perturbed samples to generate (defaults to 500).
    ///   - featureScales: Optional standard deviations per feature to scale perturbation noise.
    /// - Returns: A `Explanation` containing local feature weights and fidelity metrics.
    public func explain(
        model: @escaping @Sendable ([Double]) async -> Double,
        instance: [Double],
        numSamples: Int = 500,
        featureScales: [Double]? = nil
    ) async -> Explanation {
        let D = instance.count
        guard D > 0 else {
            return Explanation(featureWeights: [], intercept: 0.0, localR2: 0.0, prediction: 0.0)
        }
        
        let targetPred = await model(instance)
        let N = max(10, numSamples)
        
        let scales = featureScales ?? [Double](repeating: 1.0, count: D)
        
        // 1. Generate perturbations around the instance
        var perturbedData = [Double](repeating: 0.0, count: N * D)
        var distances = [Double](repeating: 0.0, count: N)
        var weights = [Double](repeating: 0.0, count: N)
        
        // First sample is the original instance (distance = 0)
        for d in 0..<D {
            perturbedData[d] = instance[d]
        }
        distances[0] = 0.0
        weights[0] = 1.0
        
        // Box-Muller Gaussian PRNG
        var rngState: UInt64 = 0x853c49e6748fea9b
        func nextGaussian() -> Double {
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            let u1 = max(1e-7, Double(rngState >> 11) / Double(1 << 53))
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            let u2 = Double(rngState >> 11) / Double(1 << 53)
            return sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
        }
        
        for i in 1..<N {
            var distSq = 0.0
            for d in 0..<D {
                let std = max(1e-4, scales[d])
                let noise = nextGaussian() * std
                let val = instance[d] + noise
                perturbedData[i * D + d] = val
                let normDiff = noise / std
                distSq += normDiff * normDiff
            }
            let dist = sqrt(distSq)
            distances[i] = dist
            // Exponential distance kernel w = exp(-d^2 / (2 * sigma^2))
            weights[i] = exp(-distSq / (2.0 * kernelWidth * kernelWidth))
        }
        
        // 2. Query the black-box model predictions for perturbed samples
        var targets = [Double](repeating: 0.0, count: N)
        targets[0] = targetPred
        
        for i in 1..<N {
            var sample = [Double](repeating: 0.0, count: D)
            for d in 0..<D {
                sample[d] = perturbedData[i * D + d]
            }
            targets[i] = await model(sample)
        }
        
        // 3. Fit Weighted Ridge Regression (Surrogate Linear Model)
        // X = [N x (D + 1)] including intercept column of 1.0
        let P = D + 1
        var XtWX = [Double](repeating: 0.0, count: P * P)
        var XtWy = [Double](repeating: 0.0, count: P)
        
        for i in 0..<N {
            let w = weights[i]
            let y = targets[i]
            
            // Construct row X_i = [1.0, x_i1, x_i2, ...]
            var xi = [Double](repeating: 1.0, count: P)
            for d in 0..<D {
                xi[d + 1] = perturbedData[i * D + d] - instance[d] // centered around instance
            }
            
            for j in 0..<P {
                XtWy[j] += xi[j] * w * y
                for k in 0..<P {
                    XtWX[j * P + k] += xi[j] * w * xi[k]
                }
            }
        }
        
        // Add Ridge Regularization lambda * I (excluding intercept)
        for d in 1..<P {
            XtWX[d * P + d] += regularization
        }
        
        // Solve linear system (XtWX) * beta = XtWy via Gaussian elimination with partial pivoting
        var A = XtWX
        var b = XtWy
        
        for col in 0..<P {
            var maxRow = col
            var maxVal = abs(A[col * P + col])
            for r in (col + 1)..<P {
                let v = abs(A[r * P + col])
                if v > maxVal {
                    maxVal = v
                    maxRow = r
                }
            }
            
            if maxRow != col {
                for c in 0..<P {
                    let temp = A[col * P + c]
                    A[col * P + c] = A[maxRow * P + c]
                    A[maxRow * P + c] = temp
                }
                let tempB = b[col]
                b[col] = b[maxRow]
                b[maxRow] = tempB
            }
            
            let pivot = A[col * P + col]
            if abs(pivot) > 1e-12 {
                for r in (col + 1)..<P {
                    let factor = A[r * P + col] / pivot
                    for c in col..<P {
                        A[r * P + c] -= factor * A[col * P + c]
                    }
                    b[r] -= factor * b[col]
                }
            }
        }
        
        var beta = [Double](repeating: 0.0, count: P)
        for r in stride(from: P - 1, through: 0, by: -1) {
            var sum = b[r]
            for c in (r + 1)..<P {
                sum -= A[r * P + c] * beta[c]
            }
            let diag = A[r * P + r]
            beta[r] = abs(diag) > 1e-12 ? sum / diag : 0.0
        }
        
        let intercept = beta[0]
        let featureWeights = Array(beta[1..<P])
        
        // Compute weighted R^2 score
        var meanY = 0.0
        var totalWeight = 0.0
        for i in 0..<N {
            meanY += weights[i] * targets[i]
            totalWeight += weights[i]
        }
        meanY = totalWeight > 0 ? meanY / totalWeight : 0.0
        
        var ssTot = 0.0
        var ssRes = 0.0
        for i in 0..<N {
            let y = targets[i]
            var pred = intercept
            for d in 0..<D {
                pred += featureWeights[d] * (perturbedData[i * D + d] - instance[d])
            }
            let diffTot = y - meanY
            let diffRes = y - pred
            ssTot += weights[i] * diffTot * diffTot
            ssRes += weights[i] * diffRes * diffRes
        }
        
        let localR2 = ssTot > 1e-9 ? max(0.0, min(1.0, 1.0 - (ssRes / ssTot))) : 1.0
        
        return Explanation(
            featureWeights: featureWeights,
            intercept: intercept,
            localR2: localR2,
            prediction: targetPred
        )
    }
}
