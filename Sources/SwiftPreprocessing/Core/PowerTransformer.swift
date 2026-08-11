import Foundation

/// PowerTransformer applies a power transform feature-wise to stabilize variance and make data more Gaussian-like.
public struct PowerTransformer: PreprocessingTransformer, Sendable {
    /// Represents method.
    public enum Method: String, Sendable, Codable {
        case boxCox = "box-cox"
        case yeoJohnson = "yeo-johnson"
    }
    
    /// The method.
    public let method: Method
    /// The standardize.
    public let standardize: Bool
    
    /// The lambdas.
    public private(set) var lambdas: [Double]?
    /// The means.
    public private(set) var means: [Double]?
    /// The stds.
    public private(set) var stds: [Double]?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - method: The method.
    ///   - standardize: The standardize.
    public init(method: Method = .yeoJohnson, standardize: Bool = true) {
        self.method = method
        self.standardize = standardize
    }
    
    /// Fits the PowerTransformer by finding the optimal lambda parameter for each column.
    public mutating func fit(_ data: [[Double]]) throws {

        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        
        let rowCount = data.count
        let colCount = data[0].count
        
        var estimatedLambdas = [Double](repeating: 0.0, count: colCount)
        var transformedMeans = [Double](repeating: 0.0, count: colCount)
        var transformedStds = [Double](repeating: 1.0, count: colCount)
        
        for col in 0..<colCount {
            var colValues = [Double]()
            colValues.reserveCapacity(rowCount)
            for row in 0..<rowCount {
                guard data[row].count == colCount else {
                    throw PreprocessingError.dimensionMismatch(expected: colCount, got: data[row].count)
                }
                let val = data[row][col]
                if method == .boxCox && val <= 0.0 {
                    throw PreprocessingError.invalidInput("Box-Cox transform strictly requires positive values. Column \(col) has value \(val).")
                }
                colValues.append(val)
            }
            
            let optimalLambda = estimateOptimalLambda(colValues)
            estimatedLambdas[col] = optimalLambda
            
            // If standardizing, compute the mean and std of the transformed column
            if standardize {
                let transformedVals = colValues.map { transformValue($0, lambda: optimalLambda) }
                let mean = transformedVals.reduce(0.0, +) / Double(rowCount)
                let varianceVal = transformedVals.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(rowCount)
                let std = varianceVal.squareRoot()
                transformedMeans[col] = mean
                transformedStds[col] = std < 1e-12 ? 1.0 : std
            }
        }
        
        self.lambdas = estimatedLambdas
        if standardize {
            self.means = transformedMeans
            self.stds = transformedStds
        }
    }
    
    /// Transforms the dataset using the fitted power transform.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let lambdas = self.lambdas else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = lambdas.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            
            let transformedRow = (0..<colCount).map { col -> Double in
                let val = row[col]
                let trans = transformValue(val, lambda: lambdas[col])
                if standardize, let means = self.means, let stds = self.stds {
                    return (trans - means[col]) / stds[col]
                }
                return trans
            }
            transformed.append(transformedRow)
        }
        
        return transformed
    }
    
    private func transformValue(_ x: Double, lambda: Double) -> Double {
        switch method {
        case .boxCox:
            if abs(lambda) < 1e-12 {
                return log(x)
            } else {
                return (pow(x, lambda) - 1.0) / lambda
            }
        case .yeoJohnson:
            if x >= 0.0 {
                if abs(lambda) < 1e-12 {
                    return log(x + 1.0)
                } else {
                    return (pow(x + 1.0, lambda) - 1.0) / lambda
                }
            } else {
                if abs(lambda - 2.0) < 1e-12 {
                    return -log(-x + 1.0)
                } else {
                    return -(pow(-x + 1.0, 2.0 - lambda) - 1.0) / (2.0 - lambda)
                }
            }
        }
    }
    
    private func computeLogLikelihood(_ values: [Double], lambda: Double) -> Double? {
        let n = Double(values.count)
        var transformed = [Double]()
        transformed.reserveCapacity(values.count)
        
        for x in values {
            if method == .boxCox && x <= 0.0 {
                return nil
            }
            transformed.append(transformValue(x, lambda: lambda))
        }
        
        let mean = transformed.reduce(0.0, +) / n
        let sumSq = transformed.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) }
        let variance = sumSq / n
        
        guard variance > 1e-12 else {
            return -Double.greatestFiniteMagnitude
        }
        
        var jacobianSum = 0.0
        switch method {
        case .boxCox:
            jacobianSum = values.reduce(0.0) { acc, val in acc + log(val) }
        case .yeoJohnson:
            jacobianSum = values.reduce(0.0) { acc, val in
                acc + (val >= 0.0 ? 1.0 : -1.0) * log(abs(val) + 1.0)
            }
        }
        
        return -0.5 * n * log(variance) + (lambda - 1.0) * jacobianSum
    }
    
    private func estimateOptimalLambda(_ values: [Double]) -> Double {
        var bestLambda = 0.0
        var maxLogLikelihood = -Double.greatestFiniteMagnitude
        
        // 1. Grid search from -2.0 to 2.0 with step 0.1
        var candidates = [Double]()
        for i in -20...20 {
            candidates.append(Double(i) * 0.1)
        }
        
        for l in candidates {
            if let ll = computeLogLikelihood(values, lambda: l) {
                if ll > maxLogLikelihood {
                    maxLogLikelihood = ll
                    bestLambda = l
                }
            }
        }
        
        // 2. Local refinement around the best candidate (step 0.01)
        let center = bestLambda
        for i in -10...10 {
            let l = center + Double(i) * 0.01
            if let ll = computeLogLikelihood(values, lambda: l) {
                if ll > maxLogLikelihood {
                    maxLogLikelihood = ll
                    bestLambda = l
                }
            }
        }
        
        return bestLambda
    }
}
