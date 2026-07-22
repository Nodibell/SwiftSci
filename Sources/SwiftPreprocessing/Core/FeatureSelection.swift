import Foundation

/// Feature selector that removes all features whose variance does not meet a threshold.
public final class VarianceThreshold: PreprocessingTransformer, @unchecked Sendable {
    public let threshold: Double
    private var selectedIndices: [Int] = []
    
    public init(threshold: Double = 0.0) {
        self.threshold = threshold
    }
    
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let numSamples = Double(data.count)
        let numFeatures = data[0].count
        
        var selected: [Int] = []
        for f in 0..<numFeatures {
            let col = data.map { $0[f] }
            let mean = col.reduce(0.0, +) / numSamples
            let variance = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
            if variance > threshold {
                selected.append(f)
            }
        }
        self.selectedIndices = selected
    }
    
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let indices = selectedIndices.isEmpty ? Array(0..<data[0].count) : selectedIndices
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}

/// Feature selector that retains the K highest scoring features according to ANOVA F-value / variance.
public final class SelectKBest: PreprocessingTransformer, @unchecked Sendable {
    public let k: Int
    private var selectedIndices: [Int] = []
    
    public init(k: Int) {
        self.k = k
    }
    
    public func fit(_ data: [[Double]]) throws {
        try fit(features: data, targets: nil)
    }
    
    public func fit(features: [[Double]], targets: [Double]?) throws {
        guard !features.isEmpty, !features[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let numFeatures = features[0].count
        let numSamples = Double(features.count)
        let kActual = min(k, numFeatures)
        
        var scores: [(index: Int, score: Double)] = []
        
        for f in 0..<numFeatures {
            let col = features.map { $0[f] }
            let mean = col.reduce(0.0, +) / numSamples
            let variance = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
            scores.append((f, variance))
        }
        
        scores.sort { $0.score > $1.score }
        self.selectedIndices = scores.prefix(kActual).map { $0.index }.sorted()
    }
    
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let indices = selectedIndices.isEmpty ? Array(0..<min(k, data[0].count)) : selectedIndices
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}
