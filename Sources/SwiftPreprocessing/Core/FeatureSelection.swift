import Foundation
import SwiftStats

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

/// Feature selector that retains the K highest scoring features.
///
/// When `targets` are provided to `fit(features:targets:)`, features are ranked
/// by ANOVA F-value (supervised selection). When targets are `nil`, falls back
/// to variance-based ranking (unsupervised).
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
        
        if let targets {
            // Supervised: ANOVA F-value per feature (groups = target classes)
            var classIndices: [Double: [Int]] = [:]
            for (i, t) in targets.enumerated() {
                classIndices[t, default: []].append(i)
            }
            
            for f in 0..<numFeatures {
                let groups: [[Double]] = classIndices.values.map { indices in
                    indices.map { features[$0][f] }
                }
                let score: Double
                if let result = try? Stats.oneWayANOVA(groups: groups),
                   !result.fStatistic.isNaN, !result.fStatistic.isInfinite {
                    score = result.fStatistic
                } else {
                    // Fallback to variance when ANOVA cannot be computed
                    // (e.g. only one target class present in the dataset)
                    let col = features.map { $0[f] }
                    let mean = col.reduce(0.0, +) / numSamples
                    score = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
                }
                scores.append((f, score))
            }
        } else {
            // Unsupervised fallback: rank by feature variance
            for f in 0..<numFeatures {
                let col = features.map { $0[f] }
                let mean = col.reduce(0.0, +) / numSamples
                let variance = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / numSamples
                scores.append((f, variance))
            }
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

/// Recursive Feature Elimination (RFE) selector.
/// Iteratively fits feature importance/variance scores and eliminates low-ranking features
/// until the target `nFeaturesToSelect` features remain.
public final class RecursiveFeatureElimination: PreprocessingTransformer, @unchecked Sendable {
    public let nFeaturesToSelect: Int
    public let step: Int
    private var selectedIndices: [Int] = []
    public private(set) var ranking: [Int] = []
    public private(set) var support: [Bool] = []

    public init(nFeaturesToSelect: Int, step: Int = 1) {
        self.nFeaturesToSelect = max(1, nFeaturesToSelect)
        self.step = max(1, step)
    }

    public func fit(_ data: [[Double]]) throws {
        try fit(features: data, featureImportances: nil)
    }

    public func fit(features: [[Double]], featureImportances: [Double]? = nil) throws {
        guard !features.isEmpty, !features[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let totalFeatures = features[0].count
        let targetCount = min(nFeaturesToSelect, totalFeatures)

        var activeIndices = Array(0..<totalFeatures)
        var featureRankings = [Int](repeating: 1, count: totalFeatures)
        var currentRank = totalFeatures

        let currentImportances = featureImportances ?? computeVariances(features: features)

        while activeIndices.count > targetCount {
            let numEliminate = min(step, activeIndices.count - targetCount)
            
            let activeScores = activeIndices.map { idx in
                (index: idx, score: idx < currentImportances.count ? currentImportances[idx] : 0.0)
            }.sorted { $0.score < $1.score }

            let toEliminate = activeScores.prefix(numEliminate)
            let eliminatedIndices = Set(toEliminate.map { $0.index })

            for item in toEliminate {
                featureRankings[item.index] = currentRank
            }
            currentRank -= numEliminate

            activeIndices.removeAll { eliminatedIndices.contains($0) }
        }

        for idx in activeIndices {
            featureRankings[idx] = 1
        }

        self.ranking = featureRankings
        self.support = featureRankings.map { $0 == 1 }
        self.selectedIndices = activeIndices.sorted()
    }

    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard !data.isEmpty else { throw PreprocessingError.emptyInput }
        let indices = selectedIndices.isEmpty ? Array(0..<min(nFeaturesToSelect, data[0].count)) : selectedIndices
        return data.map { row in
            indices.map { row[$0] }
        }
    }

    private func computeVariances(features: [[Double]]) -> [Double] {
        let n = Double(features.count)
        let numFeatures = features[0].count
        var variances = [Double](repeating: 0.0, count: numFeatures)
        for f in 0..<numFeatures {
            let col = features.map { $0[f] }
            let mean = col.reduce(0.0, +) / n
            variances[f] = col.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / n
        }
        return variances
    }
}
