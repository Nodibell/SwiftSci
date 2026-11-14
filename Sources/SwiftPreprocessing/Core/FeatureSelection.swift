import Foundation
import SwiftStats

private func validateFeatureMatrix(_ data: [[Double]], requireFinite: Bool = true) throws -> (rows: Int, cols: Int) {
    guard let first = data.first, !first.isEmpty else {
        throw PreprocessingError.emptyInput
    }
    let cols = first.count
    for row in data {
        guard row.count == cols else {
            throw PreprocessingError.dimensionMismatch(expected: cols, got: row.count)
        }
        if requireFinite {
            for value in row {
                if value.isNaN { throw PreprocessingError.containsNaN }
                if !value.isFinite { throw PreprocessingError.containsInfinity }
            }
        }
    }
    return (data.count, cols)
}

private func validateFiniteVector(_ values: [Double]) throws {
    for value in values {
        if value.isNaN { throw PreprocessingError.containsNaN }
        if !value.isFinite { throw PreprocessingError.containsInfinity }
    }
}

/// Computes origin-shifted, scale-normalized population variance and standard deviation (spread)
/// for a single feature column so large offsets keep small differences and ranking stays finite.
private func stableFeatureMoments(in data: [[Double]], column index: Int) -> (variance: Double, spread: Double) {
    let n = Double(data.count)
    let origin = data[0][index]
    var shifted = [Double]()
    shifted.reserveCapacity(data.count)
    var overflowsShift = false
    for row in data {
        let diff = row[index] - origin
        if !diff.isFinite {
            overflowsShift = true
            break
        }
        shifted.append(diff)
    }

    let scale: Double
    let working: [Double]
    if overflowsShift {
        var maxAbs = 0.0
        for row in data { maxAbs = max(maxAbs, abs(row[index])) }
        guard maxAbs > 0 else { return (0.0, 0.0) }
        let scaledOrigin = origin / maxAbs
        working = data.map { $0[index] / maxAbs - scaledOrigin }
        scale = maxAbs
    } else {
        var maxAbs = 0.0
        for val in shifted { maxAbs = max(maxAbs, abs(val)) }
        guard maxAbs > 0 else { return (0.0, 0.0) }
        working = shifted.map { $0 / maxAbs }
        scale = maxAbs
    }

    var mean = 0.0
    for val in working { mean += val }
    mean /= n

    var sumSq = 0.0
    for val in working {
        let d = val - mean
        sumSq += d * d
    }
    let normalizedVariance = sumSq / n
    return (scale * (scale * normalizedVariance), scale * sqrt(normalizedVariance))
}

/// Feature selector that removes all features whose variance does not meet a threshold.
public struct VarianceThreshold: PreprocessingTransformer, Sendable {
    /// The threshold.
    public let threshold: Double
    private var selectedIndices: [Int]?
    private var inputFeatureCount: Int?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - threshold: The threshold.
    public init(threshold: Double = 0.0) {
        self.threshold = threshold
    }
    
    /// Fit.
    /// - Throws: An error if the operation fails.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    public mutating func fit(_ data: [[Double]]) throws {
        guard threshold.isFinite, threshold >= 0 else {
            throw PreprocessingError.invalidInput("Variance threshold must be finite and nonnegative")
        }
        let (_, numFeatures) = try validateFeatureMatrix(data)
        
        var selected: [Int] = []
        for f in 0..<numFeatures {
            let variance = stableFeatureMoments(in: data, column: f).variance
            if variance > threshold {
                selected.append(f)
            }
        }
        self.selectedIndices = selected
        self.inputFeatureCount = numFeatures
    }
    
    /// Transform.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let indices = selectedIndices, let expected = inputFeatureCount else {
            throw PreprocessingError.fitNotCalled
        }
        let (_, cols) = try validateFeatureMatrix(data)
        guard cols == expected else {
            throw PreprocessingError.dimensionMismatch(expected: expected, got: cols)
        }
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
public struct SelectKBest: PreprocessingTransformer, Sendable {
    /// The k.
    public let k: Int
    private var selectedIndices: [Int]?
    private var inputFeatureCount: Int?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - k: The k.
    public init(k: Int) {
        self.k = k
    }
    
    /// Fit.
    /// - Throws: An error if the operation fails.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    public mutating func fit(_ data: [[Double]]) throws {
        try fit(features: data, targets: nil)
    }
    
    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    public mutating func fit(features: [[Double]], targets: [Double]?) throws {
        guard k >= 0 else {
            throw PreprocessingError.invalidInput("k must be non-negative")
        }
        let (numRows, numFeatures) = try validateFeatureMatrix(features)
        if let targets {
            guard targets.count == numRows else {
                throw PreprocessingError.dimensionMismatch(expected: numRows, got: targets.count)
            }
            try validateFiniteVector(targets)
        }
        let kActual = min(k, numFeatures)
        guard kActual > 0 else {
            self.selectedIndices = []
            self.inputFeatureCount = numFeatures
            return
        }
        
        var scores: [(index: Int, score: Double)] = []
        scores.reserveCapacity(numFeatures)
        
        if let targets {
            // Supervised: ANOVA F-value per feature (groups = target classes)
            var classIndices: [Double: [Int]] = [:]
            for (i, t) in targets.enumerated() {
                let canonicalTarget = t == 0.0 ? 0.0 : t
                classIndices[canonicalTarget, default: []].append(i)
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
                    // Fallback to variance/spread when ANOVA cannot be computed
                    // (e.g. only one target class present in the dataset)
                    let moments = stableFeatureMoments(in: features, column: f)
                    score = moments.variance.isFinite ? moments.variance : moments.spread
                }
                scores.append((f, score))
            }
        } else {
            // Unsupervised fallback: rank by feature spread (monotone in variance, overflow-safe)
            for f in 0..<numFeatures {
                scores.append((f, stableFeatureMoments(in: features, column: f).spread))
            }
        }
        
        scores.sort {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.index < $1.index
        }
        self.selectedIndices = scores.prefix(kActual).map { $0.index }.sorted()
        self.inputFeatureCount = numFeatures
    }
    
    /// Transform.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let indices = selectedIndices, let expected = inputFeatureCount else {
            throw PreprocessingError.fitNotCalled
        }
        let (_, cols) = try validateFeatureMatrix(data)
        guard cols == expected else {
            throw PreprocessingError.dimensionMismatch(expected: expected, got: cols)
        }
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}

/// Recursive Feature Elimination (RFE) selector.
/// Iteratively fits feature importance/variance scores and eliminates low-ranking features
/// until the target `nFeaturesToSelect` features remain.
public struct RecursiveFeatureElimination: PreprocessingTransformer, Sendable {
    /// The n features to select.
    public let nFeaturesToSelect: Int
    /// The step.
    public let step: Int
    private var selectedIndices: [Int]?
    private var inputFeatureCount: Int?
    /// The ranking.
    public private(set) var ranking: [Int] = []
    /// The support.
    public private(set) var support: [Bool] = []

    /// Creates a new instance.
    /// - Parameters:
    ///   - nFeaturesToSelect: The n features to select.
    ///   - step: The step.
    public init(nFeaturesToSelect: Int, step: Int = 1) {
        self.nFeaturesToSelect = nFeaturesToSelect
        self.step = step
    }

    /// Fit.
    /// - Throws: An error if the operation fails.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    public mutating func fit(_ data: [[Double]]) throws {
        try fit(features: data, featureImportances: nil)
    }

    /// Fit.
    /// - Parameters:
    ///   - features: The features.
    ///   - featureImportances: The feature importances.
    /// - Throws: An error if the operation fails.
    public mutating func fit(features: [[Double]], featureImportances: [Double]? = nil) throws {
        guard nFeaturesToSelect >= 1 else {
            throw PreprocessingError.invalidInput("nFeaturesToSelect must be at least 1")
        }
        guard step >= 1 else {
            throw PreprocessingError.invalidInput("step must be at least 1")
        }
        let (_, totalFeatures) = try validateFeatureMatrix(features)
        if let featureImportances {
            guard featureImportances.count == totalFeatures else {
                throw PreprocessingError.dimensionMismatch(expected: totalFeatures, got: featureImportances.count)
            }
            try validateFiniteVector(featureImportances)
        }
        let targetCount = min(nFeaturesToSelect, totalFeatures)

        var activeIndices = Array(0..<totalFeatures)
        var featureRankings = [Int](repeating: 1, count: totalFeatures)
        var currentRank = totalFeatures

        let currentImportances = featureImportances ?? (0..<totalFeatures).map {
            stableFeatureMoments(in: features, column: $0).spread
        }

        while activeIndices.count > targetCount {
            let numEliminate = min(step, activeIndices.count - targetCount)
            
            let activeScores = activeIndices.map { idx in
                (index: idx, score: currentImportances[idx])
            }.sorted {
                if $0.score != $1.score { return $0.score < $1.score }
                return $0.index < $1.index
            }

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
        self.inputFeatureCount = totalFeatures
    }

    /// Transform.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    /// - Parameters:
    ///   - data: Raw input data array or matrix for transformation.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let indices = selectedIndices, let expected = inputFeatureCount else {
            throw PreprocessingError.fitNotCalled
        }
        let (_, cols) = try validateFeatureMatrix(data)
        guard cols == expected else {
            throw PreprocessingError.dimensionMismatch(expected: expected, got: cols)
        }
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}
