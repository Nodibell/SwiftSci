import Foundation

/// Container for resampled dataset features and targets.
public struct ResampledDataset: Sendable {
    /// The features.
    public let features: [[Double]]
    /// The targets.
    public let targets: [Double]
    /// Creates a new instance.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    public init(features: [[Double]], targets: [Double]) {
        self.features = features
        self.targets = targets
    }
}

/// Unified protocol for dataset resampling estimators.
public protocol ResamplingEstimator: Sendable {
    /// Resamples features and targets to address class imbalance.
    func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset
}

/// Synthetic Minority Over-sampling Technique (SMOTE) for imbalanced classification datasets.
public final class SMOTE: Sendable, ResamplingEstimator {
    /// The k neighbors.
    public let kNeighbors: Int
    /// The seed.
    public let seed: UInt64
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - kNeighbors: The k neighbors.
    ///   - seed: The seed.
    public init(kNeighbors: Int = 5, seed: UInt64 = 42) {
        self.kNeighbors = max(1, kNeighbors)
        self.seed = seed
    }
    
    /// Resamples minority class instances by generating synthetic samples along line segments connecting k-nearest neighbors.
    /// - Parameters:
    ///   - features: 2D array of input feature vectors of shape `[N, P]`.
    ///   - targets: 1D array of ground-truth target values of length `N`.
    /// - Throws: `PreprocessingError` or `SwiftMLError` if columns are missing, types are invalid, or arrays are empty.
    /// - Returns: The computed ResampledDataset result instance.
    public func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset {
        guard !features.isEmpty, features.count == targets.count else {
            throw PreprocessingError.emptyInput
        }
        
        let numFeatures = features[0].count
        var classCounts: [Double: Int] = [:]
        var classIndices: [Double: [Int]] = [:]
        
        for (i, t) in targets.enumerated() {
            classCounts[t, default: 0] += 1
            classIndices[t, default: []].append(i)
        }
        
        guard let maxCount = classCounts.values.max() else {
            return ResampledDataset(features: features, targets: targets)
        }
        
        var syntheticFeatures = features
        var syntheticTargets = targets
        var rng = SeededRandom(seed: Int(self.seed))
        
        for (cls, indices) in classIndices {
            let count = indices.count
            guard count < maxCount else { continue }
            let samplesToGenerate = maxCount - count
            guard count > 1 else { continue }
            let k = min(kNeighbors, count - 1)
            
            let classFeatures = indices.map { features[$0] }
            
            for _ in 0..<samplesToGenerate {
                let idx = rng.nextInt(upperBound: count)
                let baseSample = classFeatures[idx]
                
                // Find k-nearest neighbors within minority class
                var distances: [(index: Int, dist: Double)] = []
                for (otherIdx, otherSample) in classFeatures.enumerated() where otherIdx != idx {
                    var d = 0.0
                    for f in 0..<numFeatures {
                        let diff = baseSample[f] - otherSample[f]
                        d += diff * diff
                    }
                    distances.append((otherIdx, d))
                }
                
                distances.sort { $0.dist < $1.dist }
                let neighborIdx = distances[rng.nextInt(upperBound: k)].index
                let neighborSample = classFeatures[neighborIdx]
                
                // Interpolate along line segment using seeded random gap
                let gap = rng.nextDouble()
                var newSample = [Double](repeating: 0.0, count: numFeatures)
                for f in 0..<numFeatures {
                    newSample[f] = baseSample[f] + gap * (neighborSample[f] - baseSample[f])
                }
                
                syntheticFeatures.append(newSample)
                syntheticTargets.append(cls)
            }
        }
        
        return ResampledDataset(features: syntheticFeatures, targets: syntheticTargets)
    }
}

/// Random Undersampler for balancing class distribution by random sub-sampling of majority classes.
public final class RandomUndersampler: Sendable, ResamplingEstimator {
    /// The seed.
    public let seed: UInt64
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - seed: The seed.
    public init(seed: UInt64 = 42) {
        self.seed = seed
    }
    
    /// Fit resample.
    /// - Parameters:
    ///   - features: The features.
    ///   - targets: The targets.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `ResampledDataset` result.
    public func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset {
        guard !features.isEmpty, features.count == targets.count else {
            throw PreprocessingError.emptyInput
        }
        
        var classIndices: [Double: [Int]] = [:]
        for (i, t) in targets.enumerated() {
            classIndices[t, default: []].append(i)
        }
        
        guard let minCount = classIndices.values.map({ $0.count }).min() else {
            return ResampledDataset(features: features, targets: targets)
        }
        
        var rng = SeededRandom(seed: Int(self.seed))
        var resampledIndices: [Int] = []
        
        for (_, indices) in classIndices {
            // Seeded Fisher-Yates shuffle
            var shuffled = indices
            for i in stride(from: shuffled.count - 1, through: 1, by: -1) {
                let j = rng.nextInt(upperBound: i + 1)
                shuffled.swapAt(i, j)
            }
            resampledIndices.append(contentsOf: shuffled.prefix(minCount))
        }
        
        resampledIndices.sort()
        let resampledFeatures = resampledIndices.map { features[$0] }
        let resampledTargets = resampledIndices.map { targets[$0] }
        
        return ResampledDataset(features: resampledFeatures, targets: resampledTargets)
    }
}

/// Borderline-SMOTE for synthesizing minority instances strictly along the decision boundary.
public final class BorderlineSMOTE: Sendable, ResamplingEstimator {
    /// Number of nearest neighbors for synthetic generation.
    public let kNeighbors: Int
    /// Number of nearest neighbors to evaluate boundary danger status.
    public let mNeighbors: Int
    /// Seed for reproducible generation.
    public let seed: UInt64
    
    /// Initializes a new BorderlineSMOTE instance.
    /// - Parameters:
    ///   - kNeighbors: Number of nearest neighbors within minority class for interpolation. Defaults to 5.
    ///   - mNeighbors: Number of nearest neighbors in whole dataset to evaluate boundary. Defaults to 10.
    ///   - seed: Random state seed. Defaults to 42.
    public init(kNeighbors: Int = 5, mNeighbors: Int = 10, seed: UInt64 = 42) {
        self.kNeighbors = max(1, kNeighbors)
        self.mNeighbors = max(1, mNeighbors)
        self.seed = seed
    }
    
    public func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset {
        guard !features.isEmpty, features.count == targets.count else {
            throw PreprocessingError.emptyInput
        }
        
        let numFeatures = features[0].count
        var classCounts: [Double: Int] = [:]
        var classIndices: [Double: [Int]] = [:]
        
        for (i, t) in targets.enumerated() {
            classCounts[t, default: 0] += 1
            classIndices[t, default: []].append(i)
        }
        
        guard let maxCount = classCounts.values.max() else {
            return ResampledDataset(features: features, targets: targets)
        }
        
        var syntheticFeatures = features
        var syntheticTargets = targets
        var rng = SeededRandom(seed: Int(self.seed))
        let nTotal = features.count
        
        for (cls, indices) in classIndices {
            let count = indices.count
            guard count < maxCount, count > 1 else { continue }
            let samplesToGenerate = maxCount - count
            
            let classFeatures = indices.map { features[$0] }
            let m = min(mNeighbors, nTotal - 1)
            
            // Identify DANGER instances
            var dangerIndices: [Int] = []
            for (idxInClass, sampleIdx) in indices.enumerated() {
                let sample = features[sampleIdx]
                var dists: [(target: Double, dist: Double)] = []
                for (otherIdx, otherSample) in features.enumerated() where otherIdx != sampleIdx {
                    var d = 0.0
                    for f in 0..<numFeatures {
                        let diff = sample[f] - otherSample[f]
                        d += diff * diff
                    }
                    dists.append((targets[otherIdx], d))
                }
                dists.sort { $0.dist < $1.dist }
                let mNearest = dists.prefix(m)
                let majorityNeighbors = mNearest.filter { $0.target != cls }.count
                
                // DANGER if at least half of neighbors are majority class, but not all (not pure noise)
                if majorityNeighbors >= m / 2 && majorityNeighbors < m {
                    dangerIndices.append(idxInClass)
                }
            }
            
            // Fallback to all class instances if no borderline instances found
            let baseCandidates = dangerIndices.isEmpty ? Array(0..<count) : dangerIndices
            let k = min(kNeighbors, count - 1)
            
            for _ in 0..<samplesToGenerate {
                let chosenIdx = baseCandidates[rng.nextInt(upperBound: baseCandidates.count)]
                let baseSample = classFeatures[chosenIdx]
                
                var distances: [(index: Int, dist: Double)] = []
                for (otherIdx, otherSample) in classFeatures.enumerated() where otherIdx != chosenIdx {
                    var d = 0.0
                    for f in 0..<numFeatures {
                        let diff = baseSample[f] - otherSample[f]
                        d += diff * diff
                    }
                    distances.append((otherIdx, d))
                }
                distances.sort { $0.dist < $1.dist }
                let neighborIdx = distances[rng.nextInt(upperBound: k)].index
                let neighborSample = classFeatures[neighborIdx]
                
                let gap = rng.nextDouble()
                var newSample = [Double](repeating: 0.0, count: numFeatures)
                for f in 0..<numFeatures {
                    newSample[f] = baseSample[f] + gap * (neighborSample[f] - baseSample[f])
                }
                syntheticFeatures.append(newSample)
                syntheticTargets.append(cls)
            }
        }
        
        return ResampledDataset(features: syntheticFeatures, targets: syntheticTargets)
    }
}

/// Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets.
public final class ADASYN: Sendable, ResamplingEstimator {
    /// Number of nearest neighbors.
    public let kNeighbors: Int
    /// Seed for reproducible generation.
    public let seed: UInt64
    
    /// Initializes a new ADASYN instance.
    /// - Parameters:
    ///   - kNeighbors: Number of nearest neighbors. Defaults to 5.
    ///   - seed: Random state seed. Defaults to 42.
    public init(kNeighbors: Int = 5, seed: UInt64 = 42) {
        self.kNeighbors = max(1, kNeighbors)
        self.seed = seed
    }
    
    public func fitResample(features: [[Double]], targets: [Double]) throws -> ResampledDataset {
        guard !features.isEmpty, features.count == targets.count else {
            throw PreprocessingError.emptyInput
        }
        
        let numFeatures = features[0].count
        var classCounts: [Double: Int] = [:]
        var classIndices: [Double: [Int]] = [:]
        
        for (i, t) in targets.enumerated() {
            classCounts[t, default: 0] += 1
            classIndices[t, default: []].append(i)
        }
        
        guard let maxCount = classCounts.values.max() else {
            return ResampledDataset(features: features, targets: targets)
        }
        
        var syntheticFeatures = features
        var syntheticTargets = targets
        var rng = SeededRandom(seed: Int(self.seed))
        let nTotal = features.count
        
        for (cls, indices) in classIndices {
            let count = indices.count
            guard count < maxCount, count > 1 else { continue }
            let totalToGenerate = maxCount - count
            let classFeatures = indices.map { features[$0] }
            let k = min(kNeighbors, nTotal - 1)
            
            var rValues = [Double](repeating: 0.0, count: count)
            for (idxInClass, sampleIdx) in indices.enumerated() {
                let sample = features[sampleIdx]
                var dists: [(target: Double, dist: Double)] = []
                for (otherIdx, otherSample) in features.enumerated() where otherIdx != sampleIdx {
                    var d = 0.0
                    for f in 0..<numFeatures {
                        let diff = sample[f] - otherSample[f]
                        d += diff * diff
                    }
                    dists.append((targets[otherIdx], d))
                }
                dists.sort { $0.dist < $1.dist }
                let kNearest = dists.prefix(k)
                let nonClassCount = kNearest.filter { $0.target != cls }.count
                rValues[idxInClass] = Double(nonClassCount) / Double(k)
            }
            
            let sumR = rValues.reduce(0.0, +)
            guard sumR > 0 else {
                let smote = SMOTE(kNeighbors: kNeighbors, seed: seed)
                let subResampled = try smote.fitResample(features: features, targets: targets)
                return subResampled
            }
            
            let normalizedR = rValues.map { $0 / sumR }
            let kMinority = min(kNeighbors, count - 1)
            
            for idxInClass in 0..<count {
                let numSamplesI = Int(round(normalizedR[idxInClass] * Double(totalToGenerate)))
                guard numSamplesI > 0 else { continue }
                
                let baseSample = classFeatures[idxInClass]
                var distances: [(index: Int, dist: Double)] = []
                for (otherIdx, otherSample) in classFeatures.enumerated() where otherIdx != idxInClass {
                    var d = 0.0
                    for f in 0..<numFeatures {
                        let diff = baseSample[f] - otherSample[f]
                        d += diff * diff
                    }
                    distances.append((otherIdx, d))
                }
                distances.sort { $0.dist < $1.dist }
                
                for _ in 0..<numSamplesI {
                    let neighborIdx = distances[rng.nextInt(upperBound: kMinority)].index
                    let neighborSample = classFeatures[neighborIdx]
                    
                    let gap = rng.nextDouble()
                    var newSample = [Double](repeating: 0.0, count: numFeatures)
                    for f in 0..<numFeatures {
                        newSample[f] = baseSample[f] + gap * (neighborSample[f] - baseSample[f])
                    }
                    syntheticFeatures.append(newSample)
                    syntheticTargets.append(cls)
                }
            }
        }
        
        return ResampledDataset(features: syntheticFeatures, targets: syntheticTargets)
    }
}

import SwiftDataFrame

extension DataFrame {
    /// Resamples the DataFrame using a specified resampling estimator (e.g. SMOTE, BorderlineSMOTE, ADASYN, RandomUndersampler).
    /// - Parameters:
    ///   - resampler: Algorithm conforming to `ResamplingEstimator`.
    ///   - targetColumn: Name of the column containing class labels.
    /// - Throws: `PreprocessingError` if column not found or dimensions mismatch.
    /// - Returns: A new balanced `DataFrame`.
    public func resample(
        using resampler: ResamplingEstimator,
        targetColumn: String
    ) throws -> DataFrame {
        guard let targetCol = self[targetColumn] else {
            throw PreprocessingError.columnNotFound(targetColumn)
        }
        
        let featureNames = columnNames.filter { $0 != targetColumn }
        guard !featureNames.isEmpty else {
            throw PreprocessingError.invalidInput("DataFrame must have at least one feature column besides targetColumn.")
        }
        
        let nRows = shape.rows
        var featureMatrix = [[Double]](repeating: [Double](repeating: 0.0, count: featureNames.count), count: nRows)
        
        for (colIdx, colName) in featureNames.enumerated() {
            guard let col = self[colName] else { continue }
            if let doubles = col.toDoubles(), doubles.count == nRows {
                for row in 0..<nRows {
                    featureMatrix[row][colIdx] = doubles[row]
                }
            } else {
                for row in 0..<nRows {
                    if let val = col.value(at: row) {
                        if let d = val as? Double {
                            featureMatrix[row][colIdx] = d
                        } else if let i = val as? Int {
                            featureMatrix[row][colIdx] = Double(i)
                        } else if let str = val as? String, let d = Double(str) {
                            featureMatrix[row][colIdx] = d
                        }
                    }
                }
            }
        }
        
        var numericTargets = [Double](repeating: 0.0, count: nRows)
        var stringLabelMap: [String: Double] = [:]
        var reverseLabelMap: [Double: String] = [:]
        var isStringTarget = false
        
        if let doubles = targetCol.toDoubles(), doubles.count == nRows {
            numericTargets = doubles
        } else {
            let strVals = targetCol.toStrings()
            isStringTarget = true
            for row in 0..<min(nRows, strVals.count) {
                let str = strVals[row]
                if let mapped = stringLabelMap[str] {
                    numericTargets[row] = mapped
                } else {
                    let newCode = Double(stringLabelMap.count)
                    stringLabelMap[str] = newCode
                    reverseLabelMap[newCode] = str
                    numericTargets[row] = newCode
                }
            }
        }
        
        let resampled = try resampler.fitResample(features: featureMatrix, targets: numericTargets)
        let nResampledRows = resampled.features.count
        
        var newColumns: [AnyColumn] = []
        for (colIdx, colName) in featureNames.enumerated() {
            var colVals = [Double?]()
            colVals.reserveCapacity(nResampledRows)
            for row in 0..<nResampledRows {
                colVals.append(resampled.features[row][colIdx])
            }
            newColumns.append(TypedColumn<Double>(name: colName, values: colVals))
        }
        
        if isStringTarget {
            var targetVals = [String?]()
            targetVals.reserveCapacity(nResampledRows)
            for row in 0..<nResampledRows {
                let code = round(resampled.targets[row])
                targetVals.append(reverseLabelMap[code] ?? "\(Int(code))")
            }
            newColumns.append(TypedColumn<String>(name: targetColumn, values: targetVals))
        } else {
            var targetVals = [Double?]()
            targetVals.reserveCapacity(nResampledRows)
            for row in 0..<nResampledRows {
                targetVals.append(resampled.targets[row])
            }
            newColumns.append(TypedColumn<Double>(name: targetColumn, values: targetVals))
        }
        
        return try DataFrame(columns: newColumns)
    }
}
