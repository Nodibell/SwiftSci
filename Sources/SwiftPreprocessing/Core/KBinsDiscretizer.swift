import Foundation

/// KBinsDiscretizer bins continuous data into intervals.
public final class KBinsDiscretizer: PreprocessingTransformer, @unchecked Sendable {
    /// Represents strategy.
    public enum Strategy: String, Sendable, Codable {
        case uniform
        case quantile
    }
    
    /// Represents encode.
    public enum Encode: String, Sendable, Codable {
        case ordinal
        case oneHot = "onehot"
    }
    
    /// The n bins.
    public let nBins: Int
    /// The strategy.
    public let strategy: Strategy
    /// The encode.
    public let encode: Encode
    
    /// The bin edges.
    public private(set) var binEdges: [[Double]]?
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - nBins: The n bins.
    ///   - strategy: The strategy.
    ///   - encode: The encode.
    public init(nBins: Int = 5, strategy: Strategy = .uniform, encode: Encode = .ordinal) {
        self.nBins = nBins
        self.strategy = strategy
        self.encode = encode
    }
    
    /// Fits the discretizer by calculating the bin edges for each column.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        guard nBins >= 2 else {
            throw PreprocessingError.invalidInput("nBins must be at least 2.")
        }
        
        let rowCount = data.count
        let colCount = data[0].count
        var computedEdges = [[Double]]()
        computedEdges.reserveCapacity(colCount)
        
        for col in 0..<colCount {
            var colValues = [Double]()
            colValues.reserveCapacity(rowCount)
            for row in 0..<rowCount {
                guard data[row].count == colCount else {
                    throw PreprocessingError.dimensionMismatch(expected: colCount, got: data[row].count)
                }
                colValues.append(data[row][col])
            }
            
            var edges = [Double]()
            edges.reserveCapacity(nBins + 1)
            
            switch strategy {
            case .uniform:
                let minVal = colValues.min() ?? 0.0
                let maxVal = colValues.max() ?? 0.0
                if maxVal - minVal < 1e-12 {
                    edges = [Double](repeating: minVal, count: nBins + 1)
                } else {
                    let step = (maxVal - minVal) / Double(nBins)
                    for i in 0...nBins {
                        edges.append(minVal + Double(i) * step)
                    }
                }
            case .quantile:
                colValues.sort()
                for i in 0...nBins {
                    let q = Double(i) * 100.0 / Double(nBins)
                    edges.append(percentile(colValues, q))
                }
            }
            
            computedEdges.append(edges)
        }
        
        self.binEdges = computedEdges
    }
    
    /// Transforms the dataset into binned/discretized features.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard let allEdges = self.binEdges else {
            throw PreprocessingError.fitNotCalled
        }
        guard !data.isEmpty else {
            return []
        }
        
        let colCount = allEdges.count
        var transformed = [[Double]]()
        transformed.reserveCapacity(data.count)
        
        for row in data {
            guard row.count == colCount else {
                throw PreprocessingError.dimensionMismatch(expected: colCount, got: row.count)
            }
            
            switch encode {
            case .ordinal:
                let ordinalRow = (0..<colCount).map { col -> Double in
                    let val = row[col]
                    let edges = allEdges[col]
                    return Double(digitize(val, edges: edges))
                }
                transformed.append(ordinalRow)
            case .oneHot:
                var oneHotRow = [Double]()
                oneHotRow.reserveCapacity(colCount * nBins)
                for col in 0..<colCount {
                    let val = row[col]
                    let edges = allEdges[col]
                    let binIdx = digitize(val, edges: edges)
                    
                    for bin in 0..<nBins {
                        oneHotRow.append(bin == binIdx ? 1.0 : 0.0)
                    }
                }
                transformed.append(oneHotRow)
            }
        }
        
        return transformed
    }
    
    private func digitize(_ val: Double, edges: [Double]) -> Int {
        if val <= edges[0] {
            return 0
        }
        if val >= edges[edges.count - 1] {
            return nBins - 1
        }
        for i in 0..<(edges.count - 1) {
            if val >= edges[i] && val < edges[i + 1] {
                return i
            }
        }
        return nBins - 1
    }
    
    private func percentile(_ sortedData: [Double], _ q: Double) -> Double {
        let n = sortedData.count
        if n == 0 { return 0.0 }
        if n == 1 { return sortedData[0] }
        let p = q / 100.0
        let idx = p * Double(n - 1)
        let i = Int(floor(idx))
        let j = Int(ceil(idx))
        let fraction = idx - Double(i)
        return sortedData[i] + fraction * (sortedData[j] - sortedData[i])
    }
}
