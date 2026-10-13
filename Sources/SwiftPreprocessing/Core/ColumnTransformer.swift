import Foundation
import SwiftDataFrame

/// Applies separate preprocessing transformers to specified subsets of columns.
public final class ColumnTransformer: PreprocessingTransformer, @unchecked Sendable {
    /// Transformation route mapping a specific transformer to target column indices.
    public struct Route {
        /// Name identifier for this transformation route.
        public let name: String
        /// Target preprocessor implementation.
        public var transformer: any PreprocessingTransformer
        /// Array of targeted column indices.
        public let columnIndices: [Int]

        /// Initializes a column transformer route mapping.
        /// - Parameters:
        ///   - name: Route identifier name.
        ///   - transformer: Preprocessor implementation.
        ///   - columnIndices: Column indices array.
        public init(name: String, transformer: any PreprocessingTransformer, columnIndices: [Int]) {
            self.name = name
            self.transformer = transformer
            self.columnIndices = columnIndices
        }
    }

    /// Strategy for unrouted columns not explicitly handled by any route.
    public enum RemainderStrategy: Sendable {
        /// Discard unrouted columns (default Scikit-Learn parity).
        case drop
        /// Keep unrouted columns unchanged, appending them to the end of transformed rows.
        case passthrough
    }

    /// The routes.
    public var routes: [Route]
    /// How unrouted columns are handled.
    public let remainder: RemainderStrategy
    private var isFitted: Bool = false
    private var fittedColumnCount: Int = 0

    /// Creates a new instance.
    /// - Parameters:
    ///   - routes: Transformation routes mapping transformers to column indices.
    ///   - remainder: Policy for unrouted columns (default: `.drop`).
    public init(routes: [Route], remainder: RemainderStrategy = .drop) {
        self.routes = routes
        self.remainder = remainder
    }

    /// Fits all route transformers on their respective column slices.
    /// - Throws: `PreprocessingError.emptyInput` or `PreprocessingError.dimensionMismatch` on ragged rows.
    /// - Parameters:
    ///   - data: Raw input data matrix `[rows, cols]`.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty, !data[0].isEmpty else {
            throw PreprocessingError.emptyInput
        }
        let numCols = data[0].count
        for row in data where row.count != numCols {
            throw PreprocessingError.dimensionMismatch(expected: numCols, got: row.count)
        }

        for i in 0..<routes.count {
            for colIdx in routes[i].columnIndices {
                guard colIdx >= 0 && colIdx < numCols else {
                    throw PreprocessingError.invalidInput("Column index \(colIdx) out of bounds (0..<\(numCols))")
                }
            }
            let slicedData = sliceColumns(data, indices: routes[i].columnIndices)
            try routes[i].transformer.fit(slicedData)
        }
        self.fittedColumnCount = numCols
        self.isFitted = true
    }

    /// Transforms data across all configured routes.
    /// - Throws: `PreprocessingError.fittingRequired` or `PreprocessingError.dimensionMismatch`.
    /// - Returns: Horizontally concatenated matrix of transformed features.
    /// - Parameters:
    ///   - data: Raw input data matrix `[rows, cols]`.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard isFitted else { throw PreprocessingError.fittingRequired }
        guard !data.isEmpty, !data[0].isEmpty else { throw PreprocessingError.emptyInput }

        let numCols = data[0].count
        guard numCols == fittedColumnCount else {
            throw PreprocessingError.dimensionMismatch(expected: fittedColumnCount, got: numCols)
        }
        for row in data where row.count != numCols {
            throw PreprocessingError.dimensionMismatch(expected: numCols, got: row.count)
        }

        var transformedRoutes = [[[Double]]]()
        for route in routes {
            let slicedData = sliceColumns(data, indices: route.columnIndices)
            let transformed = try route.transformer.transform(slicedData)
            transformedRoutes.append(transformed)
        }

        let numRows = data.count
        var result = [[Double]](repeating: [], count: numRows)
        for r in 0..<numRows {
            var combinedRow = [Double]()
            for routeData in transformedRoutes {
                combinedRow.append(contentsOf: routeData[r])
            }
            result[r] = combinedRow
        }

        if remainder == .passthrough {
            let routedIndices = Set(routes.flatMap { $0.columnIndices })
            let unroutedIndices = (0..<numCols).filter { !routedIndices.contains($0) }
            if !unroutedIndices.isEmpty {
                for r in 0..<numRows {
                    for c in unroutedIndices {
                        result[r].append(data[r][c])
                    }
                }
            }
        }

        return result
    }

    private func sliceColumns(_ data: [[Double]], indices: [Int]) -> [[Double]] {
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}
