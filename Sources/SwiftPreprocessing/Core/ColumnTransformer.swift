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

    /// The routes.
    public var routes: [Route]
    private var isFitted: Bool = false

    /// Creates a new instance.
    /// - Parameters:
    ///   - routes: The routes.
    public init(routes: [Route]) {
        self.routes = routes
    }

    /// Fit.
    /// - Throws: An error if the operation fails.
    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty else { throw PreprocessingError.invalidInput("Data cannot be empty") }
        let numCols = data[0].count

        for i in 0..<routes.count {
            for colIdx in routes[i].columnIndices {
                guard colIdx >= 0 && colIdx < numCols else {
                    throw PreprocessingError.invalidInput("Column index \(colIdx) out of bounds (0..<\(numCols))")
                }
            }
            let slicedData = sliceColumns(data, indices: routes[i].columnIndices)
            try routes[i].transformer.fit(slicedData)
        }
        isFitted = true
    }

    /// Transform.
    /// - Throws: An error if the operation fails.
    /// - Returns: A `[[Double]]` result.
    public func transform(_ data: [[Double]]) throws -> [[Double]] {
        guard isFitted else { throw PreprocessingError.notFitted }
        guard !data.isEmpty else { throw PreprocessingError.invalidInput("Data cannot be empty") }

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
        return result
    }

    private func sliceColumns(_ data: [[Double]], indices: [Int]) -> [[Double]] {
        return data.map { row in
            indices.map { row[$0] }
        }
    }
}
