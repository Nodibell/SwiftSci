import Foundation
import SwiftDataFrame

/// Applies separate preprocessing transformers to specified subsets of columns.
public final class ColumnTransformer: PreprocessingTransformer, @unchecked Sendable {
    public struct Route {
        public let name: String
        public let transformer: any PreprocessingTransformer
        public let columnIndices: [Int]

        public init(name: String, transformer: any PreprocessingTransformer, columnIndices: [Int]) {
            self.name = name
            self.transformer = transformer
            self.columnIndices = columnIndices
        }
    }

    public let routes: [Route]
    private var isFitted: Bool = false

    public init(routes: [Route]) {
        self.routes = routes
    }

    public func fit(_ data: [[Double]]) throws {
        guard !data.isEmpty else { throw PreprocessingError.invalidInput("Data cannot be empty") }
        let numCols = data[0].count

        for route in routes {
            for colIdx in route.columnIndices {
                guard colIdx >= 0 && colIdx < numCols else {
                    throw PreprocessingError.invalidInput("Column index \(colIdx) out of bounds (0..<\(numCols))")
                }
            }
            let slicedData = sliceColumns(data, indices: route.columnIndices)
            try route.transformer.fit(slicedData)
        }
        isFitted = true
    }

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
