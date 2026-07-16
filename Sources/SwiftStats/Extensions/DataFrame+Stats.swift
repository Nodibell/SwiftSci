import SwiftDataFrame

/// Statistical extensions on `DataFrame` — delegates to `Stats`.
public extension DataFrame {

    /// Descriptive statistics for a named numeric column.
    func stats(for column: String) throws -> DescriptiveStats {
        guard let col = self[column: column] else {
            throw DataFrameError.columnNotFound(column)
        }
        guard col.dtype.isNumeric, let doubles = col.toDoubles() else {
            throw DataFrameError.typeMismatch(
                column: column, expected: "numeric", got: col.dtype.description
            )
        }
        return try Stats.describe(doubles, nullCount: col.nullCount)
    }

    /// Returns a summary DataFrame (count, mean, std, min, q1, median, q3, max)
    /// for all numeric columns. Mirrors pandas `.describe()`.
    func describe() throws -> DataFrame {
        let numericCols = columns.filter { $0.dtype.isNumeric }
        guard !numericCols.isEmpty else { return DataFrame.empty }

        let statNames = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        var resultCols: [any AnyColumn] = [
            TypedColumn<String>(name: "stat", values: statNames.map { Optional($0) })
        ]

        for col in numericCols {
            guard let doubles = col.toDoubles() else { continue }
            let d = try Stats.describe(doubles, nullCount: col.nullCount)
            let values: [Double?] = [
                Double(d.count), d.mean, d.standardDeviation,
                d.min, d.q1, d.median, d.q3, d.max
            ]
            resultCols.append(TypedColumn<Double>(name: col.name, values: values))
        }

        return try DataFrame(columns: resultCols)
    }

    /// Pearson correlation matrix for all numeric columns.
    func correlationMatrix() throws -> DataFrame {
        let numericCols = columns.filter { $0.dtype.isNumeric }
        guard numericCols.count >= 2 else {
            throw DataFrameError.emptyDataFrame(operation: "correlationMatrix")
        }

        let names   = numericCols.map(\.name)
        let vectors = try numericCols.map { col -> [Double] in
            guard let d = col.toDoubles() else {
                throw DataFrameError.typeMismatch(
                    column: col.name, expected: "numeric", got: col.dtype.description
                )
            }
            return d
        }

        let matrix = try Stats.correlationMatrix(vectors)

        // Build result: first column = row label, then one column per variable
        var resultCols: [any AnyColumn] = [
            TypedColumn<String>(name: "variable", values: names.map { Optional($0) })
        ]
        for (j, colName) in names.enumerated() {
            let colValues: [Double?] = matrix.map { $0[j] }
            resultCols.append(TypedColumn<Double>(name: colName, values: colValues))
        }

        return try DataFrame(columns: resultCols)
    }
}
