import Foundation

extension DataFrame {
    /// Reshapes DataFrame from long format to wide format (pivot).
    ///
    /// - Parameters:
    ///   - indexCol: Column to use to make new frame's index.
    ///   - pivotCol: Column to use to make new frame's columns.
    ///   - valueCol: Column to use for populating new frame's values.
    ///   - aggFunc: Aggregation function to apply when duplicate index/column pairs exist.
    /// - Returns: Reshaped DataFrame.
    public func pivot(
        index indexCol: String,
        columns pivotCol: String,
        values valueCol: String,
        aggFunc: Aggregation = .mean
    ) throws -> DataFrame {
        guard let idxCol = self[column: indexCol],
              let pCol = self[column: pivotCol],
              let vCol = self[column: valueCol] else {
            throw DataFrameError.columnNotFound("\(indexCol), \(pivotCol), or \(valueCol)")
        }

        let numRows = shape.rows
        var uniqueIndices = [String]()
        var indexMap = [String: Int]()
        var uniqueColumns = [String]()
        var columnMap = [String: Int]()

        for r in 0..<numRows {
            let iVal = idxCol.value(at: r).map { "\($0)" } ?? "null"
            let pVal = pCol.value(at: r).map { "\($0)" } ?? "null"

            if indexMap[iVal] == nil {
                indexMap[iVal] = uniqueIndices.count
                uniqueIndices.append(iVal)
            }
            if columnMap[pVal] == nil {
                columnMap[pVal] = uniqueColumns.count
                uniqueColumns.append(pVal)
            }
        }

        // Group values per (rowIdx, colIdx)
        var cellValues = Array(repeating: Array(repeating: [Double](), count: uniqueColumns.count), count: uniqueIndices.count)

        for r in 0..<numRows {
            let iVal = idxCol.value(at: r).map { "\($0)" } ?? "null"
            let pVal = pCol.value(at: r).map { "\($0)" } ?? "null"

            if let rowIdx = indexMap[iVal], let colIdx = columnMap[pVal],
               let val = vCol.value(at: r).flatMap({ toDouble($0) }) {
                cellValues[rowIdx][colIdx].append(val)
            }
        }

        var resultCols = [any AnyColumn]()
        resultCols.append(TypedColumn<String>(name: indexCol, values: uniqueIndices.map { Optional($0) }))

        for (cIdx, colName) in uniqueColumns.enumerated() {
            var colVals = [Double?]()
            colVals.reserveCapacity(uniqueIndices.count)

            for rIdx in 0..<uniqueIndices.count {
                let vals = cellValues[rIdx][cIdx]
                if vals.isEmpty {
                    colVals.append(nil)
                } else {
                    switch aggFunc {
                    case .mean: colVals.append(vals.reduce(0, +) / Double(vals.count))
                    case .sum:  colVals.append(vals.reduce(0, +))
                    case .min:  colVals.append(vals.min())
                    case .max:  colVals.append(vals.max())
                    case .count: colVals.append(Double(vals.count))
                    case .first: colVals.append(vals.first)
                    case .last:  colVals.append(vals.last)
                    }
                }
            }
            resultCols.append(TypedColumn<Double>(name: colName, values: colVals))
        }

        return try DataFrame(columns: resultCols)
    }

    /// Unpivots a DataFrame from wide format to long format (melt).
    ///
    /// - Parameters:
    ///   - idVars: Column(s) to use as identifier variables.
    ///   - valueVars: Column(s) to unpivot. If empty, uses all columns not in `idVars`.
    ///   - varName: Name to use for the 'variable' column.
    ///   - valueName: Name to use for the 'value' column.
    /// - Returns: Unpivoted DataFrame.
    public func melt(
        idVars: [String],
        valueVars: [String] = [],
        varName: String = "variable",
        valueName: String = "value"
    ) throws -> DataFrame {
        for id in idVars {
            if _columns[id] == nil { throw DataFrameError.columnNotFound(id) }
        }

        let targetValueVars: [String]
        if valueVars.isEmpty {
            targetValueVars = columnNames.filter { !idVars.contains($0) }
        } else {
            targetValueVars = valueVars
            for v in targetValueVars {
                if _columns[v] == nil { throw DataFrameError.columnNotFound(v) }
            }
        }

        let numRows = shape.rows
        let totalMeltRows = numRows * targetValueVars.count
        guard totalMeltRows > 0 else { return DataFrame.empty }

        var resultCols = [any AnyColumn]()

        // Build ID columns
        for id in idVars {
            guard let col = _columns[id] else { continue }
            var idValues = [Any?]()
            idValues.reserveCapacity(totalMeltRows)
            for _ in targetValueVars {
                for r in 0..<numRows {
                    idValues.append(col.value(at: r))
                }
            }
            resultCols.append(makeColumn(name: id, dtype: col.dtype, rawValues: idValues))
        }

        // Build variable column
        var varValues = [String?]()
        varValues.reserveCapacity(totalMeltRows)
        for v in targetValueVars {
            for _ in 0..<numRows {
                varValues.append(v)
            }
        }
        resultCols.append(TypedColumn<String>(name: varName, values: varValues))

        // Build value column
        var valueValues = [Any?]()
        valueValues.reserveCapacity(totalMeltRows)
        for v in targetValueVars {
            guard let col = _columns[v] else { continue }
            for r in 0..<numRows {
                valueValues.append(col.value(at: r))
            }
        }

        let valueDType = _columns[targetValueVars[0]]?.dtype ?? .utf8
        resultCols.append(makeColumn(name: valueName, dtype: valueDType, rawValues: valueValues))

        return try DataFrame(columns: resultCols)
    }
}

private func toDouble(_ v: Any) -> Double? {
    switch v {
    case let x as Double: return x
    case let x as Float:  return Double(x)
    case let x as Int64:  return Double(x)
    case let x as Int32:  return Double(x)
    case let x as Int:    return Double(x)
    default: return nil
    }
}
