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
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    public func pivot(
        index indexCol: String,
        columns pivotCol: String,
        values valueCol: String,
        aggFunc: Aggregation = .mean
    ) throws -> DataFrame {
        guard let idxCol = self[column: indexCol],
              let pCol = self[column: pivotCol],
              let vCol = self[column: valueCol] else {
            throw SwiftMLError.columnNotFound("\(indexCol), \(pivotCol), or \(valueCol)")
        }

        let numRows = shape.rows
        var uniqueIndices = [Any?]()
        var indexMap = [AnyHashable?: Int]()
        var uniqueColumns = [String]()
        var columnMap = [AnyHashable?: Int]()

        for r in 0..<numRows {
            let rawIdx = idxCol.value(at: r)
            let iKey: AnyHashable? = rawIdx.flatMap { ($0 as? AnyHashable) ?? AnyHashable(String(describing: $0)) }

            if indexMap[iKey] == nil {
                indexMap[iKey] = uniqueIndices.count
                uniqueIndices.append(rawIdx)
            }

            let rawPCol = pCol.value(at: r)
            let pKey: AnyHashable? = rawPCol.flatMap { ($0 as? AnyHashable) ?? AnyHashable(String(describing: $0)) }

            if columnMap[pKey] == nil {
                let colName: String
                if let raw = rawPCol {
                    colName = String(describing: raw)
                } else {
                    colName = "null"
                }
                columnMap[pKey] = uniqueColumns.count
                uniqueColumns.append(colName)
            }
        }

        if uniqueColumns.contains(indexCol) {
            throw SwiftMLError.invalidParameter("Pivoted column value creates a column name identical to the index column '\(indexCol)'")
        }

        // Group values per (rowIdx, colIdx)
        var cellValues = Array(repeating: Array(repeating: [Double](), count: uniqueColumns.count), count: uniqueIndices.count)

        for r in 0..<numRows {
            let rawIdx = idxCol.value(at: r)
            let iKey: AnyHashable? = rawIdx.flatMap { ($0 as? AnyHashable) ?? AnyHashable(String(describing: $0)) }
            let rawPCol = pCol.value(at: r)
            let pKey: AnyHashable? = rawPCol.flatMap { ($0 as? AnyHashable) ?? AnyHashable(String(describing: $0)) }

            if let rowIdx = indexMap[iKey], let colIdx = columnMap[pKey],
               let val = vCol.value(at: r).flatMap({ toDouble($0) }) {
                cellValues[rowIdx][colIdx].append(val)
            }
        }

        var resultCols = [any AnyColumn]()
        resultCols.append(makeColumn(name: indexCol, dtype: idxCol.dtype, rawValues: uniqueIndices))

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
    /// - Throws: `SwiftMLError` or `DataFrameError` if column lengths mismatch, names collide, or I/O fails.
    public func melt(
        idVars: [String],
        valueVars: [String] = [],
        varName: String = "variable",
        valueName: String = "value"
    ) throws -> DataFrame {
        for id in idVars {
            if _columns[id] == nil { throw SwiftMLError.columnNotFound(id) }
        }

        // Validate duplicates and column name collisions
        guard Set(idVars).count == idVars.count else {
            throw SwiftMLError.invalidParameter("Melt 'idVars' contains duplicate column names")
        }
        if !valueVars.isEmpty {
            guard Set(valueVars).count == valueVars.count else {
                throw SwiftMLError.invalidParameter("Melt 'valueVars' contains duplicate column names")
            }
        }
        guard !idVars.contains(varName) else {
            throw SwiftMLError.invalidParameter("Melt 'varName' ('\(varName)') collides with idVars")
        }
        guard !idVars.contains(valueName) else {
            throw SwiftMLError.invalidParameter("Melt 'valueName' ('\(valueName)') collides with idVars")
        }
        guard varName != valueName else {
            throw SwiftMLError.invalidParameter("Melt 'varName' and 'valueName' cannot be identical ('\(varName)')")
        }

        let targetValueVars: [String]
        if valueVars.isEmpty {
            targetValueVars = columnNames.filter { !idVars.contains($0) }
        } else {
            targetValueVars = valueVars
            for v in targetValueVars {
                if _columns[v] == nil { throw SwiftMLError.columnNotFound(v) }
            }
        }

        let common = Set(idVars).intersection(Set(targetValueVars))
        guard common.isEmpty else {
            throw SwiftMLError.invalidParameter("Column(s) '\(common.joined(separator: ", "))' cannot be present in both idVars and valueVars")
        }

        let numRows = shape.rows
        // Integer multiplication overflow protection
        let (totalMeltRows, overflow) = numRows.multipliedReportingOverflow(by: targetValueVars.count)
        guard !overflow else {
            throw SwiftMLError.invalidParameter("Melt row count overflows Int")
        }
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

        // Target columns common dtype resolution
        let targetColumns = targetValueVars.compactMap { _columns[$0] }
        let commonDType = resolveCommonDType(targetColumns.map(\.dtype))

        var valueValues = [Any?]()
        valueValues.reserveCapacity(totalMeltRows)
        for col in targetColumns {
            for r in 0..<numRows {
                let raw = col.value(at: r)
                valueValues.append(castRawValue(raw, from: col.dtype, to: commonDType))
            }
        }

        resultCols.append(makeColumn(name: valueName, dtype: commonDType, rawValues: valueValues))

        return try DataFrame(columns: resultCols)
    }
}

private func resolveCommonDType(_ dtypes: [ColumnDType]) -> ColumnDType {
    guard let first = dtypes.first else { return .utf8 }
    if dtypes.allSatisfy({ $0 == first }) { return first }

    let allInteger = dtypes.allSatisfy { $0 == .int64 || $0 == .int32 }
    if allInteger { return .int64 }

    let allNumeric = dtypes.allSatisfy { $0 == .int64 || $0 == .int32 || $0 == .float64 || $0 == .float32 }
    if allNumeric { return .float64 }

    return .utf8
}

private func castRawValue(_ val: Any?, from source: ColumnDType, to target: ColumnDType) -> Any? {
    guard let val = val else { return nil }
    switch target {
    case source:
        return val
    case .float64:
        return toDouble(val)
    case .int64:
        if let i = val as? Int64 { return i }
        if let i = val as? Int32 { return Int64(i) }
        if let i = val as? Int { return Int64(i) }
        return nil
    case .utf8:
        return String(describing: val)
    default:
        return val
    }
}

private func toDouble(_ v: Any) -> Double? {
    switch v {
    case let x as Double: return x
    case let x as Float:  return Double(x)
    case let x as Int64:  return Double(x)
    case let x as Int32:  return Double(x)
    case let x as Int:    return Double(x)
    case let x as Bool:   return x ? 1.0 : 0.0
    default: return nil
    }
}
