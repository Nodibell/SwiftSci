extension CompactNumericColumn {
    /// The caller validates every source index before gathering any columns.
    package func gathered(at indices: [Int]) -> CompactNumericColumn {
        if indices.count <= UInt64.bitWidth { return gatheredWord(at: indices) }
        var gatheredValidity: [UInt64]?
        var missingCount = 0
        let gatheredValues = values.withUnsafeBufferPointer { source in
            indices.withUnsafeBufferPointer { rows in
                [Double](unsafeUninitializedCapacity: rows.count) { destination, initializedCount in
                    let sourceValues = source.baseAddress!
                    let outputValues = destination.baseAddress!
                    if let validity {
                        validity.withUnsafeBufferPointer { sourceBits in
                            for position in rows.indices {
                                let row = rows[position]
                                outputValues.advanced(by: position).initialize(to: sourceValues[row])
                                if sourceBits[row >> 6] & (UInt64(1) << (row & 63)) == 0 {
                                    if gatheredValidity == nil {
                                        gatheredValidity = [UInt64](repeating: .max, count: (rows.count + 63) / 64)
                                    }
                                    gatheredValidity![position >> 6] &= ~(UInt64(1) << (position & 63))
                                    missingCount += 1
                                }
                            }
                        }
                    } else {
                        for position in rows.indices {
                            outputValues.advanced(by: position).initialize(to: sourceValues[rows[position]])
                        }
                    }
                    initializedCount = rows.count
                }
            }
        }
        var result = CompactNumericColumn(gatheredValues)
        result.validity = gatheredValidity
        result.nullCount = missingCount
        return result
    }

    private func gatheredWord(at indices: [Int]) -> CompactNumericColumn {
        var result = CompactNumericColumn(indices.map { values[$0] })
        guard let validity else { return result }
        var word = UInt64.max
        var missingCount = 0
        for (position, row) in indices.enumerated() {
            if validity[row >> 6] & (UInt64(1) << (row & 63)) == 0 {
                word &= ~(UInt64(1) << position)
                missingCount += 1
            }
        }
        if missingCount > 0 { result.validity = [word] }
        result.nullCount = missingCount
        return result
    }

}
