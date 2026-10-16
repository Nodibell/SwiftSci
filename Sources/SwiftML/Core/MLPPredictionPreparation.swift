import SwiftDataFrame
import SwiftPreprocessing

extension MLPPredictionRequest {
    func preparationEstimate(preprocessing: StandardPreprocessingPlan) throws -> MemoryEstimate {
        try preprocessing.validate(source)
        let empty = try IndexedNumericBatch(source, selectingRows: [])
        let base = try preparedMLPInputBytes(empty, preprocessing: preprocessing, mappingCapacity: 0)
        let rowIndices = try preparedMLPBytes(selectedRows?.capacity ?? rowCount, MemoryLayout<Int>.stride)
        let mapping = try preparedMLPBytes(rowCount, MemoryLayout<Int>.stride, 2)
        let bitmapWords = rowCount / 64 + (rowCount % 64 == 0 ? 0 : 1)
        let editedColumn = try MemoryEstimate(capacities: [preparedMLPBytes(rowCount, MemoryLayout<Double>.stride),
            preparedMLPBytes(bitmapWords, MemoryLayout<UInt64>.stride), MLPPredictionOverhead.editedColumn]).bytes
        let edited = try preparedMLPBytes(edits.count, editedColumn)
        let metadata = try preparedMLPBytes(edits.count, MemoryLayout<IndexedEditedColumn>.stride, 2)
        var descriptors = [try preparedMLPBytes(edits.capacity, MemoryLayout<ColumnEdits>.stride)]
        for column in edits {
            descriptors.append(try preparedMLPBytes(column.updates.capacity, MemoryLayout<Update>.stride))
            for update in column.updates {
                descriptors += [try preparedMLPBytes(update.rows.capacity, MemoryLayout<Int>.stride),
                    try preparedMLPBytes(update.values.capacity, MemoryLayout<Double?>.stride)]
            }
        }
        let retained = try MemoryEstimate(capacities: [base, rowIndices, mapping, edited, metadata, MLPPredictionOverhead.input]).bytes
        return try MemoryEstimate(capacities: [retained, retained, MemoryEstimate(capacities: descriptors).bytes,
            preparedMLPBytes(rowCount, MemoryLayout<Int>.stride, 8),
            preprocessing.workspaceAllowance(rowCount: rowCount).bytes,
            preparedMLPBytes(min(32, rowCount), source.columnCount, MemoryLayout<Double>.stride), 65_536])
    }
}

func validateStandardizedMLPInput(_ input: IndexedNumericBatch,
                                  preprocessing: StandardPreprocessingPlan) throws {
    guard input.rowCount > 0 else { return }
    let count = min(32, input.rowCount), width = input.columnCount
    var writer = try preprocessing.makeRowWriter(input, maximumRows: count)
    var tile = [Double](repeating: 0, count: count * width)
    for start in stride(from: 0, to: input.rowCount, by: count) {
        let rows = min(count, input.rowCount - start)
        try tile.withUnsafeMutableBufferPointer {
            try writer.fillDouble(rows: start..<(start + rows), into: $0, stride: width)
        }
        guard tile.prefix(rows * width).allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidInput("Standardized prediction values must be finite")
        }
    }
}
