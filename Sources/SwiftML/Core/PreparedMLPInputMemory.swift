import SwiftDataFrame
import SwiftPreprocessing

func preparedMLPBytes(_ factors: Int...) throws -> Int {
    var bytes = 1
    for factor in factors {
        guard factor >= 0 else { throw MemoryAdmissionError.invalidCapacity }
        let (product, overflow) = bytes.multipliedReportingOverflow(by: factor)
        guard !overflow else { throw MemoryAdmissionError.invalidCapacity }
        bytes = product
    }
    return bytes
}

func preparedMLPInputBytes(_ input: IndexedNumericBatch, preprocessing: StandardPreprocessingPlan,
                          mappingCapacity: Int) throws -> Int {
    let source = input.source
    var capacities = [try preparedMLPBytes(source.columns.capacity, MemoryLayout<CompactNumericColumn>.stride),
        try preparedMLPBytes(source.provenanceStorageCapacity, MemoryLayout<Int>.stride),
        try preparedMLPBytes(input.sourcePositions.capacity, MemoryLayout<Int>.stride),
        try preparedMLPBytes(input.editedColumns.capacity, MemoryLayout<IndexedEditedColumn>.stride),
        try preparedMLPBytes(mappingCapacity, MemoryLayout<Int>.stride), 65_536]
    for column in source.columns {
        capacities += [try preparedMLPBytes(column.values.capacity, MemoryLayout<Double>.stride),
            try preparedMLPBytes(column.validity?.capacity ?? 0, MemoryLayout<UInt64>.stride), 128]
    }
    for column in input.editedColumns {
        capacities += [try preparedMLPBytes(column.data.values.capacity, MemoryLayout<Double>.stride),
            try preparedMLPBytes(column.data.validity?.capacity ?? 0, MemoryLayout<UInt64>.stride), 128]
    }
    capacities.append(try preparedMLPBytes(source.columnNames.capacity, MemoryLayout<String>.stride))
    for name in source.columnNames { capacities.append(try preparedMLPBytes(name.utf8.count, 2)) }
    capacities.append(try preparedMLPPlanBytes(preprocessing))
    return try MemoryEstimate(capacities: capacities).bytes
}

func preparedMLPPlanBytes(_ preprocessing: StandardPreprocessingPlan) throws -> Int {
    var capacities = [try preparedMLPBytes(preprocessing.columnNames.capacity, MemoryLayout<String>.stride)]
    for name in preprocessing.columnNames { capacities.append(try preparedMLPBytes(name.utf8.count, 2)) }
    for values in [preprocessing.replacements, preprocessing.negativeMeans, preprocessing.deviations,
                   preprocessing.imputer.statistics ?? [], preprocessing.scaler.mean ?? [], preprocessing.scaler.std ?? []] {
        capacities.append(try preparedMLPBytes(values.capacity, MemoryLayout<Double>.stride))
    }
    return try MemoryEstimate(capacities: capacities).bytes
}

func preparedMLPModelBytes(_ layers: [LayerWeights]) throws -> Int {
    var capacities = [128, try preparedMLPBytes(layers.capacity, MemoryLayout<LayerWeights>.stride)]
    for layer in layers {
        capacities += [try preparedMLPBytes(layer.W.capacity, MemoryLayout<Double>.stride),
            try preparedMLPBytes(layer.b.capacity, MemoryLayout<Double>.stride), 128]
    }
    return try MemoryEstimate(capacities: capacities).bytes
}

func preparedMLPPredictionEstimate(_ input: PreparedMLPInput, layers: [LayerWeights],
                                  modelBytes: Int) throws -> MemoryEstimate {
    guard let first = layers.first else { throw SwiftMLError.modelNotFitted }
    guard input.batch.columnCount == first.inDim else {
        throw SwiftMLError.dimensionMismatch(expected: first.inDim, got: input.batch.columnCount)
    }
    let width = layers.reduce(0) { max($0, max($1.inDim, $1.outDim)) }
    let batched = input.rowCount >= mlpPredictionBatchRows && width > 0 && width <= 512
    let workspace = try input.preprocessing.workspaceAllowance(rowCount: input.rowCount)
    return try MemoryEstimate(capacities: [input.retainedBytes, modelBytes, workspace.bytes, 65_536,
        preparedMLPBytes(width, MemoryLayout<Double>.stride, batched ? 2 * mlpPredictionBatchRows : 4),
        preparedMLPBytes(input.rowCount, MemoryLayout<Double>.stride, input.logicalPositions == nil ? 2 : 4)])
}
