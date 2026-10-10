import CoreML
import Darwin
import SwiftDataFrame
import SwiftPreprocessing

/// Immutable model-typed input prepared by a matrix pool for repeated predictions.
/// Copies share storage and one reservation. No mutable buffer or pointer escapes.
/// The last reference frees storage before scheduling release of its budget reservation.
public final class CoreMLPreparedMatrix: @unchecked Sendable {
    /// Number of rows, including repeated selections.
    public let rowCount: Int
    /// Columns in model feature order.
    public let columnNames: [String]
    /// Logical bytes in the converted matrix, excluding padding and metadata.
    public let payloadByteCount: Int
    /// Lifetime allowance for page-rounded storage and metadata, not an RSS measurement.
    public let reservedBytes: Int
    let schema: ArrayFeature
    private var storage: CoreMLMatrixStorage?
    private var sourceRows: [Int]?
    private let reservation: MemoryReservation

    static func allowance(schema: ArrayFeature, names: [String]) throws -> MemoryEstimate {
        let namesBytes = try coreMLByteCount(names.capacity, MemoryLayout<String>.stride)
        let strings = try names.map { try coreMLByteCount($0.utf8.count, 2) }
        return try MemoryEstimate(capacities: [CoreMLMatrixStorage.capacity(schema),
            coreMLByteCount(schema.shape[0], MemoryLayout<Int>.stride, 2), namesBytes, 65_536] + strings)
    }

    init(input: PreparedNumericBatch, indices: [Int], session: CoreMLPredictionSession,
         reservation: MemoryReservation, allowance: MemoryEstimate) throws {
        let schema = session.inputArray!
        let storage = try CoreMLMatrixStorage(schema)
        try session.packMatrix(input, indices: indices, into: storage.array)
        try Task.checkCancellation()
        let rows = input.originalRowIndices
        let ownedRows = rows.withUnsafeBufferPointer { source in
            Array<Int>(unsafeUninitializedCapacity: source.count) { target, initialized in
                target.baseAddress!.initialize(from: source.baseAddress!, count: source.count)
                initialized = source.count
            }
        }
        guard ownedRows.capacity <= schema.shape[0] * 2 else {
            throw MemoryAdmissionError.invalidCapacity
        }
        self.schema = schema
        self.storage = storage
        sourceRows = ownedRows
        rowCount = input.rowCount
        columnNames = session.inputColumns
        payloadByteCount = try coreMLByteCount(rowCount, schema.width, schema.elementBytes)
        reservedBytes = allowance.bytes
        self.reservation = reservation
    }

    func validate(for session: CoreMLPredictionSession) throws {
        guard let target = session.inputArray, target.shape == schema.shape,
              target.dataType == schema.dataType, session.inputColumns == columnNames else {
            throw SwiftMLError.invalidParameter("Prepared matrix does not match the pool's shape, element type, or column order")
        }
    }

    // Both allocations are private, tightly packed row-major storage with identical schemas.
    // No caller or Core ML operation can mutate the retained source.
    func copy(into destination: CoreMLMatrixStorage) {
        withExtendedLifetime(self) {
            _ = memcpy(destination.array.dataPointer, storage!.array.dataPointer, payloadByteCount)
        }
    }

    func result(_ columns: [[Double]], names: [String]) -> PreparedNumericBatch {
        PreparedNumericBatch(columnNames: names, columns: columns.map(CompactNumericColumn.init),
            rowCount: rowCount, sourceRows: sourceRows)
    }

    deinit {
        storage = nil
        sourceRows = nil
        let reservation = reservation
        Task { await reservation.finish() }
    }
}
