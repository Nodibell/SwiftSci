import SwiftDataFrame

// Both input paths share admission, exclusive slots, cancellation, and output validation.
enum CoreMLMatrixRequest: Sendable {
    case columns(PreparedNumericBatch, indices: [Int])
    case prepared(CoreMLPreparedMatrix)

    var rowCount: Int {
        switch self {
        case .columns(let input, _): input.rowCount
        case .prepared(let input): input.rowCount
        }
    }

    var unreservedInputBytes: Int {
        switch self {
        case .columns(let input, _): input.payloadByteCount
        case .prepared: 0 // Retained storage has its own lifetime reservation.
        }
    }

    func pack(using session: CoreMLPredictionSession, into destination: CoreMLMatrixStorage) throws {
        switch self {
        case .columns(let input, let indices): try session.packMatrix(input, indices: indices, into: destination.array)
        case .prepared(let input): input.copy(into: destination)
        }
    }

    func result(_ columns: [[Double]], names: [String]) -> PreparedNumericBatch {
        switch self {
        case .columns(let input, _): input.replacingNumericColumns(columns, columnNames: names)
        case .prepared(let input): input.result(columns, names: names)
        }
    }
}
