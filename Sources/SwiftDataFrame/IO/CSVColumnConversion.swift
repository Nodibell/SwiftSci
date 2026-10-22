import Foundation

/// Eligibility is fixed by the original sample; late failures only advance this list.
internal struct CSVColumnPlan {
    let candidates: [ColumnDType]
    private let validates: Bool
    private let nullValues: Set<String>
    private let dateFormat: String?

    init<Records: CSVRecordAccess>(
        buffer: UnsafeBufferPointer<UInt8>, records: Records, startRow: Int,
        rowCount: Int, column: Int, name: String, options: CSVReadOptions
    ) {
        nullValues = options.nullValues
        if let type = options.columnTypeOverrides[name] {
            candidates = [type]
            validates = false
            dateFormat = options.columnDateFormats[name]
            return
        }
        dateFormat = nil
        validates = options.inferTypes
        guard options.inferTypes else {
            candidates = [.utf8]
            return
        }
        var integer = true
        var double = true
        var boolean = true
        var date = true
        var nonNull = false
        for row in startRow..<(startRow + min(rowCount, 1_000)) {
            guard let field = records.field(row: row, column: column), field.length != 0 else { continue }
            let text = VectorizedByteParsers.parseString(buffer: buffer, offset: field)
            if nullValues.contains(text) { continue }
            nonNull = true
            if integer && Int64.parse(from: text) == nil { integer = false }
            if double && VectorizedByteParsers.parseDouble(buffer: buffer, offset: field) == nil
                && Double.parse(from: text) == nil { double = false }
            if boolean && Bool.parse(from: text) == nil { boolean = false }
            if date && Date.parse(from: text) == nil { date = false }
            if !integer && !double && !boolean && !date { break }
        }
        var types = [ColumnDType]()
        if nonNull {
            if integer { types.append(.int64) }
            if double { types.append(.float64) }
            if boolean { types.append(.boolean) }
            if date { types.append(.date32) }
        }
        types.append(.utf8)
        candidates = types
    }

    /// A rejected original token can disprove later candidates, but cannot admit new ones.
    func nextCandidate(after index: Int, buffer: UnsafeBufferPointer<UInt8>, rejectedField: CSVFieldOffset) -> Int {
        let record = CSVRecordIndex(fields: [PackedCSVField(startOffset: rejectedField.startOffset,
            length: rejectedField.length, escapedQuotesPresent: rejectedField.escapedQuotesPresent)], rowStarts: [0, 1])
        var next = index + 1
        while candidates[next] != .utf8 {
            let probe = conversion(type: candidates[next], capacity: 1)
            probe.appendIndexed(buffer: buffer, records: record, startRow: 0, rowCount: 1, column: 0)
            if !probe.rejected { return next }
            next += 1
        }
        return next
    }

    func conversion(type: ColumnDType, capacity: Int) -> CSVColumnConversion {
        let nulls = CSVNullMatcher(nullValues)
        switch type {
        case .int32, .int64:
            return .integer(CSVColumnAttempt(decoder: CSVIntegerDecoder(nulls: nulls),
                capacity: capacity, validates: validates))
        case .float32, .float64:
            return .double(CSVColumnAttempt(decoder: CSVDoubleDecoder(nulls: nulls),
                capacity: capacity, validates: validates))
        case .boolean:
            return .boolean(CSVColumnAttempt(decoder: CSVBooleanDecoder(nullValues: nullValues),
                capacity: capacity, validates: validates))
        case .date32:
            return .date(CSVColumnAttempt(decoder: CSVDateDecoder(nulls: nulls,
                parser: CSVDateParser(format: dateFormat)), capacity: capacity, validates: validates))
        case .utf8:
            if let decoder = CSVSmallStringDecoder(nullValues: nullValues) {
                return .stringWithSmallNullSet(CSVColumnAttempt(
                    decoder: decoder, capacity: capacity, validates: false))
            }
            return .string(CSVColumnAttempt(decoder: CSVStringDecoder(nullValues: nullValues),
                capacity: capacity, validates: false))
        }
    }
}

/// Dispatches once per column and batch. Each reference owns one growing typed array.
internal enum CSVColumnConversion {
    case integer(CSVColumnAttempt<CSVIntegerDecoder>)
    case double(CSVColumnAttempt<CSVDoubleDecoder>)
    case boolean(CSVColumnAttempt<CSVBooleanDecoder>)
    case date(CSVColumnAttempt<CSVDateDecoder>)
    case string(CSVColumnAttempt<CSVStringDecoder>)
    case stringWithSmallNullSet(CSVColumnAttempt<CSVSmallStringDecoder>)

    func append<Records: CSVRecordAccess>(
        buffer: UnsafeBufferPointer<UInt8>, records: Records, startRow: Int,
        rowCount: Int, column: Int, sourceOffset: Int = 0
    ) {
        switch self {
        case .integer(let attempt): attempt.append(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
        case .double(let attempt): attempt.appendDouble(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
        case .boolean(let attempt): attempt.append(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
        case .date(let attempt): attempt.append(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
        case .string(let attempt): attempt.append(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
        case .stringWithSmallNullSet(let attempt): attempt.append(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
        }
    }

    func appendIndexed(
        buffer: UnsafeBufferPointer<UInt8>, records: CSVRecordIndex,
        startRow: Int, rowCount: Int, column: Int, sourceOffset: Int = 0
    ) {
        if records.withFlatFields({ view in
            append(buffer: buffer, records: view, startRow: startRow, rowCount: rowCount, column: column, sourceOffset: sourceOffset)
            return true
        }) == true { return }
        if records.withCompactFields({ view in
            append(buffer: buffer, records: view, startRow: startRow, rowCount: rowCount, column: column, sourceOffset: sourceOffset)
            return true
        }) == true { return }
        append(buffer: buffer, records: records, startRow: startRow, rowCount: rowCount, column: column, sourceOffset: sourceOffset)
    }

    var rejectedField: CSVFieldOffset? {
        switch self {
        case .integer(let attempt): attempt.rejectedField
        case .double(let attempt): attempt.rejectedField
        case .boolean(let attempt): attempt.rejectedField
        case .date(let attempt): attempt.rejectedField
        case .string(let attempt): attempt.rejectedField
        case .stringWithSmallNullSet(let attempt): attempt.rejectedField
        }
    }

    var rejected: Bool { rejectedField != nil }

    func column(named name: String) -> any AnyColumn {
        switch self {
        case .integer(let attempt): attempt.column(named: name)
        case .double(let attempt): attempt.column(named: name)
        case .boolean(let attempt): attempt.column(named: name)
        case .date(let attempt): attempt.column(named: name)
        case .string(let attempt): attempt.column(named: name)
        case .stringWithSmallNullSet(let attempt): attempt.column(named: name)
        }
    }
}

internal enum CSVDecodedValue<Value> {
    case value(Value)
    case missing
    case invalid
}

internal protocol CSVValueDecoder {
    associatedtype Value: SupportedType
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<Value>
    /// Explicit schemas turn both missing and invalid fields into nil.
    func decodeValue(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Value?
}

internal extension CSVValueDecoder {
    func decodeValue(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Value? {
        switch decode(buffer: buffer, offset: offset) {
        case .value(let value): value
        case .missing, .invalid: nil
        }
    }
}

internal final class CSVColumnAttempt<Decoder: CSVValueDecoder> {
    private let decoder: Decoder
    private let validates: Bool
    private var values: CSVColumnBuilder<Decoder.Value>
    private(set) var rejectedField: CSVFieldOffset?
    var rejected: Bool { rejectedField != nil }

    init(decoder: Decoder, capacity: Int, validates: Bool) {
        self.decoder = decoder
        self.validates = validates
        values = CSVColumnBuilder(capacity: capacity)
    }

    func append<Records: CSVRecordAccess>(
        _ buffer: UnsafeBufferPointer<UInt8>, _ records: Records,
        _ startRow: Int, _ rowCount: Int, _ column: Int, sourceOffset: Int
    ) {
        guard !rejected else { return }
        var batchValues = CSVColumnBuilder<Decoder.Value>(capacity: 0)
        swap(&batchValues, &values)
        let field = Self.appendRows(buffer, records, startRow, rowCount, column,
            decoder: decoder, validates: validates, into: &batchValues)
        swap(&batchValues, &values)
        if let field { reject(field, sourceOffset: sourceOffset) }
    }

    // Keep Double visible in the validating loop so decoder calls can be direct.
    func appendDouble<Records: CSVRecordAccess>(
        _ buffer: UnsafeBufferPointer<UInt8>, _ records: Records,
        _ startRow: Int, _ rowCount: Int, _ column: Int, sourceOffset: Int
    ) where Decoder == CSVDoubleDecoder {
        guard validates else {
            append(buffer, records, startRow, rowCount, column, sourceOffset: sourceOffset)
            return
        }
        guard !rejected else { return }
        var batchValues = CSVColumnBuilder<Double>(capacity: 0)
        swap(&batchValues, &values)
        let field = appendValidatedDoubles(buffer, records, startRow, rowCount, column,
            decoder: decoder, into: &batchValues)
        swap(&batchValues, &values)
        if let field { reject(field, sourceOffset: sourceOffset) }
    }

    private func reject(_ field: CSVFieldOffset, sourceOffset: Int) {
        rejectedField = CSVFieldOffset(startOffset: sourceOffset + field.startOffset,
            length: field.length, escapedQuotesPresent: field.escapedQuotesPresent)
        values = CSVColumnBuilder(capacity: 0)
    }

    // Row loops mutate a local builder instead of a class property.
    private static func appendRows<Records: CSVRecordAccess>(
        _ buffer: UnsafeBufferPointer<UInt8>, _ records: Records,
        _ startRow: Int, _ rowCount: Int, _ column: Int,
        decoder: Decoder, validates: Bool, into values: inout CSVColumnBuilder<Decoder.Value>
    ) -> CSVFieldOffset? {
        // Choose the validation contract once per batch. Inference keeps the
        // classified result; an explicit schema only needs an optional value.
        if !validates {
            for row in startRow..<(startRow + rowCount) {
                let value = records.field(row: row, column: column).flatMap {
                    decoder.decodeValue(buffer: buffer, offset: $0)
                }
                values.append(value)
            }
            return nil
        }
        for row in startRow..<(startRow + rowCount) {
            guard let field = records.field(row: row, column: column) else {
                values.append(nil)
                continue
            }
            switch decoder.decode(buffer: buffer, offset: field) {
            case .value(let value): values.append(value)
            case .missing: values.append(nil)
            case .invalid: return field
            }
        }
        return nil
    }

    func column(named name: String) -> TypedColumn<Decoder.Value> {
        values.column(named: name)
    }
}

internal struct CSVIntegerDecoder: CSVValueDecoder {
    let nulls: CSVNullMatcher

    // Explicit schemas consume optional values directly. Inference retains
    // its classified decoder so missing fields do not need a second parse.
    @inline(__always)
    func decodeValue(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Int64? {
        if !nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return nil }
        if let value = VectorizedByteParsers.parseInt(buffer: buffer, offset: offset) { return Int64(value) }
        if nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return nil }
        return Int64.parse(from: VectorizedByteParsers.parseString(buffer: buffer, offset: offset))
    }

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<Int64> {
        if !nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return .missing }
        if let value = VectorizedByteParsers.parseInt(buffer: buffer, offset: offset) { return .value(Int64(value)) }
        if nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return .missing }
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        return Int64.parse(from: text).map(CSVDecodedValue.value) ?? .invalid
    }
}

internal struct CSVDoubleDecoder: CSVValueDecoder {
    let nulls: CSVNullMatcher

    @inline(__always)
    func decodeValue(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Double? {
        if !nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return nil }
        if let value = VectorizedByteParsers.parseDouble(buffer: buffer, offset: offset) { return value }
        if nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return nil }
        return Double.parse(from: VectorizedByteParsers.parseString(buffer: buffer, offset: offset))
    }

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<Double> {
        if !nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return .missing }
        if let value = VectorizedByteParsers.parseDouble(buffer: buffer, offset: offset) { return .value(value) }
        if nulls.canSkipNumericMatch && nulls.contains(buffer: buffer, offset: offset) { return .missing }
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        return Double.parse(from: text).map(CSVDecodedValue.value) ?? .invalid
    }
}

internal struct CSVBooleanDecoder: CSVValueDecoder {
    let nullValues: Set<String>

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<Bool> {
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        if nullValues.contains(text) { return .missing }
        return Bool.parse(from: text).map(CSVDecodedValue.value) ?? .invalid
    }
}

internal struct CSVStringDecoder: CSVValueDecoder {
    let nullValues: Set<String>

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<String> {
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        return nullValues.contains(text) ? .missing : .value(text)
    }
}

internal struct CSVSmallStringDecoder: CSVValueDecoder, Sendable {
    private let first: String
    private let second: String?

    init?(nullValues: Set<String>) {
        guard nullValues.count <= 2 else { return nil }
        var iterator = nullValues.makeIterator()
        guard let first = iterator.next() else { return nil }
        self.first = first
        second = iterator.next()
    }

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<String> {
        let text = VectorizedByteParsers.parseString(buffer: buffer, offset: offset)
        return first == text || second == text ? .missing : .value(text)
    }
}

internal struct CSVDateDecoder: CSVValueDecoder {
    let nulls: CSVNullMatcher
    let parser: CSVDateParser

    // Returning Foundation's optional directly avoids an additional generic
    // result tag on the explicit-schema path.
    @inline(__always)
    func decodeValue(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> Date? {
        if nulls.contains(buffer: buffer, offset: offset) { return nil }
        return parser.parse(buffer: buffer, offset: offset)
    }

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<Date> {
        if nulls.contains(buffer: buffer, offset: offset) { return .missing }
        return parser.parse(buffer: buffer, offset: offset).map(CSVDecodedValue.value) ?? .invalid
    }
}

private func appendValidatedDoubles<Records: CSVRecordAccess>(
    _ buffer: UnsafeBufferPointer<UInt8>, _ records: Records,
    _ startRow: Int, _ rowCount: Int, _ column: Int,
    decoder: CSVDoubleDecoder, into values: inout CSVColumnBuilder<Double>
) -> CSVFieldOffset? {
    for row in startRow..<(startRow + rowCount) {
        guard let field = records.field(row: row, column: column) else {
            values.append(nil)
            continue
        }
        switch decoder.decode(buffer: buffer, offset: field) {
        case .value(let value): values.append(value)
        case .missing: values.append(nil)
        case .invalid: return field
        }
    }
    return nil
}
