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

    func append<Records: CSVRecordAccess>(
        buffer: UnsafeBufferPointer<UInt8>, records: Records, startRow: Int,
        rowCount: Int, column: Int
    ) {
        switch self {
        case .integer(let attempt): attempt.append(buffer, records, startRow, rowCount, column)
        case .double(let attempt): attempt.append(buffer, records, startRow, rowCount, column)
        case .boolean(let attempt): attempt.append(buffer, records, startRow, rowCount, column)
        case .date(let attempt): attempt.append(buffer, records, startRow, rowCount, column)
        case .string(let attempt): attempt.append(buffer, records, startRow, rowCount, column)
        }
    }

    var rejected: Bool {
        switch self {
        case .integer(let attempt): attempt.rejected
        case .double(let attempt): attempt.rejected
        case .boolean(let attempt): attempt.rejected
        case .date(let attempt): attempt.rejected
        case .string(let attempt): attempt.rejected
        }
    }

    func column(named name: String) -> any AnyColumn {
        switch self {
        case .integer(let attempt): attempt.column(named: name)
        case .double(let attempt): attempt.column(named: name)
        case .boolean(let attempt): attempt.column(named: name)
        case .date(let attempt): attempt.column(named: name)
        case .string(let attempt): attempt.column(named: name)
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
}

internal final class CSVColumnAttempt<Decoder: CSVValueDecoder> {
    private let decoder: Decoder
    private let validates: Bool
    private var values: CSVColumnBuilder<Decoder.Value>
    private(set) var rejected = false

    init(decoder: Decoder, capacity: Int, validates: Bool) {
        self.decoder = decoder
        self.validates = validates
        values = CSVColumnBuilder(capacity: capacity)
    }

    func append<Records: CSVRecordAccess>(
        _ buffer: UnsafeBufferPointer<UInt8>, _ records: Records,
        _ startRow: Int, _ rowCount: Int, _ column: Int
    ) {
        guard !rejected else { return }
        for row in startRow..<(startRow + rowCount) {
            guard let field = records.field(row: row, column: column) else {
                values.append(nil)
                continue
            }
            switch decoder.decode(buffer: buffer, offset: field) {
            case .value(let value): values.append(value)
            case .missing: values.append(nil)
            case .invalid:
                if validates {
                    rejected = true
                    values = CSVColumnBuilder(capacity: 0)
                    return
                }
                values.append(nil)
            }
        }
    }

    func column(named name: String) -> TypedColumn<Decoder.Value> {
        values.column(named: name)
    }
}

internal struct CSVIntegerDecoder: CSVValueDecoder {
    let nulls: CSVNullMatcher

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

internal struct CSVDateDecoder: CSVValueDecoder {
    let nulls: CSVNullMatcher
    let parser: CSVDateParser

    @inline(__always)
    func decode(buffer: UnsafeBufferPointer<UInt8>, offset: CSVFieldOffset) -> CSVDecodedValue<Date> {
        if nulls.contains(buffer: buffer, offset: offset) { return .missing }
        return parser.parse(buffer: buffer, offset: offset).map(CSVDecodedValue.value) ?? .invalid
    }
}
