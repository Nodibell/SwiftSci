internal protocol CSVColumnBuilding<Value> {
    associatedtype Value: SupportedType
    init(capacity: Int)
    mutating func append(_ value: Value?)
    mutating func column(named name: String) -> TypedColumn<Value>
}

internal struct CSVColumnBuilder<T: SupportedType>: CSVColumnBuilding {
    typealias Value = T
    private var values: [T?]
    private var nullCount = 0

    init(capacity: Int) {
        values = []
        values.reserveCapacity(capacity)
    }

    mutating func append(_ value: T?) {
        values.append(value)
        if case nil = value { nullCount += 1 }
    }

    mutating func column(named name: String) -> TypedColumn<T> {
        defer { self = CSVColumnBuilder(capacity: 0) }
        return TypedColumn(name: name, values: values, nullCount: nullCount)
    }
}

/// Experimental direct construction using the numerical pipeline's existing buffers.
internal struct CSVCompactDoubleBuilder: CSVColumnBuilding {
    private var values = [Double]()
    private var validity: [UInt64]?
    private var nullCount = 0
    private var pendingValidity: UInt64 = .max
    private let expectedRows: Int

    init(capacity: Int) {
        expectedRows = capacity
        values.reserveCapacity(capacity)
    }

    mutating func append(_ value: Double?) {
        let row = values.count
        values.append(value ?? .nan)
        if value == nil {
            pendingValidity &= ~(UInt64(1) << (row & 63))
            nullCount += 1
        }
        if row & 63 == 63 { finishValidityWord(at: row >> 6) }
    }

    private mutating func finishValidityWord(at index: Int) {
        if validity == nil {
            guard pendingValidity != .max else { return }
            validity = [UInt64](repeating: .max, count: (max(expectedRows, values.count) + 63) / 64)
        }
        if index == validity!.count {
            validity!.append(pendingValidity)
        } else {
            validity![index] = pendingValidity
        }
        pendingValidity = .max
    }

    mutating func column(named name: String) -> TypedColumn<Double> {
        defer { self = CSVCompactDoubleBuilder(capacity: 0) }
        if values.count & 63 != 0 { finishValidityWord(at: values.count >> 6) }
        var output = CompactNumericColumn(values)
        output.validity = validity
        output.nullCount = nullCount
        return TypedColumn(name: name, compactColumn: output)
    }
}
