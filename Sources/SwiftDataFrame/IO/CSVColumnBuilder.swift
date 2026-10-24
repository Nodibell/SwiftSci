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
    private var output = CompactNumericColumn([])
    private let expectedRows: Int

    init(capacity: Int) {
        expectedRows = capacity
        output.values.reserveCapacity(capacity)
    }

    mutating func append(_ value: Double?) {
        let row = output.values.count
        output.values.append(value ?? .nan)
        if output.validity != nil, row >> 6 == output.validity!.count {
            output.validity!.append(.max)
        }
        if value == nil {
            if output.validity == nil {
                output.validity = [UInt64](repeating: .max, count: (max(expectedRows, row + 1) + 63) / 64)
            }
            output.validity![row >> 6] &= ~(UInt64(1) << (row & 63))
            output.nullCount += 1
        }
    }

    mutating func column(named name: String) -> TypedColumn<Double> {
        defer { self = CSVCompactDoubleBuilder(capacity: 0) }
        return TypedColumn(name: name, compactColumn: output)
    }
}
