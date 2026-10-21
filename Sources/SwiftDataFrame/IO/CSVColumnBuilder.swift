internal struct CSVColumnBuilder<T: SupportedType> {
    private var values: [T?]
    private var chunks = [[T?]]()
    private let chunkCapacity: Int?
    private var count = 0
    private var nullCount = 0

    init(capacity: Int, chunked: Bool = false) {
        values = []
        values.reserveCapacity(capacity)
        chunkCapacity = chunked ? max(1, capacity) : nil
    }

    mutating func append(_ value: T?) {
        if let chunkCapacity, values.count == chunkCapacity {
            chunks.append(values)
            values = []
            values.reserveCapacity(chunkCapacity)
        }
        values.append(value)
        count += 1
        if case nil = value { nullCount += 1 }
    }

    /// Finalize once, releasing temporary chunks before another column is assembled.
    mutating func column(named name: String) -> TypedColumn<T> {
        defer { self = CSVColumnBuilder(capacity: 0) }
        if chunks.isEmpty { return TypedColumn(name: name, values: values, nullCount: nullCount) }
        var output = [T?]()
        output.reserveCapacity(count)
        for chunk in chunks { output.append(contentsOf: chunk) }
        output.append(contentsOf: values)
        return TypedColumn(name: name, values: output, nullCount: nullCount)
    }
}
