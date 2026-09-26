internal struct CSVColumnBuilder<T: SupportedType> {
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

    func column(named name: String) -> TypedColumn<T> {
        TypedColumn(name: name, values: values, nullCount: nullCount)
    }
}
