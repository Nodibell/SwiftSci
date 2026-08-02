/// Describes the structure of a DataFrame: column names and their types.
public struct Schema: Sendable, CustomStringConvertible {

    /// Field definition specifying column name, data type, and nullability.
    public struct Field: Sendable {
        /// Name identifier for the column field.
        public let name: String
        /// Data type classification for column elements.
        public let dtype: ColumnDType
        /// Whether the column permits nil values.
        public let nullable: Bool

        /// Initializes a schema field description.
        /// - Parameters:
        ///   - name: Column name string.
        ///   - dtype: Data type classification.
        ///   - nullable: Nullability flag. Defaults to true.
        public init(name: String, dtype: ColumnDType, nullable: Bool = true) {
            self.name     = name
            self.dtype    = dtype
            self.nullable = nullable
        }
    }

    /// The fields.
    public let fields: [Field]

    /// Creates a new instance.
    /// - Parameters:
    ///   - fields: The fields.
    public init(fields: [Field]) {
        self.fields = fields
    }

    /// Ordered list of column names.
    public var columnNames: [String] { fields.map(\.name) }

    /// Map from column name to ColumnDType.
    public var dtypes: [String: ColumnDType] {
        Dictionary(uniqueKeysWithValues: fields.map { ($0.name, $0.dtype) })
    }

    /// Accesses the element at the given index.
    /// - Parameters:
    ///   - name: The name.
    public subscript(name: String) -> Field? {
        fields.first { $0.name == name }
    }

    /// The description.
    public var description: String {
        let rows = fields.map { f in
            "  \(f.name): \(f.dtype)\(f.nullable ? "?" : "")"
        }.joined(separator: "\n")
        return "Schema(\n\(rows)\n)"
    }
}
