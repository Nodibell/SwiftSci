import Foundation

/// Options for reading CSV files.
public struct CSVReadOptions: Sendable {
    /// Column separator character.
    public var delimiter: Character = ","
    /// Whether the first row contains column names.
    public var hasHeader: Bool = true
    /// Strings that should be interpreted as null.
    public var nullValues: Set<String> = ["", "NA", "na", "N/A", "n/a", "null", "NULL", "NaN", "nan", "None"]
    /// Whether to automatically infer column types.
    public var inferTypes: Bool = true
    /// Maximum number of rows to read (nil = unlimited).
    public var maxRows: Int? = nil

    public init() {}
    public static let `default` = CSVReadOptions()
}
