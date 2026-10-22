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
    /// Sampled types are validated during conversion. Later incompatible values
    /// can promote integers to Double or cause a column to retain its text.
    /// Explicit type overrides keep their conversion-failure-as-null behavior.
    public var inferTypes: Bool = true
    /// How eager and streamed reads schedule column conversion.
    /// The default converts columns in parallel using the system scheduler.
    /// Serial conversion can reduce CPU work at the cost of longer elapsed time.
    /// This choice does not change type inference, row batches, or output values.
    public var columnConversionMode: CSVColumnConversionMode = .parallel
    /// Maximum number of rows to read (nil = unlimited).
    public var maxRows: Int? = nil
    /// Type overrides for specific columns by name.
    public var columnTypeOverrides: [String: ColumnDType] = [:]

    /// Foundation date format patterns for columns explicitly overridden to `.date32`.
    /// Parsing uses the Gregorian calendar, `en_US_POSIX` locale, and UTC.
    /// For example, set `columnTypeOverrides["Date"] = .date32` and
    /// `columnDateFormats["Date"] = "d/M/yyyy"` for day-first source dates.
    /// Invalid dates become null, as with other explicit type conversions.
    /// Entries for inferred or non-date columns have no effect. Without a format,
    /// dates use `Date.parse(from:)`, including its month-first ambiguity rule.
    public var columnDateFormats: [String: String] = [:]

    /// Type overrides for specific columns by name (alias for columnTypeOverrides).
    public var columnTypes: [String: ColumnDType] {
        get { columnTypeOverrides }
        set { columnTypeOverrides = newValue }
    }

    /// Creates a new instance.
    public init() {}
    /// A public declaration.
    public static let `default` = CSVReadOptions()
}

/// Scheduling choices for converting CSV fields into native columns.
/// Both modes complete conversion before releasing the batch's borrowed bytes.
public enum CSVColumnConversionMode: Sendable {
    /// Convert independent columns with the system's native parallelism.
    case parallel
    /// Convert columns sequentially on the calling thread, without dispatching workers.
    /// This does not constrain other work in the process or set a thread affinity.
    case serial

    internal func forEachColumn(count: Int, _ body: @Sendable (Int) -> Void) {
        switch self {
        case .parallel:
            DispatchQueue.concurrentPerform(iterations: count, execute: body)
        case .serial:
            for column in 0..<count { body(column) }
        }
    }
}
