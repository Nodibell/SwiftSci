/// The data type of a DataFrame column.
public enum ColumnDType: Sendable, Hashable, CustomStringConvertible {
    case int32
    case int64
    case float32
    case float64
    case boolean
    case utf8
    case date32     // Days since Unix epoch (1970-01-01)

    public var description: String {
        switch self {
        case .int32:   return "Int32"
        case .int64:   return "Int64"
        case .float32: return "Float"
        case .float64: return "Double"
        case .boolean: return "Bool"
        case .utf8:    return "String"
        case .date32:  return "Date"
        }
    }

    /// Whether this dtype represents a numeric type.
    public var isNumeric: Bool {
        switch self {
        case .int32, .int64, .float32, .float64: return true
        default: return false
        }
    }

    /// Whether this dtype can be cast to Double for arithmetic.
    public var isArithmetic: Bool { isNumeric }
}
