import Foundation

/// A unified, consolidated error type for all modules in the SwiftSci ecosystem.
public enum SwiftMLError: Error, LocalizedError, Sendable, Equatable, CustomStringConvertible {
    
    // MARK: - Validation & Input Errors (Global & Preprocessing)
    case invalidInput(String)
    case emptyInput
    case dimensionMismatch(expected: Int, got: Int)
    case unknownCategory(String)
    case constantColumn(index: Int)
    
    // MARK: - Training & Fitting Errors
    case modelNotFitted
    case trainingFailed(String)
    case convergenceFailed(iterations: Int)
    
    // MARK: - Parameter Errors
    case invalidParameter(String)
    case unsupportedOperation(String)
    
    // MARK: - DataFrame Schema & I/O Errors
    case emptySchema
    case columnLengthMismatch(expected: Int, got: Int, column: String)
    case columnNotFound(String)
    case duplicateColumnName(String)
    case typeMismatch(column: String, expected: String, got: String)
    case fileNotFound(URL)
    case parseError(line: Int, description: String)
    case unsupportedFormat(String)
    case writeError(String)
    case indexOutOfRange(index: Int, count: Int)
    case invalidSampleSize(requested: Int, available: Int)
    case emptyDataFrame(operation: String)
    case castFailed(column: String, targetType: String)
    case partialCastFailure(column: String, targetType: String, failed: Int, total: Int)
    
    // MARK: - Statistics & Numerical Errors
    case insufficientData(minimum: Int, got: Int)
    case sizeMismatch(sizeA: Int, sizeB: Int)
    case containsNaN
    case containsInfinity
    case tooManySamples(limit: Int, got: Int)
    case invalidPercentile(Double)
    case invalidDDOF(Int)
    case invalidGroupCount(minimum: Int, got: Int)
    case divisionByZero(context: String)
    case negativeVariance
    
    // MARK: - Forecasting Errors
    case emptyTimeSeries
    case insufficientLength(minimum: Int, got: Int)
    case invalidAlpha(Double)
    case invalidBeta(Double)
    case invalidGamma(Double)
    case invalidPeriod(Int)
    case invalidSeasonalPeriod(Int)
    case invalidAROrder(Int)
    case invalidDifferencing(Int)
    case invalidMAOrder(Int)
    case invalidHorizon(Int)
    case singularMatrix
    case matrixDimensionMismatch(expectedRows: Int, expectedCols: Int, gotRows: Int, gotCols: Int)
    
    // MARK: - NLP Errors
    case invalidVocabulary
    
    // MARK: - Cluster & MLX Errors
    case svdFailed(info: Int32)
    
    // MARK: - Backward Compatibility Aliases
    public static var fitNotCalled: SwiftMLError { .modelNotFitted }
    public static var fittingRequired: SwiftMLError { .modelNotFitted }
    public static var notFitted: SwiftMLError { .modelNotFitted }
    
    public var description: String {
        return errorDescription ?? "Unknown SwiftSci error."
    }
    
    public var errorDescription: String? {
        switch self {
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .emptyInput:
            return "Input dataset or array must not be empty."
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected) features, but got \(got)."
        case .unknownCategory(let cat):
            return "Unknown category encountered: '\(cat)'."
        case .constantColumn(let idx):
            return "Column at index \(idx) has zero variance (constant values)."
        case .modelNotFitted:
            return "Model must be fitted before calling predict or transform."
        case .trainingFailed(let reason):
            return "Model training failed: \(reason)"
        case .convergenceFailed(let iter):
            return "Numerical solver failed to converge after \(iter) iterations."
        case .invalidParameter(let msg):
            return "Invalid parameter: \(msg)"
        case .unsupportedOperation(let op):
            return "Unsupported operation: \(op)"
        case .emptySchema:
            return "Cannot create DataFrame from empty list of columns."
        case .columnLengthMismatch(let expected, let got, let col):
            return "Column '\(col)' has \(got) elements, expected \(expected)."
        case .columnNotFound(let name):
            return "Column '\(name)' not found."
        case .duplicateColumnName(let name):
            return "Duplicate column name '\(name)'."
        case .typeMismatch(let col, let expected, let got):
            return "Type mismatch in '\(col)': expected \(expected), got \(got)."
        case .fileNotFound(let url):
            return "File not found: \(url.path)."
        case .parseError(let line, let desc):
            return "Parse error at line \(line): \(desc)."
        case .unsupportedFormat(let fmt):
            return "Unsupported format: \(fmt)."
        case .writeError(let msg):
            return "Write error: \(msg)."
        case .indexOutOfRange(let index, let count):
            return "Index \(index) is out of range (count: \(count))."
        case .invalidSampleSize(let req, let avail):
            return "Requested sample size \(req) exceeds available rows \(avail)."
        case .emptyDataFrame(let op):
            return "Cannot perform '\(op)' on an empty DataFrame."
        case .castFailed(let col, let type):
            return "Cannot cast column '\(col)' to \(type)."
        case .partialCastFailure(let col, let type, let failed, let total):
            return "Partial cast failure in column '\(col)' to \(type): \(failed) of \(total) values could not be cast."
        case .insufficientData(let min, let got):
            return "Insufficient data: at least \(min) elements required, got \(got)."
        case .sizeMismatch(let a, let b):
            return "Array sizes must match: got \(a) and \(b)."
        case .containsNaN:
            return "Input contains NaN values."
        case .containsInfinity:
            return "Input contains infinite values."
        case .tooManySamples(let limit, let got):
            return "Input size \(got) exceeds maximum supported sample limit of \(limit)."
        case .invalidPercentile(let q):
            return "Percentile q=\(q) is out of range [0, 1]."
        case .invalidDDOF(let ddof):
            return "Delta degrees of freedom \(ddof) must be ≥ 0."
        case .invalidGroupCount(let min, let got):
            return "At least \(min) groups required, got \(got)."
        case .divisionByZero(let ctx):
            return "Division by zero in \(ctx)."
        case .negativeVariance:
            return "Computed variance is negative."
        case .emptyTimeSeries:
            return "Time series must not be empty."
        case .insufficientLength(let min, let got):
            return "Time series length is insufficient: minimum \(min) points required, got \(got)."
        case .invalidAlpha(let alpha):
            return "Alpha parameter (\(alpha)) must be in (0, 1)."
        case .invalidBeta(let beta):
            return "Beta parameter (\(beta)) must be in (0, 1)."
        case .invalidGamma(let gamma):
            return "Gamma parameter (\(gamma)) must be in (0, 1)."
        case .invalidPeriod(let period):
            return "Seasonality period (\(period)) must be >= 2."
        case .invalidSeasonalPeriod(let s):
            return "Seasonal period (\(s)) must be >= 1."
        case .invalidAROrder(let p):
            return "AR order p (\(p)) must be >= 0."
        case .invalidDifferencing(let d):
            return "Differencing parameter d (\(d)) must be >= 0."
        case .invalidMAOrder(let q):
            return "MA order q (\(q)) must be >= 0."
        case .invalidHorizon(let h):
            return "Forecast horizon (\(h)) must be >= 1."
        case .singularMatrix:
            return "Matrix is singular (cannot invert)."
        case .matrixDimensionMismatch(let expectedRows, let expectedCols, let gotRows, let gotCols):
            return "Matrix dimension mismatch: expected (\(expectedRows), \(expectedCols)), got (\(gotRows), \(gotCols))."
        case .invalidVocabulary:
            return "The vocabulary is empty or invalid."
        case .svdFailed(let info):
            return "Singular Value Decomposition (SVD) failed with LAPACK info code: \(info)."
        }
    }
}

// MARK: - Compatibility Typealiases
public typealias DataFrameError = SwiftMLError
public typealias MLError = SwiftMLError
public typealias PreprocessingError = SwiftMLError
public typealias ForecastError = SwiftMLError
public typealias StatsError = SwiftMLError
public typealias ClusterError = SwiftMLError
public typealias NLPError = SwiftMLError
