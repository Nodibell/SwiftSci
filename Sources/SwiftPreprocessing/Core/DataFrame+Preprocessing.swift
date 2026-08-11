import Foundation
import SwiftDataFrame

extension PreprocessingTransformer {
    /// Fits transformer parameters on the specified DataFrame columns.
    /// - Parameters:
    ///   - df: Target input `DataFrame`.
    ///   - columns: Array of column names to compute parameters from.
    /// - Throws: `PreprocessingError` if feature extraction or fitting fails.
    public mutating func fit(_ df: DataFrame, columns: [String]) throws {
        try fit(df.toFeatureMatrix(columns))
    }

    /// Transforms specified columns of a DataFrame using fitted transformer parameters.
    /// - Parameters:
    ///   - df: Target input `DataFrame`.
    ///   - columns: Array of column names to transform.
    /// - Throws: `PreprocessingError` if transformation fails.
    /// - Returns: A new `DataFrame` with transformed numeric column values.
    public func transform(_ df: DataFrame, columns: [String]) throws -> DataFrame {
        let result = try transform(df.toFeatureMatrix(columns))
        var out = df
        for (i, name) in columns.enumerated() {
            out = try out.withColumn(name, column: TypedColumn<Double>(
                name: name, values: result.map { Optional($0[i]) }
            ))
        }
        return out
    }

    /// Fits transformer parameters on specified DataFrame columns and transforms them in a single step.
    /// - Parameters:
    ///   - df: Target input `DataFrame`.
    ///   - columns: Array of column names to fit and transform.
    /// - Throws: `PreprocessingError` if fitting or transformation fails.
    /// - Returns: A new `DataFrame` with transformed numeric column values.
    public mutating func fitTransform(_ df: DataFrame, columns: [String]) throws -> DataFrame {
        try fit(df, columns: columns)
        return try transform(df, columns: columns)
    }
}

extension DataFrame {
    private func extractFeatures(columns names: [String], allowNaN: Bool = false) throws -> [[Double]] {
        return try toFeatureMatrix(names)
    }

    /// Fits a StandardScaler on the specified columns.
    public func fitStandardScaler(columns names: [String]) throws -> StandardScaler {
        let features = try extractFeatures(columns: names)
        var scaler = StandardScaler()
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted StandardScaler, returning a new DataFrame.
    public func standardScale(columns names: [String], scaler: StandardScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using StandardScaler, returning the scaled DataFrame and scaler.
    public func standardScale(columns names: [String]) throws -> (scaled: DataFrame, scaler: StandardScaler) {
        let scaler = try fitStandardScaler(columns: names)
        let scaledDf = try standardScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }

    /// Fits a MinMaxScaler on the specified columns.
    public func fitMinMaxScaler(columns names: [String]) throws -> MinMaxScaler {
        let features = try extractFeatures(columns: names)
        var scaler = MinMaxScaler()
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted MinMaxScaler, returning a new DataFrame.
    public func minMaxScale(columns names: [String], scaler: MinMaxScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using MinMaxScaler, returning the scaled DataFrame and scaler.
    public func minMaxScale(columns names: [String]) throws -> (scaled: DataFrame, scaler: MinMaxScaler) {
        let scaler = try fitMinMaxScaler(columns: names)
        let scaledDf = try minMaxScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }

    /// Fits a LabelEncoder and encodes a category column into integers. Supports String, Int64, and Double columns.
    public func labelEncode(column name: String) throws -> (encoded: DataFrame, encoder: LabelEncoder) {
        let rawValues: [String]

        if let col = self[column: name, as: String.self] {
            rawValues = col.values.map { $0 ?? "" }
        } else if let col = self[column: name, as: Int64.self] {
            rawValues = col.values.map { $0.map { String($0) } ?? "" }
        } else if let col = self[column: name, as: Double.self] {
            rawValues = col.values.map { $0.map { String($0) } ?? "" }
        } else {
            if self[column: name] == nil {
                throw DataFrameError.columnNotFound(name)
            } else {
                throw DataFrameError.castFailed(column: name, targetType: "String | Int64 | Double")
            }
        }

        let encoder = LabelEncoder()
        encoder.fit(rawValues)
        let labels = try encoder.transform(rawValues)
        
        let intValues: [Int64?] = labels.map { Int64($0) }
        let newCol = TypedColumn<Int64>(name: name, values: intValues)
        let encodedDf = try withColumn(name, column: newCol)
        return (encodedDf, encoder)
    }

    // MARK: - Imputer
    
    /// Fits an Imputer on the specified columns.
    public func fitImputer(columns names: [String], strategy: Imputer.Strategy = .mean) throws -> Imputer {
        let features = try extractFeatures(columns: names, allowNaN: true)
        var imputer = Imputer(strategy: strategy)
        try imputer.fit(features)
        return imputer
    }
    
    /// Imputes the specified columns using a fitted Imputer, returning a new DataFrame.
    public func impute(columns names: [String], imputer: Imputer) throws -> DataFrame {
        let features = try extractFeatures(columns: names, allowNaN: true)
        let imputed = try imputer.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in imputed[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and imputes the specified columns, returning the imputed DataFrame and the imputer.
    public func impute(columns names: [String], strategy: Imputer.Strategy = .mean) throws -> (imputed: DataFrame, imputer: Imputer) {
        let imputer = try fitImputer(columns: names, strategy: strategy)
        let imputedDf = try impute(columns: names, imputer: imputer)
        return (imputedDf, imputer)
    }
    
    // MARK: - Normalizer
    
    /// Normalizes the specified columns, returning the normalized DataFrame and normalizer.
    public func normalize(columns names: [String], norm: Normalizer.NormType = .l2) throws -> (normalized: DataFrame, normalizer: Normalizer) {
        let features = try extractFeatures(columns: names)
        var normalizer = Normalizer(norm: norm)
        try normalizer.fit(features)
        let normalized = try normalizer.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in normalized[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return (df, normalizer)
    }
    
    // MARK: - RobustScaler
    
    /// Fits a RobustScaler on the specified columns.
    public func fitRobustScaler(columns names: [String], withCentering: Bool = true, withScaling: Bool = true, quantileRange: (Double, Double) = (25.0, 75.0)) throws -> RobustScaler {
        let features = try extractFeatures(columns: names)
        var scaler = RobustScaler(withCentering: withCentering, withScaling: withScaling, quantileRange: quantileRange)
        try scaler.fit(features)
        return scaler
    }
    
    /// Scales the specified columns using a fitted RobustScaler, returning a new DataFrame.
    public func robustScale(columns names: [String], scaler: RobustScaler) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let scaled = try scaler.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in scaled[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and scales the specified columns using RobustScaler, returning the scaled DataFrame and scaler.
    public func robustScale(columns names: [String], withCentering: Bool = true, withScaling: Bool = true, quantileRange: (Double, Double) = (25.0, 75.0)) throws -> (scaled: DataFrame, scaler: RobustScaler) {
        let scaler = try fitRobustScaler(columns: names, withCentering: withCentering, withScaling: withScaling, quantileRange: quantileRange)
        let scaledDf = try robustScale(columns: names, scaler: scaler)
        return (scaledDf, scaler)
    }
    
    // MARK: - PowerTransformer
    
    /// Fits a PowerTransformer on the specified columns.
    public func fitPowerTransformer(columns names: [String], method: PowerTransformer.Method = .yeoJohnson, standardize: Bool = true) throws -> PowerTransformer {
        let features = try extractFeatures(columns: names)
        var transformer = PowerTransformer(method: method, standardize: standardize)
        try transformer.fit(features)
        return transformer
    }
    
    /// Transforms the specified columns using a fitted PowerTransformer, returning a new DataFrame.
    public func powerTransform(columns names: [String], transformer: PowerTransformer) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let transformed = try transformer.transform(features)
        
        var df = self
        for (colIdx, colName) in names.enumerated() {
            let colValues = (0..<shape.rows).map { rowIdx in transformed[rowIdx][colIdx] }
            let newCol = TypedColumn<Double>(name: colName, values: colValues)
            df = try df.withColumn(colName, column: newCol)
        }
        return df
    }
    
    /// Fits and transforms the specified columns using PowerTransformer, returning the transformed DataFrame and transformer.
    public func powerTransform(columns names: [String], method: PowerTransformer.Method = .yeoJohnson, standardize: Bool = true) throws -> (transformed: DataFrame, transformer: PowerTransformer) {
        let transformer = try fitPowerTransformer(columns: names, method: method, standardize: standardize)
        let transformedDf = try powerTransform(columns: names, transformer: transformer)
        return (transformedDf, transformer)
    }
    
    // MARK: - KBinsDiscretizer
    
    /// Fits a KBinsDiscretizer on the specified columns.
    public func fitKBinsDiscretizer(columns names: [String], nBins: Int = 5, strategy: KBinsDiscretizer.Strategy = .uniform, encode: KBinsDiscretizer.Encode = .ordinal) throws -> KBinsDiscretizer {
        let features = try extractFeatures(columns: names)
        var discretizer = KBinsDiscretizer(nBins: nBins, strategy: strategy, encode: encode)
        try discretizer.fit(features)
        return discretizer
    }
    
    /// Discretizes the specified columns using a fitted KBinsDiscretizer, returning a new DataFrame.
    public func kBinsDiscretize(columns names: [String], discretizer: KBinsDiscretizer) throws -> DataFrame {
        let features = try extractFeatures(columns: names)
        let binned = try discretizer.transform(features)
        
        var df = self
        
        switch discretizer.encode {
        case .ordinal:
            for (colIdx, colName) in names.enumerated() {
                let colValues = (0..<shape.rows).map { rowIdx in binned[rowIdx][colIdx] }
                let newCol = TypedColumn<Double>(name: colName, values: colValues)
                df = try df.withColumn(colName, column: newCol)
            }
        case .oneHot:
            df = try df.drop(names)
            for (colIdx, colName) in names.enumerated() {
                for bin in 0..<discretizer.nBins {
                    let binColName = "\(colName)_bin_\(bin)"
                    let colValues = (0..<shape.rows).map { rowIdx in
                        binned[rowIdx][colIdx * discretizer.nBins + bin]
                    }
                    let binCol = TypedColumn<Double>(name: binColName, values: colValues)
                    df = try df.withColumn(binColName, column: binCol)
                }
            }
        }
        
        return df
    }
    
    /// Fits and discretizes the specified columns, returning the discretized DataFrame and discretizer.
    public func kBinsDiscretize(columns names: [String], nBins: Int = 5, strategy: KBinsDiscretizer.Strategy = .uniform, encode: KBinsDiscretizer.Encode = .ordinal) throws -> (discretized: DataFrame, discretizer: KBinsDiscretizer) {
        let discretizer = try fitKBinsDiscretizer(columns: names, nBins: nBins, strategy: strategy, encode: encode)
        let discretizedDf = try kBinsDiscretize(columns: names, discretizer: discretizer)
        return (discretizedDf, discretizer)
    }

    // MARK: - Time Series Rolling & EWMA Features

    /// Adds a rolling mean column for the specified numeric column.
    public func withRollingMean(column name: String, window: Int) throws -> DataFrame {
        guard let col = self[column: name, as: Double.self] else {
            throw DataFrameError.columnNotFound(name)
        }
        let values = col.values
        let n = values.count
        var rolling = [Double?](repeating: nil, count: n)

        for i in 0..<n {
            let start = max(0, i - window + 1)
            let slice = values[start...i].compactMap { $0 }
            rolling[i] = slice.isEmpty ? nil : slice.reduce(0, +) / Double(slice.count)
        }

        let newName = "\(name)_rolling_mean_\(window)"
        let newCol = TypedColumn<Double>(name: newName, values: rolling)
        return try withColumn(newName, column: newCol)
    }

    /// Adds a rolling standard deviation column for the specified numeric column.
    public func withRollingStd(column name: String, window: Int) throws -> DataFrame {
        guard let col = self[column: name, as: Double.self] else {
            throw DataFrameError.columnNotFound(name)
        }
        let values = col.values
        let n = values.count
        var rolling = [Double?](repeating: nil, count: n)

        for i in 0..<n {
            let start = max(0, i - window + 1)
            let slice = values[start...i].compactMap { $0 }
            if slice.count <= 1 {
                rolling[i] = slice.isEmpty ? nil : 0.0
            } else {
                let mean = slice.reduce(0, +) / Double(slice.count)
                let variance = slice.map { pow($0 - mean, 2) }.reduce(0, +) / Double(slice.count - 1)
                rolling[i] = sqrt(variance)
            }
        }

        let newName = "\(name)_rolling_std_\(window)"
        let newCol = TypedColumn<Double>(name: newName, values: rolling)
        return try withColumn(newName, column: newCol)
    }

    /// Adds an Exponentially Weighted Moving Average (EWMA) column for the specified numeric column.
    public func withEWMA(column name: String, alpha: Double) throws -> DataFrame {
        guard alpha > 0 && alpha <= 1.0 else {
            throw DataFrameError.invalidParameter("alpha must be in (0, 1], got \(alpha)")
        }
        guard let col = self[column: name, as: Double.self] else {
            throw DataFrameError.columnNotFound(name)
        }
        let values = col.values
        let n = values.count
        var ewma = [Double?](repeating: nil, count: n)

        var prev: Double? = nil
        for i in 0..<n {
            if let val = values[i] {
                if let currentPrev = prev {
                    let current = alpha * val + (1.0 - alpha) * currentPrev
                    ewma[i] = current
                    prev = current
                } else {
                    ewma[i] = val
                    prev = val
                }
            } else {
                ewma[i] = nil
            }
        }

        let alphaStr = String(format: "%.2f", alpha)
        let newName = "\(name)_ewma_\(alphaStr)"
        let newCol = TypedColumn<Double>(name: newName, values: ewma)
        return try withColumn(newName, column: newCol)
    }
}

