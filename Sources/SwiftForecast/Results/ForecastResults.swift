import Foundation

/// Decomposed time series components.
public struct DecompositionResult: Sendable {
    /// The trend.
    public let trend: [Double]
    /// The seasonal.
    public let seasonal: [Double]
    /// The residual.
    public let residual: [Double]
    /// The original.
    public let original: [Double]
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - trend: The trend.
    ///   - seasonal: The seasonal.
    ///   - residual: The residual.
    ///   - original: The original.
    public init(trend: [Double], seasonal: [Double], residual: [Double], original: [Double]) {
        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual
        self.original = original
    }
}

/// Forecast output with optional confidence intervals.
public struct ForecastResult: Sendable {
    /// The predictions.
    public let predictions: [Double]
    /// The lower bound.
    public let lowerBound: [Double]?
    /// The upper bound.
    public let upperBound: [Double]?
    /// The fitted values.
    public let fittedValues: [Double]
    /// The residuals.
    public let residuals: [Double]
    /// The aic.
    public let aic: Double?
    /// The mse.
    public let mse: Double
    /// The mae.
    public let mae: Double
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - predictions: The predictions.
    ///   - lowerBound: The lower bound.
    ///   - upperBound: The upper bound.
    ///   - fittedValues: The fitted values.
    ///   - residuals: The residuals.
    ///   - aic: The aic.
    ///   - mse: The mse.
    ///   - mae: The mae.
    public init(
        predictions: [Double],
        lowerBound: [Double]? = nil,
        upperBound: [Double]? = nil,
        fittedValues: [Double],
        residuals: [Double],
        aic: Double? = nil,
        mse: Double,
        mae: Double
    ) {
        self.predictions = predictions
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.fittedValues = fittedValues
        self.residuals = residuals
        self.aic = aic
        self.mse = mse
        self.mae = mae
    }
}

/// ARIMA-specific result.
public struct ARIMAResult: Sendable {
    /// The order.
    public let order: (p: Int, d: Int, q: Int)
    /// The ar coefficients.
    public let arCoefficients: [Double]
    /// The ma coefficients.
    public let maCoefficients: [Double]
    /// The intercept.
    public let intercept: Double
    /// The exog coefficients.
    public let exogCoefficients: [Double]
    /// The forecast.
    public let forecast: ForecastResult
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - order: The order.
    ///   - d: The d.
    ///   - q: The q.
    ///   - arCoefficients: The ar coefficients.
    ///   - maCoefficients: The ma coefficients.
    ///   - intercept: The intercept.
    ///   - exogCoefficients: The exog coefficients.
    ///   - forecast: The forecast.
    public init(
        order: (p: Int, d: Int, q: Int),
        arCoefficients: [Double],
        maCoefficients: [Double],
        intercept: Double,
        exogCoefficients: [Double] = [],
        forecast: ForecastResult
    ) {
        self.order = order
        self.arCoefficients = arCoefficients
        self.maCoefficients = maCoefficients
        self.intercept = intercept
        self.exogCoefficients = exogCoefficients
        self.forecast = forecast
    }
}
/// Kalman Filter state estimate.
public struct KalmanState: Sendable {
    /// The mean.
    public let mean: [Double]      // state vector x̂
    /// The covariance.
    public let covariance: [[Double]] // P matrix (row-major 2D)
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - mean: The mean.
    ///   - covariance: The covariance.
    public init(mean: [Double], covariance: [[Double]]) {
        self.mean = mean
        self.covariance = covariance
    }
}
