import Foundation

/// Decomposed time series components.
public struct DecompositionResult: Sendable {
    public let trend: [Double]
    public let seasonal: [Double]
    public let residual: [Double]
    public let original: [Double]
    
    public init(trend: [Double], seasonal: [Double], residual: [Double], original: [Double]) {
        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual
        self.original = original
    }
}

/// Forecast output with optional confidence intervals.
public struct ForecastResult: Sendable {
    public let predictions: [Double]
    public let lowerBound: [Double]?
    public let upperBound: [Double]?
    public let fittedValues: [Double]
    public let residuals: [Double]
    public let aic: Double?
    public let mse: Double
    public let mae: Double
    
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
    public let order: (p: Int, d: Int, q: Int)
    public let arCoefficients: [Double]
    public let maCoefficients: [Double]
    public let intercept: Double
    public let exogCoefficients: [Double]
    public let forecast: ForecastResult
    
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
    public let mean: [Double]      // state vector x̂
    public let covariance: [[Double]] // P matrix (row-major 2D)
    
    public init(mean: [Double], covariance: [[Double]]) {
        self.mean = mean
        self.covariance = covariance
    }
}
