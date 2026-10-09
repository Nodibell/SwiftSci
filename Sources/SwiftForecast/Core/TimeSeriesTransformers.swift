import Foundation
import SwiftStats

/// LagTransformer constructs lagged feature columns for time-series supervised learning.
public final class LagTransformer: Sendable {
    /// The lags.
    public let lags: [Int]
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - lags: The lags.
    public init(lags: [Int]) {
        self.lags = lags.filter { $0 > 0 }.sorted()
    }
    
    /// Generates feature matrix of lagged values and aligned target values.
    /// - Parameters:
    ///   - series: 1D temporal time-series observations array.
    /// - Returns: Named tuple containing generated feature matrix `features` and target vector `targets`.
    public func transform(series: [Double]) -> (features: [[Double]], targets: [Double]) {
        guard !series.isEmpty, !lags.isEmpty else {
            return (features: [], targets: [])
        }
        let maxLag = lags.max()!
        guard series.count > maxLag else {
            return (features: [], targets: [])
        }
        
        var features: [[Double]] = []
        var targets: [Double] = []
        
        for t in maxLag..<series.count {
            let row = lags.map { lag in series[t - lag] }
            features.append(row)
            targets.append(series[t])
        }
        
        return (features: features, targets: targets)
    }
}

/// RollingWindow computes sliding rolling window statistics (mean and std dev) over a time series.
public final class RollingWindow: Sendable {
    /// The window size.
    public let windowSize: Int
    
    /// Creates a new instance.
    /// - Parameters:
    ///   - windowSize: The window size.
    public init(windowSize: Int) {
        self.windowSize = max(1, windowSize)
    }
    
    /// Computes rolling mean and rolling standard deviation series.
    /// - Parameters:
    ///   - series: 1D temporal time-series observations array.
    /// - Throws: `ForecastError` if series length is insufficient, values contain NaNs, or model is unfitted.
    /// - Returns: The computed (rollingMean: [Double], rollingStd: [Double]) result instance.
    public func transform(series: [Double]) throws -> (rollingMean: [Double], rollingStd: [Double]) {

        guard series.count >= windowSize else {
            return (rollingMean: series, rollingStd: [Double](repeating: 0.0, count: series.count))
        }
        
        var means = [Double](repeating: 0.0, count: series.count)
        var stds = [Double](repeating: 0.0, count: series.count)
        
        // M-03: O(N) running accumulator with offset shift to prevent catastrophic cancellation
        let c = series[0]
        var runningSum = 0.0
        var runningSumSq = 0.0
        
        for i in 0..<series.count {
            let yIn = series[i] - c
            runningSum += yIn
            runningSumSq += yIn * yIn
            
            if i >= windowSize {
                let yOut = series[i - windowSize] - c
                runningSum -= yOut
                runningSumSq -= yOut * yOut
            }
            
            let count = Double(min(i + 1, windowSize))
            let meanY = runningSum / count
            means[i] = c + meanY
            
            if count > 1.0 {
                let varUnbiased = (runningSumSq - (runningSum * runningSum) / count) / (count - 1.0)
                stds[i] = sqrt(max(0.0, varUnbiased))
            } else {
                stds[i] = 0.0
            }
        }
        
        return (rollingMean: means, rollingStd: stds)
    }
}

/// ExpandingWindow computes cumulative expanding statistics (expanding mean and expanding std dev) over a time series.
public final class ExpandingWindow: Sendable {
    /// The min periods.
    public let minPeriods: Int

    /// Creates a new instance.
    /// - Parameters:
    ///   - minPeriods: The min periods.
    public init(minPeriods: Int = 1) {
        self.minPeriods = max(1, minPeriods)
    }

    /// Computes expanding mean and expanding standard deviation series.
    /// - Parameters:
    ///   - series: 1D temporal time-series observations array.
    /// - Returns: The computed (expandingMean: [Double], expandingStd: [Double]) result instance.
    public func transform(series: [Double]) -> (expandingMean: [Double], expandingStd: [Double]) {
        guard !series.isEmpty else {
            return (expandingMean: [], expandingStd: [])
        }

        var means = [Double](repeating: 0.0, count: series.count)
        var stds = [Double](repeating: 0.0, count: series.count)

        // M-04: Numerically stable one-pass Welford algorithm
        var count = 0
        var mean = 0.0
        var M2 = 0.0

        for i in 0..<series.count {
            let val = series[i]
            count += 1
            let delta = val - mean
            mean += delta / Double(count)
            let delta2 = val - mean
            M2 += delta * delta2

            if count >= minPeriods {
                means[i] = mean
                if count > 1 {
                    let varUnbiased = M2 / Double(count - 1)
                    stds[i] = sqrt(max(0.0, varUnbiased))
                } else {
                    stds[i] = 0.0
                }
            } else {
                means[i] = Double.nan
                stds[i] = Double.nan
            }
        }
        
        return (expandingMean: means, expandingStd: stds)
    }
}
