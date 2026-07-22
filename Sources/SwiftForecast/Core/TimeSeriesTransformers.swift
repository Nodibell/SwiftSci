import Foundation
import SwiftStats

/// LagTransformer constructs lagged feature columns for time-series supervised learning.
public final class LagTransformer: Sendable {
    public let lags: [Int]
    
    public init(lags: [Int]) {
        self.lags = lags.filter { $0 > 0 }.sorted()
    }
    
    /// Generates feature matrix of lagged values and aligned target values.
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
    public let windowSize: Int
    
    public init(windowSize: Int) {
        self.windowSize = max(1, windowSize)
    }
    
    /// Computes rolling mean and rolling standard deviation series.
    public func transform(series: [Double]) -> (rollingMean: [Double], rollingStd: [Double]) {
        guard series.count >= windowSize else {
            return (rollingMean: series, rollingStd: [Double](repeating: 0.0, count: series.count))
        }
        
        var means = [Double](repeating: 0.0, count: series.count)
        var stds = [Double](repeating: 0.0, count: series.count)
        
        for i in 0..<series.count {
            let start = max(0, i - windowSize + 1)
            let window = Array(series[start...i])
            let count = Double(window.count)
            let mean = window.reduce(0.0, +) / count
            means[i] = mean
            
            if count > 1 {
                let variance = window.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / (count - 1.0)
                stds[i] = sqrt(max(0.0, variance))
            } else {
                stds[i] = 0.0
            }
        }
        
        return (rollingMean: means, rollingStd: stds)
    }
}

/// ExpandingWindow computes cumulative expanding statistics (expanding mean and expanding std dev) over a time series.
public final class ExpandingWindow: Sendable {
    public let minPeriods: Int

    public init(minPeriods: Int = 1) {
        self.minPeriods = max(1, minPeriods)
    }

    /// Computes expanding mean and expanding standard deviation series.
    public func transform(series: [Double]) -> (expandingMean: [Double], expandingStd: [Double]) {
        guard !series.isEmpty else {
            return (expandingMean: [], expandingStd: [])
        }

        var means = [Double](repeating: 0.0, count: series.count)
        var stds = [Double](repeating: 0.0, count: series.count)

        var sum = 0.0
        var sumSq = 0.0

        for i in 0..<series.count {
            let val = series[i]
            sum += val
            sumSq += val * val
            let count = Double(i + 1)

            if (i + 1) >= minPeriods {
                let mean = sum / count
                means[i] = mean

                if count > 1 {
                    let varUnbiased = (sumSq - (sum * sum) / count) / (count - 1.0)
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
