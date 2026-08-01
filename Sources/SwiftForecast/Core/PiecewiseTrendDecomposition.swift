import Foundation

/// Growth trend type for Prophet-style decomposition.
public enum GrowthType: String, Sendable {
    case linear
    case logistic
}

/// Prophet-style Piecewise Linear & Logistic Trend Decomposition model.
public actor PiecewiseTrendDecomposition {
    public let growth: GrowthType
    public let nChangepoints: Int
    public let capacity: Double

    private var changepointIndices: [Int] = []
    private var baseGrowth: Double = 0.0
    private var baseOffset: Double = 0.0
    private var deltaSlopes: [Double] = []
    private var isFitted: Bool = false
    private var seriesLength: Int = 0

    public init(
        growth: GrowthType = .linear,
        nChangepoints: Int = 5,
        capacity: Double = 1.0
    ) {
        self.growth = growth
        self.nChangepoints = max(1, nChangepoints)
        self.capacity = capacity
    }

    /// Fits piecewise linear/logistic trend on the given time series.
    public func fit(series: [Double]) async throws {
        guard series.count >= 4 else {
            throw ForecastError.insufficientLength(minimum: 4, got: series.count)
        }
        
        self.seriesLength = series.count
        let n = series.count

        // Uniformly distribute candidate changepoints across 80% of series length
        let maxIndex = Int(Double(n) * 0.8)
        let step = max(1, maxIndex / (nChangepoints + 1))
        self.changepointIndices = (1...nChangepoints).map { i in min(maxIndex, i * step) }

        // Initial linear fit for baseline slope and intercept in index units
        let tValues = (0..<n).map { Double($0) }
        let meanT = Double(n - 1) / 2.0
        let meanY = series.reduce(0.0, +) / Double(n)

        var num = 0.0
        var den = 0.0
        for i in 0..<n {
            num += (tValues[i] - meanT) * (series[i] - meanY)
            den += (tValues[i] - meanT) * (tValues[i] - meanT)
        }

        self.baseGrowth = den != 0 ? num / den : 0.0
        self.baseOffset = meanY - baseGrowth * meanT

        // Compute slope adjustments (deltas) at each changepoint
        var deltas = [Double](repeating: 0.0, count: changepointIndices.count)
        var currentSlope = baseGrowth

        for (idx, cp) in changepointIndices.enumerated() {
            let leftStart = max(0, cp - step)
            let rightEnd = min(n - 1, cp + step)
            if rightEnd > leftStart {
                let localSlope = (series[rightEnd] - series[leftStart]) / Double(rightEnd - leftStart)
                deltas[idx] = (localSlope - currentSlope) * 0.5
                currentSlope = localSlope
            }
        }

        self.deltaSlopes = deltas
        self.isFitted = true
    }

    /// Generates predicted trend components for `steps` future points.
    public func predict(steps: Int) async throws -> [Double] {
        guard isFitted else {
            throw ForecastError.emptyTimeSeries
        }
        guard steps > 0 else { return [] }

        var predictions: [Double] = []

        for h in 1...steps {
            let t = Double(seriesLength - 1 + h)
            var currentGrowth = baseGrowth

            for (idx, cp) in changepointIndices.enumerated() {
                if t >= Double(cp) {
                    currentGrowth += deltaSlopes[idx]
                }
            }

            let trendVal: Double
            if growth == .linear {
                trendVal = baseOffset + currentGrowth * t
            } else {
                // Logistic growth bounded by capacity
                let expVal = exp(-currentGrowth * (t - baseOffset / max(1e-6, baseGrowth)))
                trendVal = capacity / (1.0 + expVal)
            }

            predictions.append(trendVal)
        }

        return predictions
    }
}
