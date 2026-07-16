import Accelerate

// MARK: – Correlation & Covariance

extension Stats {

    /// Pearson correlation coefficient using vDSP primitives.
    /// r = Σ((x-μx)(y-μy)) / (n * σx * σy)
    public static func pearsonCorrelation(_ x: [Double], _ y: [Double]) throws -> Double {
        try requireNonEmpty(x)
        try requireSameSize(x, y)
        try requireNoNaN(x)
        try requireNoNaN(y)

        let _    = Double(x.count)   // reserved for future weight-based overloads
        let muX  = vDSP.mean(x)
        let muY  = vDSP.mean(y)

        // Centred vectors
        var cx = [Double](repeating: 0, count: x.count)
        var cy = [Double](repeating: 0, count: y.count)
        vDSP.add(-muX, x, result: &cx)
        vDSP.add(-muY, y, result: &cy)

        let numerator   = vDSP.dot(cx, cy)
        let denominator = vDSP.sumOfSquares(cx).squareRoot() *
                          vDSP.sumOfSquares(cy).squareRoot()

        guard denominator > 0 else { throw StatsError.divisionByZero(context: "pearsonCorrelation") }
        let r = numerator / denominator
        // Clamp to [-1, 1] to absorb floating-point rounding
        return Swift.max(-1.0, Swift.min(1.0, r))
    }

    /// Spearman rank correlation: Pearson applied to ranks.
    public static func spearmanCorrelation(_ x: [Double], _ y: [Double]) throws -> Double {
        try requireNonEmpty(x)
        try requireSameSize(x, y)
        let rx = rank(x)
        let ry = rank(y)
        return try pearsonCorrelation(rx, ry)
    }

    /// Sample or population covariance.
    public static func covariance(_ x: [Double], _ y: [Double], ddof: Int = 1) throws -> Double {
        try requireNonEmpty(x)
        try requireSameSize(x, y)
        try requireNoNaN(x)
        try requireNoNaN(y)
        guard ddof >= 0 else { throw StatsError.invalidDDOF(ddof) }
        let n = Double(x.count)
        guard n > Double(ddof) else {
            throw StatsError.insufficientData(minimum: ddof + 1, got: x.count)
        }

        let muX = vDSP.mean(x)
        let muY = vDSP.mean(y)
        var cx  = [Double](repeating: 0, count: x.count)
        var cy  = [Double](repeating: 0, count: y.count)
        vDSP.add(-muX, x, result: &cx)
        vDSP.add(-muY, y, result: &cy)

        return vDSP.dot(cx, cy) / (n - Double(ddof))
    }

    /// Full correlation matrix for a list of variable vectors.
    /// Result[i][j] = pearsonCorrelation(data[i], data[j]).
    public static func correlationMatrix(_ data: [[Double]]) throws -> [[Double]] {
        let k = data.count
        guard k >= 2 else { throw StatsError.invalidGroupCount(minimum: 2, got: k) }
        var matrix = [[Double]](repeating: [Double](repeating: 0, count: k), count: k)
        for i in 0..<k {
            matrix[i][i] = 1.0
            for j in (i+1)..<k {
                let r = try pearsonCorrelation(data[i], data[j])
                matrix[i][j] = r
                matrix[j][i] = r
            }
        }
        return matrix
    }

    // MARK: – Ranking helper

    /// Computes average ranks (handles ties with average rank).
    internal static func rank(_ values: [Double]) -> [Double] {
        let n = values.count
        let sorted = values.enumerated().sorted { $0.element < $1.element }
        var ranks = [Double](repeating: 0, count: n)

        var i = 0
        while i < n {
            var j = i
            while j < n - 1 && sorted[j].element == sorted[j+1].element { j += 1 }
            // Average rank for tied group (1-indexed)
            let avgRank = Double(i + j) / 2.0 + 1.0
            for k in i...j { ranks[sorted[k].offset] = avgRank }
            i = j + 1
        }
        return ranks
    }
}
