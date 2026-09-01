import Foundation

/// Nelder-Mead Simplex optimization algorithm for derivative-free non-linear function minimization.
///
/// Efficiently finds optimal smoothing parameters (alpha, beta, gamma) within bounded domains
/// in ~50-150 iterations, replacing coarse grid searches.
public struct NelderMead: Sendable {
    
    /// Maximum number of iterations before terminating.
    public let maxIterations: Int
    
    /// Convergence tolerance on function value variance across the simplex.
    public let tolerance: Double
    
    /// Creates a new Nelder-Mead optimizer instance.
    /// - Parameters:
    ///   - maxIterations: Maximum iterations (default: 500).
    ///   - tolerance: Function convergence tolerance (default: 1e-6).
    public init(maxIterations: Int = 500, tolerance: Double = 1e-6) {
        self.maxIterations = maxIterations
        self.tolerance = tolerance
    }
    
    /// Minimizes an objective function starting from an initial guess with optional parameter bounds.
    /// - Parameters:
    ///   - objective: The scalar cost function to minimize.
    ///   - initialGuess: Initial vector of parameters.
    ///   - lowerBounds: Optional minimum allowed value for each parameter.
    ///   - upperBounds: Optional maximum allowed value for each parameter.
    /// - Returns: Optimized parameter vector.
    public func minimize(
        objective: ([Double]) -> Double,
        initialGuess: [Double],
        lowerBounds: [Double]? = nil,
        upperBounds: [Double]? = nil
    ) -> [Double] {
        let n = initialGuess.count
        guard n > 0 else { return initialGuess }
        
        let clamp = { (v: [Double]) -> [Double] in
            var res = v
            for i in 0..<n {
                if let lb = lowerBounds?[i] { res[i] = max(lb, res[i]) }
                if let ub = upperBounds?[i] { res[i] = min(ub, res[i]) }
            }
            return res
        }
        
        // Coefficients
        let alpha = 1.0   // Reflection
        let gamma = 2.0   // Expansion
        let rho   = 0.5   // Contraction
        let sigma = 0.5   // Shrink
        
        // Construct initial simplex with (n + 1) vertices
        var simplex: [[Double]] = [clamp(initialGuess)]
        for i in 0..<n {
            var vertex = initialGuess
            let step = abs(vertex[i]) > 1e-4 ? vertex[i] * 0.05 : 0.00025
            vertex[i] += step
            simplex.append(clamp(vertex))
        }
        
        var values = simplex.map { objective($0) }
        
        for _ in 0..<maxIterations {
            // Sort simplex by function values
            let order = (0...(n)).sorted { values[$0] < values[$1] }
            simplex = order.map { simplex[$0] }
            values  = order.map { values[$0] }
            
            // Check termination criteria (standard deviation of simplex values)
            let meanVal = values.reduce(0.0, +) / Double(n + 1)
            let variance = values.reduce(0.0) { $0 + ($1 - meanVal) * ($1 - meanVal) } / Double(n + 1)
            if sqrt(variance) < tolerance {
                break
            }
            
            // Centroid of the best n vertices (excluding worst vertex x_{n+1})
            var centroid = [Double](repeating: 0.0, count: n)
            for i in 0..<n {
                for d in 0..<n {
                    centroid[d] += simplex[i][d]
                }
            }
            for d in 0..<n {
                centroid[d] /= Double(n)
            }
            
            // 1. Reflection
            var xr = [Double](repeating: 0.0, count: n)
            for d in 0..<n {
                xr[d] = centroid[d] + alpha * (centroid[d] - simplex[n][d])
            }
            xr = clamp(xr)
            let fr = objective(xr)
            
            if values[0] <= fr && fr < values[n - 1] {
                simplex[n] = xr
                values[n] = fr
                continue
            }
            
            // 2. Expansion
            if fr < values[0] {
                var xe = [Double](repeating: 0.0, count: n)
                for d in 0..<n {
                    xe[d] = centroid[d] + gamma * (xr[d] - centroid[d])
                }
                xe = clamp(xe)
                let fe = objective(xe)
                if fe < fr {
                    simplex[n] = xe
                    values[n] = fe
                } else {
                    simplex[n] = xr
                    values[n] = fr
                }
                continue
            }
            
            // 3. Contraction
            var xc = [Double](repeating: 0.0, count: n)
            if fr < values[n] {
                // Outside contraction
                for d in 0..<n {
                    xc[d] = centroid[d] + rho * (xr[d] - centroid[d])
                }
                xc = clamp(xc)
                let fc = objective(xc)
                if fc <= fr {
                    simplex[n] = xc
                    values[n] = fc
                    continue
                }
            } else {
                // Inside contraction
                for d in 0..<n {
                    xc[d] = centroid[d] + rho * (simplex[n][d] - centroid[d])
                }
                xc = clamp(xc)
                let fc = objective(xc)
                if fc < values[n] {
                    simplex[n] = xc
                    values[n] = fc
                    continue
                }
            }
            
            // 4. Shrink
            for i in 1...n {
                for d in 0..<n {
                    simplex[i][d] = simplex[0][d] + sigma * (simplex[i][d] - simplex[0][d])
                }
                simplex[i] = clamp(simplex[i])
                values[i] = objective(simplex[i])
            }
        }
        
        let bestIdx = values.enumerated().min(by: { $0.element < $1.element })?.offset ?? 0
        return simplex[bestIdx]
    }
}
