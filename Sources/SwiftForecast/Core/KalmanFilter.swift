import Foundation
import Accelerate

// MARK: - KalmanFilter (3.5 Flat Buffer Optimization)
//
// All internal state matrices (F, H, Q, R, P) and temporary variables
// are stored as contiguous row-major [Double] arrays.
// Operations use BLAS (forecast_dgemm, forecast_dgemv), vDSP, and LAPACK (dgesv)
// directly on flat pointers, eliminating [[Double]] heap fragmentation and flatMap copies.

/// Linear Kalman Filter state estimation actor for 1D/multivariate tracking.
public actor KalmanFilter {
    /// The dimension of the state vector.
    public let stateSize: Int

    /// The dimension of the measurement observation vector.
    public let observationSize: Int

    // Row-major flat matrix buffers
    private var F: [Double] = [] // State transition matrix (stateSize x stateSize)
    private var H: [Double] = [] // Observation matrix (observationSize x stateSize)
    private var Q: [Double] = [] // Process noise covariance (stateSize x stateSize)
    private var R: [Double] = [] // Measurement noise covariance (observationSize x observationSize)

    private var x: [Double] = [] // State estimate mean vector (stateSize)
    private var P: [Double] = [] // State estimate covariance (stateSize x stateSize)
    private var isInitialized = false

    /// Creates a new KalmanFilter instance.
    /// - Parameters:
    ///   - stateSize: Dimension of the state vector.
    ///   - observationSize: Dimension of the observation vector.
    /// - Throws: `ForecastError.invalidAROrder` / `invalidMAOrder` if dimensions <= 0.
    public init(stateSize: Int, observationSize: Int) throws {
        guard stateSize > 0 else { throw ForecastError.invalidAROrder(stateSize) }
        guard observationSize > 0 else { throw ForecastError.invalidMAOrder(observationSize) }
        self.stateSize = stateSize
        self.observationSize = observationSize
    }

    /// Sets the state transition matrix F (stateSize x stateSize).
    public func setTransitionMatrix(_ matrix: [[Double]]) throws {
        try validateMatrixDimensions(matrix, expectedRows: stateSize, expectedCols: stateSize)
        self.F = flatten(matrix)
    }

    /// Sets the observation matrix H (observationSize x stateSize).
    public func setObservationMatrix(_ matrix: [[Double]]) throws {
        try validateMatrixDimensions(matrix, expectedRows: observationSize, expectedCols: stateSize)
        self.H = flatten(matrix)
    }

    /// Sets the process noise covariance matrix Q (stateSize x stateSize).
    public func setProcessNoise(_ matrix: [[Double]]) throws {
        try validateMatrixDimensions(matrix, expectedRows: stateSize, expectedCols: stateSize)
        self.Q = flatten(matrix)
    }

    /// Sets the measurement noise covariance matrix R (observationSize x observationSize).
    public func setMeasurementNoise(_ matrix: [[Double]]) throws {
        try validateMatrixDimensions(matrix, expectedRows: observationSize, expectedCols: observationSize)
        self.R = flatten(matrix)
    }

    /// Sets the initial state mean vector and covariance matrix.
    public func setInitialState(mean: [Double], covariance: [[Double]]) throws {
        guard mean.count == stateSize else {
            throw ForecastError.matrixDimensionMismatch(
                expectedRows: stateSize, expectedCols: 1,
                gotRows: mean.count, gotCols: 1
            )
        }
        try validateMatrixDimensions(covariance, expectedRows: stateSize, expectedCols: stateSize)
        self.x = mean
        self.P = flatten(covariance)
        self.isInitialized = true
    }

    /// Run Kalman Filter forward over observations.
    public func filter(observations: [[Double]]) throws -> [KalmanState] {
        try checkInitialization()

        let n = stateSize
        let m = observationSize
        var states: [KalmanState] = []
        states.reserveCapacity(observations.count)

        let FT = transposeFlat(F, rows: n, cols: n)
        let HT = transposeFlat(H, rows: m, cols: n)
        let I = identityFlat(n)

        for z in observations {
            guard z.count == m else {
                throw ForecastError.matrixDimensionMismatch(
                    expectedRows: m, expectedCols: 1,
                    gotRows: z.count, gotCols: 1
                )
            }

            // 1. Predict
            // xPred = F * x
            let xPred = matVecMulFlat(F, rows: n, cols: n, vec: x)
            // PPred = F * P * F^T + Q
            let FP = matMulFlat(F, rA: n, cA: n, P, rB: n, cB: n)
            let FPFT = matMulFlat(FP, rA: n, cA: n, FT, rB: n, cB: n)
            let PPred = vecAddFlat(FPFT, Q)

            // 2. Innovation
            // y = z - H * xPred
            let HxPred = matVecMulFlat(H, rows: m, cols: n, vec: xPred)
            let y = vecSubFlat(z, HxPred)

            // Innovation covariance S = H * PPred * H^T + R
            let HP = matMulFlat(H, rA: m, cA: n, PPred, rB: n, cB: n)
            let HPHT = matMulFlat(HP, rA: m, cA: n, HT, rB: n, cB: m)
            let S = vecAddFlat(HPHT, R)

            // SInv
            let SInv = try matInverseFlat(S, n: m)

            // Kalman gain K = PPred * H^T * SInv (n x m)
            let PPredHT = matMulFlat(PPred, rA: n, cA: n, HT, rB: n, cB: m)
            let K = matMulFlat(PPredHT, rA: n, cA: m, SInv, rB: m, cB: m)

            // Updated state: x = xPred + K * y
            let Ky = matVecMulFlat(K, rows: n, cols: m, vec: y)
            self.x = vecAddFlat(xPred, Ky)

            // Updated covariance (Joseph form): P = (I - K*H) * PPred * (I - K*H)^T + K * R * K^T
            let KH = matMulFlat(K, rA: n, cA: m, H, rB: m, cB: n)
            let IMinusKH = vecSubFlat(I, KH)
            let IMinusKHT = transposeFlat(IMinusKH, rows: n, cols: n)
            let term1_temp = matMulFlat(IMinusKH, rA: n, cA: n, PPred, rB: n, cB: n)
            let term1 = matMulFlat(term1_temp, rA: n, cA: n, IMinusKHT, rB: n, cB: n)

            let KR = matMulFlat(K, rA: n, cA: m, R, rB: m, cB: m)
            let KT = transposeFlat(K, rows: n, cols: m)
            let term2 = matMulFlat(KR, rA: n, cA: m, KT, rB: m, cB: n)

            self.P = vecAddFlat(term1, term2)

            states.append(KalmanState(mean: self.x, covariance: unflatten(self.P, rows: n, cols: n)))
        }

        return states
    }

    /// RTS (Rauch-Tung-Striebel) smoother.
    public func smooth(observations: [[Double]]) throws -> [KalmanState] {
        try checkInitialization()

        let nObs = observations.count
        guard nObs > 0 else { return [] }

        let n = stateSize
        let m = observationSize
        let FT = transposeFlat(F, rows: n, cols: n)
        let HT = transposeFlat(H, rows: m, cols: n)
        let I = identityFlat(n)

        // Forward pass
        var xFilt: [[Double]] = []
        var PFilt: [[Double]] = []
        var xPred: [[Double]] = []
        var PPred: [[Double]] = []

        xFilt.reserveCapacity(nObs)
        PFilt.reserveCapacity(nObs)
        xPred.reserveCapacity(nObs)
        PPred.reserveCapacity(nObs)

        for z in observations {
            let xp = matVecMulFlat(F, rows: n, cols: n, vec: x)
            let FP = matMulFlat(F, rA: n, cA: n, P, rB: n, cB: n)
            let FPFT = matMulFlat(FP, rA: n, cA: n, FT, rB: n, cB: n)
            let Pp = vecAddFlat(FPFT, Q)

            xPred.append(xp)
            PPred.append(Pp)

            let Hxp = matVecMulFlat(H, rows: m, cols: n, vec: xp)
            let y = vecSubFlat(z, Hxp)
            let HP = matMulFlat(H, rA: m, cA: n, Pp, rB: n, cB: n)
            let HPHT = matMulFlat(HP, rA: m, cA: n, HT, rB: n, cB: m)
            let S = vecAddFlat(HPHT, R)
            let SInv = try matInverseFlat(S, n: m)
            let PHT = matMulFlat(Pp, rA: n, cA: n, HT, rB: n, cB: m)
            let K = matMulFlat(PHT, rA: n, cA: m, SInv, rB: m, cB: m)

            let Ky = matVecMulFlat(K, rows: n, cols: m, vec: y)
            self.x = vecAddFlat(xp, Ky)

            let KH = matMulFlat(K, rA: n, cA: m, H, rB: m, cB: n)
            let IMinusKH = vecSubFlat(I, KH)
            let IMinusKHT = transposeFlat(IMinusKH, rows: n, cols: n)
            let term1_temp = matMulFlat(IMinusKH, rA: n, cA: n, Pp, rB: n, cB: n)
            let term1 = matMulFlat(term1_temp, rA: n, cA: n, IMinusKHT, rB: n, cB: n)
            let KR = matMulFlat(K, rA: n, cA: m, R, rB: m, cB: m)
            let KT = transposeFlat(K, rows: n, cols: m)
            let term2 = matMulFlat(KR, rA: n, cA: m, KT, rB: m, cB: n)
            self.P = vecAddFlat(term1, term2)

            xFilt.append(self.x)
            PFilt.append(self.P)
        }

        // Backward pass
        var xSmooth = xFilt
        var PSmooth = PFilt

        for t in (0..<(nObs - 1)).reversed() {
            let PpNextInv = try matInverseFlat(PPred[t + 1], n: n)
            let PFT = matMulFlat(PFilt[t], rA: n, cA: n, FT, rB: n, cB: n)
            let C = matMulFlat(PFT, rA: n, cA: n, PpNextInv, rB: n, cB: n)
            let CT = transposeFlat(C, rows: n, cols: n)

            let dx = vecSubFlat(xSmooth[t + 1], xPred[t + 1])
            let Cdx = matVecMulFlat(C, rows: n, cols: n, vec: dx)
            xSmooth[t] = vecAddFlat(xFilt[t], Cdx)

            let dP = vecSubFlat(PSmooth[t + 1], PPred[t + 1])
            let CdP = matMulFlat(C, rA: n, cA: n, dP, rB: n, cB: n)
            let CdPCT = matMulFlat(CdP, rA: n, cA: n, CT, rB: n, cB: n)
            PSmooth[t] = vecAddFlat(PFilt[t], CdPCT)
        }

        return (0..<nObs).map {
            KalmanState(mean: xSmooth[$0], covariance: unflatten(PSmooth[$0], rows: n, cols: n))
        }
    }

    /// Predict one step ahead.
    public func predict() throws -> KalmanState {
        try checkInitialization()
        let n = stateSize
        let FT = transposeFlat(F, rows: n, cols: n)
        let xp = matVecMulFlat(F, rows: n, cols: n, vec: x)
        let FP = matMulFlat(F, rA: n, cA: n, P, rB: n, cB: n)
        let FPFT = matMulFlat(FP, rA: n, cA: n, FT, rB: n, cB: n)
        let Pp = vecAddFlat(FPFT, Q)
        return KalmanState(mean: xp, covariance: unflatten(Pp, rows: n, cols: n))
    }

    // MARK: - Private matrix math helpers (Flat buffer implementations)

    private func checkInitialization() throws {
        guard isInitialized else { throw ForecastError.notFitted }
        guard !F.isEmpty, !H.isEmpty, !Q.isEmpty, !R.isEmpty else {
            throw ForecastError.notFitted
        }
    }

    private func validateMatrixDimensions(_ mat: [[Double]], expectedRows: Int, expectedCols: Int) throws {
        guard mat.count == expectedRows else {
            throw ForecastError.matrixDimensionMismatch(
                expectedRows: expectedRows, expectedCols: expectedCols,
                gotRows: mat.count, gotCols: mat.first?.count ?? 0
            )
        }
        for row in mat {
            guard row.count == expectedCols else {
                throw ForecastError.matrixDimensionMismatch(
                    expectedRows: expectedRows, expectedCols: expectedCols,
                    gotRows: mat.count, gotCols: row.count
                )
            }
        }
    }

    private func flatten(_ matrix: [[Double]]) -> [Double] {
        var flat = [Double]()
        let r = matrix.count
        let c = matrix.first?.count ?? 0
        flat.reserveCapacity(r * c)
        for row in matrix {
            flat.append(contentsOf: row)
        }
        return flat
    }

    private func unflatten(_ flat: [Double], rows: Int, cols: Int) -> [[Double]] {
        var result = [[Double]]()
        result.reserveCapacity(rows)
        for r in 0..<rows {
            let start = r * cols
            result.append(Array(flat[start..<(start + cols)]))
        }
        return result
    }

    private func matMulFlat(_ A: [Double], rA: Int, cA: Int, _ B: [Double], rB: Int, cB: Int) -> [Double] {
        var C = [Double](repeating: 0.0, count: rA * cB)
        forecast_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rA, cB, cA,
            1.0, A, cA,
            B, cB,
            0.0, &C, cB
        )
        return C
    }

    private func matVecMulFlat(_ A: [Double], rows: Int, cols: Int, vec: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: rows)
        forecast_dgemv(
            CblasRowMajor, CblasNoTrans,
            rows, cols,
            1.0, A, cols,
            vec, 1,
            0.0, &result, 1
        )
        return result
    }

    private func transposeFlat(_ A: [Double], rows: Int, cols: Int) -> [Double] {
        var result = [Double](repeating: 0.0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                result[c * rows + r] = A[r * cols + c]
            }
        }
        return result
    }

    private func vecAddFlat(_ a: [Double], _ b: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: a.count)
        vDSP.add(a, b, result: &result)
        return result
    }

    private func vecSubFlat(_ a: [Double], _ b: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: a.count)
        vDSP.subtract(a, b, result: &result)
        return result
    }

    private func identityFlat(_ size: Int) -> [Double] {
        var mat = [Double](repeating: 0.0, count: size * size)
        for i in 0..<size {
            mat[i * size + i] = 1.0
        }
        return mat
    }

    private func matInverseFlat(_ A: [Double], n: Int) throws -> [Double] {
        var AColMajor = [Double](repeating: 0.0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                AColMajor[c * n + r] = A[r * n + c]
            }
        }

        var ipiv = [LAPACKInteger](repeating: 0, count: n)
        var identity = [Double](repeating: 0.0, count: n * n)
        for i in 0..<n {
            identity[i * n + i] = 1.0
        }

        var dimN = LAPACKInteger(n)
        var dimN2 = dimN
        var lda = dimN
        var ldb = dimN
        var info = LAPACKInteger(0)
        dgesv_wrapper(&dimN, &dimN2, &AColMajor, &lda, &ipiv, &identity, &ldb, &info)
        guard info == 0 else {
            throw ForecastError.singularMatrix
        }

        var result = [Double](repeating: 0.0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                result[r * n + c] = identity[c * n + r]
            }
        }
        return result
    }
}

extension KalmanFilter {
    /// Pre-configured 1D constant-velocity model.
    public static func oneDimensional(
        processNoise: Double,
        measurementNoise: Double
    ) async throws -> KalmanFilter {
        let kf = try KalmanFilter(stateSize: 2, observationSize: 1)

        try await kf.setTransitionMatrix([
            [1.0, 1.0],
            [0.0, 1.0]
        ])

        try await kf.setObservationMatrix([
            [1.0, 0.0]
        ])

        try await kf.setProcessNoise([
            [processNoise, 0.0],
            [0.0, processNoise]
        ])

        try await kf.setMeasurementNoise([
            [measurementNoise]
        ])

        try await kf.setInitialState(
            mean: [0.0, 0.0],
            covariance: [
                [10.0, 0.0],
                [0.0, 10.0]
            ]
        )

        return kf
    }
}
