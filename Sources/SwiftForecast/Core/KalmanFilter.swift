import Foundation
import Accelerate



public actor KalmanFilter {
    private let stateSize: Int
    private let observationSize: Int
    
    private var F: [[Double]] = [] // State transition matrix (n x n)
    private var H: [[Double]] = [] // Observation matrix (m x n)
    private var Q: [[Double]] = [] // Process noise covariance (n x n)
    private var R: [[Double]] = [] // Measurement noise covariance (m x m)
    
    private var x: [Double] = []   // State estimate mean (n)
    private var P: [[Double]] = [] // State estimate covariance (n x n)
    private var isInitialized = false
    
    public init(stateSize: Int, observationSize: Int) throws {
        guard stateSize > 0 else { throw ForecastError.invalidAROrder(stateSize) }
        guard observationSize > 0 else { throw ForecastError.invalidMAOrder(observationSize) }
        self.stateSize = stateSize
        self.observationSize = observationSize
    }
    
    public func setTransitionMatrix(_ F: [[Double]]) throws {
        try validateMatrixDimensions(F, expectedRows: stateSize, expectedCols: stateSize)
        self.F = F
    }
    
    public func setObservationMatrix(_ H: [[Double]]) throws {
        try validateMatrixDimensions(H, expectedRows: observationSize, expectedCols: stateSize)
        self.H = H
    }
    
    public func setProcessNoise(_ Q: [[Double]]) throws {
        try validateMatrixDimensions(Q, expectedRows: stateSize, expectedCols: stateSize)
        self.Q = Q
    }
    
    public func setMeasurementNoise(_ R: [[Double]]) throws {
        try validateMatrixDimensions(R, expectedRows: observationSize, expectedCols: observationSize)
        self.R = R
    }
    
    public func setInitialState(mean: [Double], covariance: [[Double]]) throws {
        guard mean.count == stateSize else {
            throw ForecastError.matrixDimensionMismatch(expectedRows: stateSize, expectedCols: 1, gotRows: mean.count, gotCols: 1)
        }
        try validateMatrixDimensions(covariance, expectedRows: stateSize, expectedCols: stateSize)
        self.x = mean
        self.P = covariance
        self.isInitialized = true
    }
    
    /// Run Kalman Filter forward over observations.
    public func filter(observations: [[Double]]) throws -> [KalmanState] {
        try checkInitialization()
        
        var states: [KalmanState] = []
        
        for z in observations {
            guard z.count == observationSize else {
                throw ForecastError.matrixDimensionMismatch(expectedRows: observationSize, expectedCols: 1, gotRows: z.count, gotCols: 1)
            }
            
            // 1. Predict
            let xPred = matMul(F, x)
            let PPred = matAdd(matMul(matMul(F, P), transpose(F)), Q)
            
            // 2. Update
            // innovation residual y = z - H * x_pred
            let y = vecSub(z, matMul(H, xPred))
            
            // innovation covariance S = H * P_pred * H^T + R
            let S = matAdd(matMul(matMul(H, PPred), transpose(H)), R)
            
            // S_inv
            let SInv = try matInverse(S)
            
            // Kalman gain K = P_pred * H^T * S_inv
            let K = matMul(matMul(PPred, transpose(H)), SInv)
            
            // updated state estimate: x = x_pred + K * y
            self.x = vecAdd(xPred, matMul(K, y))
            
            // updated covariance (Joseph form): P = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
            let I = identityMatrix(stateSize)
            let KH = matMul(K, H)
            let IMinusKH = matSub(I, KH)
            let IMinusKHT = transpose(IMinusKH)
            let term1 = matMul(matMul(IMinusKH, PPred), IMinusKHT)
            let term2 = matMul(matMul(K, R), transpose(K))
            self.P = matAdd(term1, term2)
            
            states.append(KalmanState(mean: self.x, covariance: self.P))
        }
        
        return states
    }
    
    /// RTS (Rauch-Tung-Striebel) smoother.
    public func smooth(observations: [[Double]]) throws -> [KalmanState] {
        try checkInitialization()
        
        let nObs = observations.count
        guard nObs > 0 else { return [] }
        
        // Run forward pass and save state steps
        var xFilt: [[Double]] = []
        var PFilt: [[[Double]]] = []
        var xPred: [[Double]] = []
        var PPred: [[[Double]]] = []
        
        for z in observations {
            // Predict
            let xp = matMul(F, x)
            let Pp = matAdd(matMul(matMul(F, P), transpose(F)), Q)
            xPred.append(xp)
            PPred.append(Pp)
            
            // Update
            let y = vecSub(z, matMul(H, xp))
            let S = matAdd(matMul(matMul(H, Pp), transpose(H)), R)
            let SInv = try matInverse(S)
            let K = matMul(matMul(Pp, transpose(H)), SInv)
            
            self.x = vecAdd(xp, matMul(K, y))
            let I = identityMatrix(stateSize)
            let KH = matMul(K, H)
            let IMinusKH = matSub(I, KH)
            let IMinusKHT = transpose(IMinusKH)
            let term1 = matMul(matMul(IMinusKH, Pp), IMinusKHT)
            let term2 = matMul(matMul(K, R), transpose(K))
            self.P = matAdd(term1, term2)
            
            xFilt.append(self.x)
            PFilt.append(self.P)
        }
        
        // Backward pass
        var xSmooth = xFilt
        var PSmooth = PFilt
        
        for t in (0..<(nObs - 1)).reversed() {
            let PpNextInv = try matInverse(PPred[t+1])
            let C = matMul(matMul(PFilt[t], transpose(F)), PpNextInv)
            
            let dx = vecSub(xSmooth[t+1], xPred[t+1])
            xSmooth[t] = vecAdd(xFilt[t], matMul(C, dx))
            
            let dP = matSub(PSmooth[t+1], PPred[t+1])
            PSmooth[t] = matAdd(PFilt[t], matMul(matMul(C, dP), transpose(C)))
        }
        
        return (0..<nObs).map { KalmanState(mean: xSmooth[$0], covariance: PSmooth[$0]) }
    }
    
    /// Predict one step ahead.
    public func predict() throws -> KalmanState {
        try checkInitialization()
        let xp = matMul(F, x)
        let Pp = matAdd(matMul(matMul(F, P), transpose(F)), Q)
        return KalmanState(mean: xp, covariance: Pp)
    }
    
    // MARK: - Private matrix math helpers
    
    private func checkInitialization() throws {
        guard isInitialized else { throw ForecastError.notFitted }
        guard !F.isEmpty, !H.isEmpty, !Q.isEmpty, !R.isEmpty else {
            throw ForecastError.notFitted
        }
    }
    
    private func validateMatrixDimensions(_ mat: [[Double]], expectedRows: Int, expectedCols: Int) throws {
        guard mat.count == expectedRows else {
            throw ForecastError.matrixDimensionMismatch(expectedRows: expectedRows, expectedCols: expectedCols, gotRows: mat.count, gotCols: mat.first?.count ?? 0)
        }
        for row in mat {
            guard row.count == expectedCols else {
                throw ForecastError.matrixDimensionMismatch(expectedRows: expectedRows, expectedCols: expectedCols, gotRows: mat.count, gotCols: row.count)
            }
        }
    }
    
    private func matMul(_ A: [[Double]], _ B: [[Double]]) -> [[Double]] {
        let rA = A.count
        let cA = A[0].count
        let cB = B[0].count
        
        var result = [[Double]](repeating: [Double](repeating: 0.0, count: cB), count: rA)
        
        // Flatten inputs for BLAS
        let flatA = A.flatMap { $0 }
        let flatB = B.flatMap { $0 }
        var flatC = [Double](repeating: 0.0, count: rA * cB)
        
        forecast_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rA, cB, cA,
            1.0, flatA, cA,
            flatB, cB,
            0.0, &flatC, cB
        )
        
        for r in 0..<rA {
            for c in 0..<cB {
                result[r][c] = flatC[r * cB + c]
            }
        }
        return result
    }
    
    private func matMul(_ A: [[Double]], _ x: [Double]) -> [Double] {
        let rA = A.count
        let cA = A[0].count
        var result = [Double](repeating: 0.0, count: rA)
        let flatA = A.flatMap { $0 }
        
        forecast_dgemv(
            CblasRowMajor, CblasNoTrans,
            rA, cA,
            1.0, flatA, cA,
            x, 1,
            0.0, &result, 1
        )
        return result
    }
    
    private func transpose(_ A: [[Double]]) -> [[Double]] {
        let r = A.count
        let c = A[0].count
        var result = [[Double]](repeating: [Double](repeating: 0.0, count: r), count: c)
        for i in 0..<r {
            for j in 0..<c {
                result[j][i] = A[i][j]
            }
        }
        return result
    }
    
    private func matAdd(_ A: [[Double]], _ B: [[Double]]) -> [[Double]] {
        let r = A.count
        let c = A[0].count
        var result = [[Double]](repeating: [Double](repeating: 0.0, count: c), count: r)
        for i in 0..<r {
            vDSP.add(A[i], B[i], result: &result[i])
        }
        return result
    }
    
    private func matSub(_ A: [[Double]], _ B: [[Double]]) -> [[Double]] {
        let r = A.count
        let c = A[0].count
        var result = [[Double]](repeating: [Double](repeating: 0.0, count: c), count: r)
        for i in 0..<r {
            vDSP.subtract(A[i], B[i], result: &result[i])
        }
        return result
    }
    
    private func vecAdd(_ a: [Double], _ b: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: a.count)
        vDSP.add(a, b, result: &result)
        return result
    }
    
    private func vecSub(_ a: [Double], _ b: [Double]) -> [Double] {
        var result = [Double](repeating: 0.0, count: a.count)
        vDSP.subtract(a, b, result: &result) // computes a - b
        return result
    }
    
    private func identityMatrix(_ size: Int) -> [[Double]] {
        var mat = [[Double]](repeating: [Double](repeating: 0.0, count: size), count: size)
        for i in 0..<size {
            mat[i][i] = 1.0
        }
        return mat
    }
    
    private func matInverse(_ A: [[Double]]) throws -> [[Double]] {
        let n = A.count
        
        var AColMajor = [Double](repeating: 0.0, count: n * n)
        for r in 0..<n {
            for c in 0..<n {
                AColMajor[c * n + r] = A[r][c]
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
        
        // Reconstruct result matrix from col-major identity output
        var result = [[Double]](repeating: [Double](repeating: 0.0, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                // dgesv output in identity is column-major
                result[i][j] = identity[j * n + i]
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
        
        // state = [position, velocity]
        // transition: position_t = position_{t-1} + velocity_{t-1}, velocity_t = velocity_{t-1}
        try await kf.setTransitionMatrix([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        
        // observation: z = position
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
