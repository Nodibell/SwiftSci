import Testing
import SwiftML
import SwiftPreprocessing

@Suite("Training-only prepared imputation and regression")
struct ImputedRegressionWorkflowTests {
    private func training() throws -> PreparedNumericBatch {
        var batch = try PreparedNumericBatch(columnNames: ["a","b","c"], columns: (0..<3).map { c in
            let half = (0..<64).map { r -> Double in
                c > 0 && r % 11 == 0 ? .nan : Double((r * (c * 2 + 3)) % 17 - 8)
            }
            return half + half.map { -$0 }
        })
        try batch.updateColumn(at: 1, rows: [0,64], values: [nil,nil])
        return batch
    }

    private func truth(_ data: PreparedNumericBatch) -> [Double] {
        (0..<data.rowCount).map { r in
            (0..<3).reduce(5.0) { sum, c in
                let x = data[r,c] ?? .nan
                return sum + (x.isNaN ? 0 : x) * Double(c + 1)
            }
        }
    }

    @Test func retainedAndConsumedPathsFitOnceAndPredictHeldOutObservations() async throws {
        var reference: [Double]?
        for owned in [false,true] {
            let rawTraining = try training()
            let trainingRows = rawTraining.rowValues().map { $0.map(\.bitPattern) }
            var preprocessing = Pipeline(steps: [Imputer(), MinMaxScaler(range: (-2,3))])
            try preprocessing.fit(rawTraining)
            let cleanTraining = try preprocessing.transform(rawTraining)
            let model = LinearRegression(device: .cpu)
            let regression = RegressionPipeline(estimator: model)
            try await regression.fit(consuming: consume cleanTraining, targets: truth(rawTraining))
            let parameters = await model.getWeightsAndBias()
            var heldOut = try PreparedNumericBatch(columnNames: ["a","b","c"], columns: [
                (0..<129).map { Double($0 % 13 + 20) },
                (0..<129).map { $0 % 7 == 0 ? .nan : Double($0 % 17 - 30) },
                (0..<129).map { $0 % 11 == 0 ? .nan : Double($0 % 19 + 40) }
            ])
            try heldOut.updateColumn(at: 2, rows: [3,17,128], values: [nil,nil,nil])
            let expected = truth(heldOut)
            let original = heldOut.rowValues().map { $0.map(\.bitPattern) }
            let retained: PreparedNumericBatch?
            let cleaned: PreparedNumericBatch
            if owned {
                retained = nil
                cleaned = try preprocessing.transform(consuming: consume heldOut)
            } else {
                retained = heldOut
                cleaned = try preprocessing.transform(heldOut)
            }
            let predicted = try await regression.predict(consuming: consume cleaned)
            #expect(zip(predicted,expected).allSatisfy { abs($0 - $1) < 1e-10 })
            if let reference { #expect(predicted == reference) } else { reference = predicted }
            if let retained {
                #expect(retained.rowValues().map { $0.map(\.bitPattern) } == original)
                #expect(retained[3,2] == nil)
            }
            #expect(rawTraining.rowValues().map { $0.map(\.bitPattern) } == trainingRows)
            let afterward = await model.getWeightsAndBias()
            #expect(parameters.weights == afterward.weights && parameters.bias == afterward.bias)
            // A later request still uses the training statistics and fitted model.
            let again = try preprocessing.transform(rawTraining)
            let trainingPrediction = try await regression.predict(consuming: consume again)
            #expect(zip(trainingPrediction,truth(rawTraining)).allSatisfy { abs($0 - $1) < 1e-10 })
        }
    }

    @Test func failedCleaningReleasesAdmissionAndFiniteBoundaryRemainsEnforced() async throws {
        let budget = try MemoryBudget(limit: 4096)
        let estimate = try MemoryEstimate(capacities: [4096])
        let raw = try training()
        var invalid = Pipeline(steps: [Imputer(strategy: .constant(.nan))])
        try invalid.fit(raw)
        let bad = invalid
        let regression = RegressionPipeline(estimator: LinearRegression(device: .cpu))
        await #expect(throws: (any Error).self) {
            try await budget.withReservation(estimate) {
                let clean = try bad.transform(raw)
                try await regression.fit(consuming: consume clean, targets: truth(raw))
            }
        }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
        await #expect(throws: (any Error).self) {
            try await regression.fit(features: raw, targets: truth(raw))
        }
        var valid = Pipeline(steps: [Imputer(), MinMaxScaler()])
        try valid.fit(raw)
        let good = valid
        try await budget.withReservation(estimate) {
            let clean = try good.transform(raw)
            try await regression.fit(consuming: consume clean, targets: truth(raw))
        }
        #expect(await budget.reservedBytes == 0)
        #expect(await budget.queuedCount == 0)
    }
}
