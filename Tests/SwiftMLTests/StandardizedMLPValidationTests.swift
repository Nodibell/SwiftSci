import Testing
import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

@Suite("Public standardized MLP validation split")
struct StandardizedMLPValidationTests {
    @Test func validationUsesTrainingStatisticsAndSameEarlyStopping() async throws {
        let training = try publicMLPTraining()
        let rows: [[Double]] = matrixMLPFeatures(7, width: 7).map { row -> [Double] in
            row.map { value -> Double in value + 100 }
        }
        let targets: [Double] = rows.map { $0[0] - $0[1] }
        let columns: [[Double]] = (0..<7).map { column -> [Double] in
            rows.map { row -> Double in row[column] }
        } + [targets]
        let raw = try PreparedNumericBatch(columnNames: training.batch.columnNames + ["target"], columns: columns)
        let validation = try PreparedSupervisedBatch(raw, targetColumn: "target")
        var pipeline = Pipeline(steps: [Imputer(), StandardScaler()])
        try pipeline.fit(training.batch)
        let cleanTraining = try pipeline.transform(training.batch)
        let cleanValidation = try pipeline.transform(validation.features)
        let model = publicMLPModel(), reference = publicMLPModel()
        let stopping = EarlyStopping(patience: 1)
        let fitted = try await model.fitStandardized(features: training.batch, targets: training.targets,
            validation: validation, earlyStopping: stopping)
        try await reference.fit(features: cleanTraining.rowValues(), targets: training.targets,
            validationFeatures: cleanValidation.rowValues(), validationTargets: targets, earlyStopping: stopping)
        let budget = try MemoryBudget(limit: 16 * 1024 * 1024)
        let input = try await fitted.prepare(MLPPredictionRequest(source: validation.features), budget: budget)
        let actual = try await fitted.predict(input, budget: budget)
        let expected = try await reference.predict(features: cleanValidation)
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        await #expect(throws: SwiftMLError.self) {
            try await model.fitStandardized(features: training.batch, targets: training.targets, earlyStopping: stopping)
        }
        let empty = try validation.selectingRows([])
        await #expect(throws: SwiftMLError.self) {
            try await model.fitStandardized(features: training.batch, targets: training.targets, validation: empty)
        }
    }
}
