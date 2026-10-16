import SwiftML
import SwiftPreprocessing
import SwiftDataFrame

func publicMLPModel() -> MLPRegressor {
    MLPRegressor(hiddenLayerSizes: [9, 5], maxIter: 2, seed: 42, requestedDevice: .cpu)
}

struct PublicMLPReference {
    let preprocessing: Pipeline
    let model: MLPRegressor
    func predict(_ batch: PreparedNumericBatch) async throws -> [Double] {
        try await model.predict(features: preprocessing.transform(batch))
    }
}

func publicMLPReference(_ training: PreparedNumericBatch, targets: [Double]) async throws -> PublicMLPReference {
    var preprocessing = Pipeline(steps: [Imputer(), StandardScaler()])
    try preprocessing.fit(training)
    let clean = try preprocessing.transform(training)
    let model = publicMLPModel()
    try await model.fit(features: clean.rowValues(), targets: targets)
    return PublicMLPReference(preprocessing: preprocessing, model: model)
}

func publicMLPTraining() throws -> (batch: PreparedNumericBatch, targets: [Double]) {
    let rows = matrixMLPFeatures(18, width: 7)
    var batch = try matrixMLPBatch(rows, width: 7)
    try batch.updateColumn(at: 0, rows: [0, 1], values: [nil, .nan])
    try batch.updateColumn(at: 6, rows: Array(0..<18), values: Array(repeating: 3, count: 18))
    let targets: [Double] = rows.map { $0[0] - $0[1] }
    return (batch, targets)
}
