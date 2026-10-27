import ArgumentParser
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftCluster
import SwiftML
import SwiftOptimize

@main
struct ScientificWorkflow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ScientificWorkflow",
        abstract: "Run a CSV-to-predictions SwiftSci numerical workflow."
    )

    @Option(help: "CSV with signal, companion, constant and binary class columns; defaults to the bundled sample.")
    var input: String?

    @Option(help: "Directory for predictions.csv and report.json; those two files are replaced on each run.")
    var output = ".build/numerical-workflow"

    func run() async throws {
        let inputURL: URL
        if let input {
            inputURL = URL(fileURLWithPath: input)
        } else {
            guard let bundled = Bundle.module.url(forResource: "sample", withExtension: "csv", subdirectory: "Data") else {
                throw ValidationError("The bundled sample.csv resource is missing.")
            }
            inputURL = bundled
        }
        let directory = URL(fileURLWithPath: output, isDirectory: true)
        let csvURL = directory.appendingPathComponent("predictions.csv")
        let jsonURL = directory.appendingPathComponent("report.json")
        let resolvedInput = inputURL.resolvingSymlinksInPath().standardizedFileURL
        guard ![csvURL, jsonURL].contains(where: {
            $0.resolvingSymlinksInPath().standardizedFileURL == resolvedInput
        }) else {
            throw ValidationError("Choose an output directory that does not overwrite the input CSV.")
        }

        var options = CSVReadOptions()
        options.doubleStorage = .compact
        // Keep a numeric NaN distinct from a missing CSV cell. Both are imputed below.
        options.nullValues.remove("NaN")
        options.nullValues.remove("nan")
        let frame = try await DataFrame(csv: inputURL, options: options)
        let source = try frame.prepareNumericBatch(["signal", "companion", "constant", "class"])
        guard source.rowCount >= 6 else {
            throw ValidationError("This example needs at least six data rows for its training/test split.")
        }
        let supervised = try PreparedSupervisedBatch(source, targetColumn: "class")
        guard supervised.targets.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw ValidationError("Every class value must be 0 or 1, without missing values.")
        }
        let features = supervised.features
        let rawColumns = (0..<features.columnCount).map { features.columnValues(at: $0) }
        guard !rawColumns.contains(where: { $0.contains(where: { $0?.isInfinite == true }) }) else {
            throw ValidationError("Feature values may be missing or NaN, but must not be infinite.")
        }
        let originalBits = (0..<source.columnCount).map { source.columnValues(at: $0).map { $0?.bitPattern } }

        // Select features and targets together before any learned stage is fitted.
        let trainingCount = source.rowCount * 2 / 3
        let trainingIDs = Array(0..<trainingCount)
        let queryIDs = Array((trainingCount..<source.rowCount).reversed())
        let training = try supervised.selectingRows(trainingIDs)
        let query = try supervised.selectingRows(queryIDs)
        guard Set(training.targets) == Set([0.0, 1.0]) else {
            throw ValidationError("The training portion must contain both class 0 and class 1.")
        }

        var imputer = Imputer(strategy: .mean)
        try imputer.fit(training.features)
        let imputedTraining = try imputer.transform(training.features)
        let imputedQuery = try imputer.transform(query.features)
        var scaler = StandardScaler()
        try scaler.fit(imputedTraining)
        let scaledTraining = try scaler.transform(imputedTraining)
        let scaledQuery = try scaler.transform(imputedQuery)

        // Query rows only transform and predict. The target never enters preprocessing.
        let pca = try PCA(nComponents: 2, svdSolver: .full, device: .cpu)
        let trainingScores = try await pca.fitTransform(scaledTraining)
        let queryScores = try await pca.transform(scaledQuery)
        let classifier = LogisticRegression(device: .cpu)
        try await classifier.fit(features: trainingScores, targets: training.targets,
                                 learningRate: 0.125, epochs: 64)
        let probabilities = try await classifier.predictProbability(features: queryScores)
        let predictions = try await classifier.predict(features: queryScores)
        let actual = query.targets.map { Int($0) }
        let accuracy = Metrics.accuracy(yTrue: actual, yPred: predictions)

        guard let replacements = imputer.statistics, let means = scaler.mean, let deviations = scaler.std,
              let pcaMean = await pca.mean, let components = await pca.components,
              let variance = await pca.explainedVariance,
              let weights = await classifier.getWeights(), let bias = await classifier.getBias() else {
            throw ValidationError("The workflow did not publish its fitted state.")
        }
        let currentBits = (0..<source.columnCount).map { source.columnValues(at: $0).map { $0?.bitPattern } }
        guard originalBits == currentBits, trainingScores.originalRowIndices == trainingIDs,
              queryScores.originalRowIndices == queryIDs, probabilities.count == queryIDs.count,
              predictions.count == queryIDs.count else {
            throw ValidationError("Source snapshot or prediction row alignment changed.")
        }

        // Arrays do not carry row IDs. Associate each prediction with its score batch.
        let resultFrame = try DataFrame(columns: [
            TypedColumn<Int64>(name: "source_row", values: queryScores.originalRowIndices.map { Int64($0) }),
            TypedColumn<Int64>(name: "actual", values: actual.map { Int64($0) }),
            TypedColumn<Int64>(name: "predicted", values: predictions.map { Int64($0) }),
            TypedColumn<Double>(name: "probability_class_0", values: probabilities.map { $0[0] }),
            TypedColumn<Double>(name: "probability_class_1", values: probabilities.map { $0[1] })
        ])
        let rawRows: [[Any]] = (0..<source.rowCount).map { row in
            rawColumns.map { column -> Any in
                guard let value = column[row], value.isFinite else { return NSNull() }
                return value
            }
        }
        let report: [String: Any] = [
            "schemaVersion": 1, "trainingSourceRows": trainingIDs, "querySourceRows": queryIDs,
            "featureNames": features.columnNames, "scoreNames": queryScores.columnNames,
            "rawRows": rawRows, "sourceTargets": supervised.targets,
            "nullFeatureCount": rawColumns.flatMap { $0 }.filter { $0 == nil }.count,
            "nanFeatureCount": rawColumns.flatMap { $0 }.filter { $0?.isNaN == true }.count,
            "trainingTargets": training.targets, "queryTargets": query.targets,
            "imputationStatistics": replacements, "scalingMean": means, "scalingStd": deviations,
            "scaledTraining": scaledTraining.rowValues(), "scaledQuery": scaledQuery.rowValues(),
            "pcaMean": pcaMean, "pcaComponents": components, "pcaVariance": variance,
            "trainingScores": trainingScores.rowValues(), "queryScores": queryScores.rowValues(),
            "weights": weights, "bias": bias, "probabilities": probabilities, "predictions": predictions,
            "accuracy": accuracy, "learningRate": 0.125, "epochs": 64, "sourceSnapshotPreserved": true
        ]
        let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try await resultFrame.writeCSV(to: csvURL)
        try json.write(to: jsonURL, options: .atomic)
        print("Training rows: \(trainingIDs.count); held-out rows: \(queryIDs.count)")
        print("Held-out accuracy: \(accuracy)")
        print("Predictions: \(csvURL.path)")
        print("Stage report: \(jsonURL.path)")
    }
}
