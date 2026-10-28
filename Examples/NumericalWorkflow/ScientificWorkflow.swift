import ArgumentParser
import Foundation
import SwiftDataFrame
import SwiftWorkflow
import SwiftOptimize

@main
struct ScientificWorkflow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ScientificWorkflow",
        abstract: "Train, save and reuse a complete fitted numerical workflow.",
        subcommands: [Train.self, Predict.self], defaultSubcommand: Train.self)

    static let featureNames = ["signal", "companion", "constant"]

    static func readCSV(_ url: URL) async throws -> DataFrame {
        var options = CSVReadOptions()
        options.doubleStorage = .compact
        options.nullValues.remove("NaN")
        options.nullValues.remove("nan")
        return try await DataFrame(csv: url, options: options)
    }

    static func protect(_ inputs: [URL], from outputs: [URL]) throws {
        let protected = inputs.map { $0.resolvingSymlinksInPath().standardizedFileURL }
        guard !outputs.contains(where: { protected.contains($0.resolvingSymlinksInPath().standardizedFileURL) }) else {
            throw ValidationError("Choose an output path that does not overwrite an input or model file.")
        }
    }

    static func predictionFrame(_ result: PCAClassificationWorkflow.Predictions,
                                actual: [Double]? = nil) throws -> DataFrame {
        var columns: [any AnyColumn] = [
            TypedColumn<Int64>(name: "source_row", values: result.sourceRowIndices.map { Int64($0) })
        ]
        if let actual {
            columns.append(TypedColumn<Int64>(name: "actual", values: actual.map { Int64($0) }))
        }
        columns.append(TypedColumn<Int64>(name: "predicted", values: result.labels.map { Int64($0) }))
        columns.append(TypedColumn<Double>(name: "probability_class_0", values: result.probabilities.map { $0[0] }))
        columns.append(TypedColumn<Double>(name: "probability_class_1", values: result.probabilities.map { $0[1] }))
        return try DataFrame(columns: columns)
    }

    struct Train: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Fit training rows, evaluate held-out rows and save model.json.")
        @Option(help: "CSV with signal, companion, constant and binary class; defaults to the bundled sample.")
        var input: String?
        @Option(help: "Directory for model.json, predictions.csv and report.json; those files are replaced.")
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
            let modelURL = directory.appendingPathComponent("model.json")
            let csvURL = directory.appendingPathComponent("predictions.csv")
            let reportURL = directory.appendingPathComponent("report.json")
            try ScientificWorkflow.protect([inputURL], from: [modelURL, csvURL, reportURL])
            let frame = try await ScientificWorkflow.readCSV(inputURL)
            let source = try frame.prepareNumericBatch(ScientificWorkflow.featureNames + ["class"])
            guard source.rowCount >= 6 else {
                throw ValidationError("This example needs at least six data rows for its training/test split.")
            }
            let supervised = try PreparedSupervisedBatch(source, targetColumn: "class")
            let count = source.rowCount * 2 / 3
            let trainingIDs = Array(0..<count)
            let queryIDs = Array((count..<source.rowCount).reversed())
            let training = try supervised.selectingRows(trainingIDs)
            let query = try supervised.selectingRows(queryIDs)
            guard query.targets.allSatisfy({ $0 == 0 || $0 == 1 }) else {
                throw ValidationError("Every class value must be 0 or 1, without missing values.")
            }
            let workflow = try await PCAClassificationWorkflow.train(training: training)
            let result = try await workflow.predict(query.features)
            let accuracy = Metrics.accuracy(yTrue: query.targets.map { Int($0) }, yPred: result.labels)
            let report: [String: Any] = ["schemaVersion": 2, "trainingSourceRows": trainingIDs,
                "querySourceRows": result.sourceRowIndices, "accuracy": accuracy]
            let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try workflow.save(to: modelURL)
            try await ScientificWorkflow.predictionFrame(result, actual: query.targets).writeCSV(to: csvURL)
            try json.write(to: reportURL, options: .atomic)
            print("Training rows: \(trainingIDs.count); held-out rows: \(queryIDs.count)")
            print("Held-out accuracy: \(accuracy)")
            print("Saved workflow: \(modelURL.path)")
            print("Predictions: \(csvURL.path)")
        }
    }

    struct Predict: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Load a fitted workflow and predict unlabeled CSV rows without retraining.")
        @Option(help: "Saved model.json workflow artifact.") var model: String
        @Option(help: "CSV containing the saved feature names; no class column is needed.") var input: String
        @Option(help: "Prediction CSV file to replace.") var output = ".build/predictions.csv"

        func run() async throws {
            let modelURL = URL(fileURLWithPath: model)
            let inputURL = URL(fileURLWithPath: input)
            let outputURL = URL(fileURLWithPath: output)
            try ScientificWorkflow.protect([modelURL, inputURL], from: [outputURL])
            let workflow = try PCAClassificationWorkflow.load(from: modelURL)
            let frame = try await ScientificWorkflow.readCSV(inputURL)
            let features = try frame.prepareNumericBatch(workflow.featureNames)
            let result = try await workflow.predict(features)
            let outputFrame = try ScientificWorkflow.predictionFrame(result)
            try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try await outputFrame.writeCSV(to: outputURL)
            print("Predicted rows: \(result.labels.count)")
            print("Predictions: \(outputURL.path)")
        }
    }
}
