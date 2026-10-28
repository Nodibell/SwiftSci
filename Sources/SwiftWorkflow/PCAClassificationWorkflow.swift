#if os(macOS)
import Foundation
import SwiftDataFrame
import SwiftPreprocessing
import SwiftCluster
import SwiftML

/// A fitted CPU mean-imputation, standardization, PCA and binary logistic workflow.
/// Published instances expose no mutable stages. Saving never refits or reads prediction data.
public struct PCAClassificationWorkflow: Sendable {
    /// Required input feature names in fitted order.
    public var featureNames: [String] { artifact.featureNames }

    /// Prediction arrays share the original source-row order of the supplied batch.
    public struct Predictions: Sendable {
        public let sourceRowIndices: [Int]
        public let labels: [Int]
        /// Each pair contains probabilities for class 0 and class 1, respectively.
        public let probabilities: [[Double]]
    }

    private struct Artifact: Codable, Sendable {
        let version: Int
        let featureNames: [String]
        let imputationStatistics: [Double]
        let scalingMean: [Double]
        let scalingStd: [Double]
        let pcaMean: [Double]
        let pcaComponents: [[Double]]
        let pcaVariance: [Double]
        let pcaTotalVariance: Double
        let weights: [Double]
        let bias: Double
    }

    private let artifact: Artifact
    private let imputer: Imputer
    private let scaler: StandardScaler
    private let pca: PCA
    private let classifier: LogisticRegression

    private init(_ artifact: Artifact) throws {
        let width = artifact.featureNames.count
        guard width > 0, Set(artifact.featureNames).count == width,
              artifact.featureNames.allSatisfy({ !$0.isEmpty }),
              artifact.imputationStatistics.count == width,
              artifact.scalingMean.count == width, artifact.scalingStd.count == width,
              artifact.pcaMean.count == width,
              artifact.weights.count == artifact.pcaComponents.count,
              artifact.weights.allSatisfy(\.isFinite), artifact.bias.isFinite else {
            throw SwiftMLError.invalidParameter("Workflow artifact has incompatible feature or model dimensions.")
        }
        imputer = try Imputer(statistics: artifact.imputationStatistics)
        scaler = try StandardScaler(mean: artifact.scalingMean, std: artifact.scalingStd)
        let parameters = try PCA.Parameters(mean: artifact.pcaMean,
            components: artifact.pcaComponents, explainedVariance: artifact.pcaVariance,
            totalVariance: artifact.pcaTotalVariance, featureNames: artifact.featureNames)
        pca = PCA(parameters: parameters)
        classifier = LogisticRegression(weights: artifact.weights, bias: artifact.bias, device: .cpu)
        self.artifact = artifact
    }

    /// Fits all learned stages only on the caller's selected training rows.
    /// Features may contain missing values or NaNs, but not infinity. Targets must include both 0 and 1.
    public static func train(training: PreparedSupervisedBatch, components: Int = 2,
                             learningRate: Float = 0.125, epochs: Int = 64) async throws -> Self {
        let features = training.features
        guard Set(training.targets) == Set([0.0, 1.0]),
              features.columnCount > 0, Set(features.columnNames).count == features.columnCount,
              features.columnNames.allSatisfy({ !$0.isEmpty }) else {
            throw SwiftMLError.invalidParameter("Training requires unique feature names and both binary classes 0 and 1.")
        }
        try requireNoInfinity(features)
        var imputer = Imputer(strategy: .mean)
        try imputer.fit(features)
        let imputed = try imputer.transform(features)
        var scaler = StandardScaler()
        try scaler.fit(imputed)
        let scaled = try scaler.transform(imputed)
        let pca = try PCA(nComponents: components, svdSolver: .full, device: .cpu)
        let scores = try await pca.fitTransform(scaled)
        let classifier = LogisticRegression(device: .cpu)
        try await classifier.fit(features: scores, targets: training.targets,
                                 learningRate: learningRate, epochs: epochs)
        let parameters = try await pca.parameters()
        let (weights, bias) = await classifier.getWeightsAndBias()
        guard let replacements = imputer.statistics, let mean = scaler.mean, let std = scaler.std,
              let weights, let bias else { throw SwiftMLError.modelNotFitted }
        return try Self(Artifact(version: 1, featureNames: features.columnNames,
            imputationStatistics: replacements, scalingMean: mean, scalingStd: std,
            pcaMean: parameters.mean, pcaComponents: parameters.components,
            pcaVariance: parameters.explainedVariance, pcaTotalVariance: parameters.totalVariance,
            weights: weights, bias: bias))
    }

    /// Applies fitted stages without learning from the supplied rows.
    /// Prepared names must match exactly; dataframe callers can select `featureNames` in that order.
    public func predict(_ features: PreparedNumericBatch) async throws -> Predictions {
        guard features.columnNames == featureNames else {
            throw SwiftMLError.invalidParameter("Prediction feature names and order must match the saved workflow.")
        }
        try Self.requireNoInfinity(features)
        if features.rowCount == 0 {
            return Predictions(sourceRowIndices: [], labels: [], probabilities: [])
        }
        let imputed = try imputer.transform(features)
        let scaled = try scaler.transform(imputed)
        let scores = try await pca.transform(scaled)
        let probabilities = try await classifier.predictProbability(features: scores)
        return Predictions(sourceRowIndices: scores.originalRowIndices,
            labels: probabilities.map { $0[1] > 0.5 ? 1 : 0 }, probabilities: probabilities)
    }

    /// Atomically writes learned parameters and schema, without samples or targets.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(artifact)
        try data.write(to: url, options: .atomic)
    }

    /// Loads a version-1 artifact and validates all stage parameters before publishing a workflow.
    public static func load(from url: URL) throws -> Self {
        struct Header: Decodable { let version: Int }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let header = try decoder.decode(Header.self, from: data)
        guard header.version == 1 else {
            throw SwiftMLError.unsupportedOperation("Unsupported workflow artifact version \(header.version).")
        }
        return try Self(decoder.decode(Artifact.self, from: data))
    }

    private static func requireNoInfinity(_ features: PreparedNumericBatch) throws {
        for column in 0..<features.columnCount {
            guard !features.columnValues(at: column).contains(where: { $0?.isInfinite == true }) else {
                throw SwiftMLError.invalidParameter("Workflow features must not contain infinity.")
            }
        }
    }
}
#endif
