import Foundation
import Testing
@testable import SwiftML

@Suite("Batched MLP prediction")
struct MLPBatchedPredictionTests {
    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func regressionMatchesScalar(_ activation: ActivationFunction) async throws {
        let training = batchFeatures(18)
        for hidden in [[], [9], [11, 5], [128, 64]] {
            let model = MLPRegressor(hiddenLayerSizes: hidden, activation: activation,
                                     maxIter: 3, seed: 42, requestedDevice: .cpu)
            let targets: [Double] = training.map { 2 * $0[0] - $0[1] + 0.5 }
            try await model.fit(features: training, targets: targets)
            let layers = try #require(await model.trainedLayers)
            for rows in [0, 1, 31, 32, 33, 129, 257] {
                let input = batchFeatures(rows)
                let actual = try await model.predict(features: input)
                let expected: [Double] = input.map { scalarForward($0, layers, activation)[0] }
                expectClose(actual, expected)
                _ = try await model.predict(features: batchFeatures(65, offset: 101))
                expectClose(actual, expected)
            }
            #expect(await model.trainedLayers?.map { $0.W } == layers.map { $0.W })
        }
    }

    @Test(arguments: [ActivationFunction.relu, .sigmoid, .tanh])
    func classificationMatchesScalar(_ activation: ActivationFunction) async throws {
        let training = batchFeatures(18)
        for classes in [[-3.0, 7.0], [-3.0, 7.0, 12.0]] {
            let model = MLPClassifier(hiddenLayerSizes: [11, 5], activation: activation,
                                      maxIter: 3, seed: 42, requestedDevice: .cpu)
            let targets: [Double] = training.indices.map { classes[$0 % classes.count] }
            try await model.fit(features: training, targets: targets)
            let layers = try #require(await model.trainedLayers)
            for rows in [0, 1, 31, 32, 33, 129] {
                let input = batchFeatures(rows)
                let actual = try await model.predictProbability(features: input)
                #expect(actual.count == rows)
                var expectedLabels: [Int] = []
                for (row, result) in zip(input, actual) {
                    let logits = scalarForward(row, layers, activation)
                    let probabilities: [Double]
                    if classes.count == 2 {
                        let p = sigmoid(logits[0])
                        probabilities = [1 - p, p]
                        expectedLabels.append(Int(p >= 0.5 ? classes[1] : classes[0]))
                    } else {
                        let exps = logits.map { exp($0 - logits.max()!) }
                        let total = exps.reduce(0, +)
                        probabilities = exps.map { $0 / total }
                        let best = probabilities.enumerated().max { $0.element < $1.element }!.offset
                        expectedLabels.append(Int(classes[best]))
                    }
                    expectClose(result, probabilities)
                }
                #expect(try await model.predict(features: input) == expectedLabels)
            }
        }
    }

    @Test func boundedFallbackAndMalformedInput() throws {
        let small = LayerWeights(W: [2, -1], b: [0.5], inDim: 2, outDim: 1)
        let short = try mlpBatchedPrediction(features: [[3, 4]], layers: [small],
                                             activate: { $0 }, finish: { $0[0] })
        #expect(short == nil)
        let wide = LayerWeights(W: Array(repeating: 1, count: 8193), b: [0], inDim: 8193, outDim: 1)
        let fallback = try mlpBatchedPrediction(features: Array(repeating: wide.W, count: 32),
            layers: [wide], activate: { $0 }, finish: { $0[0] })
        #expect(fallback == nil)
        let maxBatched = LayerWeights(W: Array(repeating: 1, count: 8192), b: [0.5], inDim: 8192, outDim: 1)
        let maxBatchedResult = try mlpBatchedPrediction(features: Array(repeating: Array(repeating: 0.5, count: 8192), count: 5),
            layers: [maxBatched], activate: { $0 }, finish: { $0[0] })
        #expect(maxBatchedResult == Array(repeating: 4096.5, count: 5))
        #expect(throws: SwiftMLError.self) {
            _ = try mlpBatchedPrediction(features: Array(repeating: [1.0], count: 32),
                layers: [small], activate: { $0 }, finish: { $0[0] })
        }
        let optionalTail: [Double]? = try mlpBatchedPrediction(
            features: Array(repeating: [3, 4], count: 65),
            layers: [small], activate: { $0 }, finish: { $0[0] })
        let tail = try #require(optionalTail)
        #expect(tail == Array(repeating: 2.5, count: 65))
    }

    @Test(arguments: [768, 2048, 4096])
    func wideLayersUseBatchedParityWithinScratchBudget(_ width: Int) throws {
        let batchRows = mlpEffectivePredictionBatchRows(width: width)
        #expect(batchRows == min(32, 16_384 / width))
        #expect(batchRows >= 2)
        #expect(2 * batchRows * width * MemoryLayout<Double>.stride <= 262_144)

        let l1 = LayerWeights(
            W: (0..<(width * 16)).map { Double(($0 % 19) - 9) / 100.0 },
            b: (0..<16).map { Double($0) * 0.05 },
            inDim: width,
            outDim: 16
        )
        let l2 = LayerWeights(
            W: (0..<16).map { Double($0 - 8) * 0.1 },
            b: [0.25],
            inDim: 16,
            outDim: 1
        )
        let layers = [l1, l2]
        let rows: [[Double]] = (0..<(batchRows * 2 + 3)).map { r in
            (0..<width).map { c in Double(((r * 7 + c * 3) % 23) - 11) / 31.0 }
        }

        let batched = try #require(try mlpBatchedPrediction(
            features: rows,
            layers: layers,
            activate: { max(0.0, $0) },
            finish: { $0[0] }
        ))
        let scalar = rows.map { scalarForward($0, layers, .relu)[0] }
        expectClose(batched, scalar)

        let flatRowMajor = rows.flatMap { $0 }
        let matrix = PreparedNumericMatrix(
            values: flatRowMajor,
            columnNames: (0..<width).map { "f\($0)" },
            rowCount: rows.count,
            order: .rowMajor,
            sourceRows: Array(0..<rows.count)
        )
        let batchedMatrix = try #require(mlpBatchedPrediction(
            matrix: matrix,
            layers: layers,
            activate: { max(0.0, $0) },
            finish: { $0[0] }
        ) as [Double]?)
        expectClose(batchedMatrix, scalar)
    }

    @Test func scratchLimitBoundary() throws {
        let weights = [Double](repeating: 2, count: 512)
        let layer = LayerWeights(W: weights, b: [0.5], inDim: 512, outDim: 1)
        let row = [Double](repeating: 1, count: 512)
        let input = [[Double]](repeating: row, count: 33)
        let actual: [Double]? = try mlpBatchedPrediction(features: input, layers: [layer],
            activate: { $0 }, finish: { $0[0] })
        #expect(actual == Array(repeating: 1024.5, count: 33))
        let noLayers: [Double]? = try mlpBatchedPrediction(features: input, layers: [],
            activate: { $0 }, finish: { $0[0] })
        #expect(noLayers == nil)
    }

    @Test func refitUsesNewWeights() async throws {
        let input = batchFeatures(65)
        let model = MLPRegressor(hiddenLayerSizes: [9], maxIter: 3, seed: 42, requestedDevice: .cpu)
        try await model.fit(features: input, targets: Array(repeating: 1, count: 65))
        let old = try await model.predict(features: input)
        try await model.fit(features: input, targets: Array(repeating: -2, count: 65))
        let layers = try #require(await model.trainedLayers)
        let current = try await model.predict(features: input)
        expectClose(current, input.map { scalarForward($0, layers, .relu)[0] })
        #expect(old != current)
    }
}

private func batchFeatures(_ rows: Int, offset: Int = 19) -> [[Double]] {
    var result: [[Double]] = []
    for row in 0..<rows {
        var values: [Double] = []
        for col in 0..<7 {
            let integer: Int = ((row + offset) * 13 + col * 7) % 31 - 15
            values.append(Double(integer) / 17)
        }
        result.append(values)
    }
    return result
}

private func scalarForward(_ input: [Double], _ layers: [LayerWeights],
                           _ activation: ActivationFunction) -> [Double] {
    var current = input
    for (index, layer) in layers.enumerated() {
        var output = layer.b
        for j in 0..<layer.outDim {
            for k in 0..<layer.inDim { output[j] += current[k] * layer.W[k * layer.outDim + j] }
            if index < layers.count - 1 {
                switch activation {
                case .relu: output[j] = max(0, output[j])
                case .sigmoid: output[j] = sigmoid(output[j])
                case .tanh: output[j] = tanh(output[j])
                }
            }
        }
        current = output
    }
    return current
}

private func expectClose(_ actual: [Double], _ expected: [Double]) {
    #expect(actual.count == expected.count)
    for (a, e) in zip(actual, expected) {
        #expect(a.isFinite && e.isFinite)
        #expect(abs(a - e) <= 1e-10 + 1e-12 * abs(e))
    }
}
