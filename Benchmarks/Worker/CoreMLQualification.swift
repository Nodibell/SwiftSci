import CoreML
import CryptoKit
import Darwin
import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing
import SwiftSciBenchmarkSupport

// A transport qualification, separate from the standardized production timing profiles.
private struct CoreMLQualificationRecord: Encodable {
    let mode: String
    let taskOutputs: [Double]?
    let workload: String
    let maximumBatchSize: Int
    let matrixInput: Bool
    let publicAdapter: Bool
    let parameterSHA256: String
    let rows: Int
    let columns: Int
    let outputColumns: Int
    let modelSHA256: String
    let inputSHA256: String
    let loadSeconds: Double
    let firstPrediction: CoreMLQualificationSample
    let warmPredictions: [CoreMLQualificationSample]
    let logicalInputPackingBytes: Int?
    let logicalOutputCopyBytes: Int?
    let reservedBytesAfterPrediction: Int
    let reservationPeakBytes: Int
    let residentBytesAfterLoad: UInt64
    let residentBytesAfterWarmup: UInt64
    let residentBytesAfterSamples: UInt64
    let residentBytesPerSample: [UInt64]
    let processPeakRSSBytes: Int64
    let plannedDevices: [String]
    let warmupFailures: Int
    let executionTraceVerified = false
}

func residentBytes() throws -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { throw BenchmarkFailure("Cannot read resident memory") }
    return UInt64(info.resident_size)
}

func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = start.duration(to: .now).components
    return Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
}


func qualifyCoreML(mode: String, output: URL) async -> Int32 {
    do {
        let policies: [String: MLComputeUnits] = ["coreml-cpu": .cpuOnly, "coreml-gpu": .cpuAndGPU,
                                                "coreml-neural": .cpuAndNeuralEngine, "coreml-all": .all]
        let matrixAdapter = mode.hasSuffix("-matrix-adapter")
        let matrixInput = matrixAdapter || mode.hasSuffix("-matrix")
        let suffixCount = matrixAdapter ? 15 : 7
        let policyName = matrixInput ? String(mode.dropLast(suffixCount)) : mode
        let units = policies[policyName]
        guard units != nil || ["native-cpu", "native-gpu", "mlx-cpu", "mlx-gpu"].contains(mode) else {
            throw BenchmarkFailure("Unknown Core ML qualification mode")
        }
        let sampleText = ProcessInfo.processInfo.environment["SWIFTSCI_COREML_SAMPLES"] ?? "32"
        guard let sampleCount = Int(sampleText), (1...512).contains(sampleCount) else {
            throw BenchmarkFailure("Core ML sample count must be between 1 and 512")
        }
        let env = ProcessInfo.processInfo.environment
        let workload = env["SWIFTSCI_COREML_WORKLOAD"] ?? "linear128"
        guard let rows = Int(env["SWIFTSCI_COREML_ROWS"] ?? "256"),
              let batchSize = Int(env["SWIFTSCI_COREML_BATCH"] ?? "1"), (1...8192).contains(batchSize) else {
            throw BenchmarkFailure("Invalid Core ML rows or maximum batch size")
        }
        let fixture: CoreMLQualificationFixture
        let fixturePath = env["SWIFTSCI_COREML_FIXTURE"]
        if let fixturePath {
            fixture = try CoreMLQualificationFixture(contentsOf: URL(fileURLWithPath: fixturePath), name: workload, rows: rows)
        } else {
            fixture = try CoreMLQualificationFixture(name: workload, rows: rows)
        }
        let width = fixture.width
        let names = fixture.names
        let columns = fixture.columns
        let expected = fixture.expected
        let input = try PreparedNumericBatch(columnNames: names, columns: columns)
        let data = fixture.artifact(matrix: matrixInput)
        if mode.hasPrefix("native-"), workload != "linear128" {
            throw BenchmarkFailure("Native LinearRegression is only comparable with linear128")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("Qualification.mlmodel")
        try data.write(to: source)
        let compiled = try await MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let budget = try MemoryBudget(limit: 256 * 1_048_576)
        let loadStart = ContinuousClock.now
        let predictor: CoreMLPredictor?
        let native: LinearRegression?
        let mlx: CoreMLQualificationMLX?
        let matrix: CoreMLQualificationMatrix?
        if let units {
            if matrixInput && !matrixAdapter {
                matrix = try CoreMLQualificationMatrix(url: compiled, units: units, rows: rows, width: width, outputs: fixture.outputWidth)
                predictor = nil
            } else {
                predictor = try CoreMLPredictor(compiledModelURL: compiled, inputColumns: names,
                                                outputName: "result", computeUnits: units,
                                                inputLayout: matrixAdapter ? .matrix : .examples)
                matrix = nil
            }
            native = nil
            mlx = nil
        } else {
            predictor = nil
            matrix = nil
            if mode.hasPrefix("mlx-") {
                mlx = CoreMLQualificationMLX(fixture: fixture, gpu: mode == "mlx-gpu")
                native = nil
            } else {
                let layer = fixture.layers[0]
                native = LinearRegression(weights: layer.W, bias: layer.b[0], device: mode == "native-cpu" ? .cpu : .gpu)
                mlx = nil
            }
        }
        let loadSeconds = elapsedSeconds(since: loadStart)
        let afterLoad = try residentBytes()
        var taskOutputs: [Double]?
        var samples: [CoreMLQualificationSample] = []
        var packingBytes: Int?
        var outputBytes: Int?
        var warmupFailures = 0
        var afterWarmup: UInt64 = 0
        var residentSamples: [UInt64] = []
        for iteration in 0..<(sampleCount + 4) {
            let start = ContinuousClock.now
            let values: [Double]
            if let predictor {
                let result = try await predictor.predict(input, budget: budget, workspaceBytes: 8 * 1_048_576, maximumBatchSize: matrixAdapter ? 1 : batchSize)
                var packed = [Double]()
                packed.reserveCapacity(rows * fixture.outputWidth)
                for row in 0..<rows {
                    for column in 0..<fixture.outputWidth { packed.append(result.values[row, column]!) }
                }
                values = packed
                packingBytes = result.inputPackingBytes
                outputBytes = result.outputCopyBytes
            } else if let matrix {
                values = try matrix.predict(input)
                packingBytes = rows * width * MemoryLayout<Double>.stride
                outputBytes = rows * fixture.outputWidth * MemoryLayout<Double>.stride
            } else if let mlx {
                values = try mlx.predict(input)
            } else if let native {
                values = try await native.predict(features: input)
            } else { throw BenchmarkFailure("No predictor") }
            let nanos = Int64(elapsedSeconds(since: start) * 1e9)
            if iteration == 0, fixturePath != nil { taskOutputs = values }
            let sample = try CoreMLQualificationSample(elapsed: nanos, values: values, expected: expected)
            if iteration < 4, !sample.validated { warmupFailures += 1 }
            if iteration == 0 || iteration >= 4 { samples.append(sample) }
            if iteration == 3 { afterWarmup = try residentBytes() }
            if iteration >= 4 { residentSamples.append(try residentBytes()) }
        }
        let afterSamples = try residentBytes()
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { throw BenchmarkFailure("Cannot read peak RSS") }
        let reserved = await budget.reservedBytes
        guard reserved == 0 else { throw BenchmarkFailure("Prediction leaked an admission reservation") }
        let devices: [String]
        if let units { devices = try await plannedDevices(at: compiled, units: units) }
        else { devices = [mode] }
        let inputData = columns.flatMap { $0 }.withUnsafeBytes { Data($0) }
        let parameterEncoder = JSONEncoder()
        parameterEncoder.outputFormatting = [.sortedKeys]
        let parameters = try parameterEncoder.encode(fixture.layers)
        let record = CoreMLQualificationRecord(mode: mode, taskOutputs: taskOutputs, workload: workload, maximumBatchSize: batchSize,
            matrixInput: matrixInput, publicAdapter: predictor != nil,
            parameterSHA256: SHA256.hash(data: parameters).map { String(format: "%02x", $0) }.joined(),
            rows: rows, columns: width, outputColumns: fixture.outputWidth,
            modelSHA256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            inputSHA256: SHA256.hash(data: inputData).map { String(format: "%02x", $0) }.joined(),
            loadSeconds: loadSeconds, firstPrediction: samples[0], warmPredictions: Array(samples.dropFirst()),
            logicalInputPackingBytes: packingBytes, logicalOutputCopyBytes: outputBytes,
            reservedBytesAfterPrediction: reserved, reservationPeakBytes: await budget.peak,
            residentBytesAfterLoad: afterLoad, residentBytesAfterWarmup: afterWarmup,
            residentBytesAfterSamples: afterSamples, residentBytesPerSample: residentSamples, processPeakRSSBytes: Int64(usage.ru_maxrss), plannedDevices: devices, warmupFailures: warmupFailures)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(record).write(to: output, options: .atomic)
        return warmupFailures == 0 && samples.allSatisfy(\.validated) ? 0 : 2
    } catch {
        fputs("Core ML qualification failed: \(error)\n", stderr)
        return 1
    }
}

// Legacy validation fields describe Double-reference accuracy, independently of execution completion.
private struct CoreMLQualificationSample: Encodable {
    let elapsed_ns: Int64
    let output_sha256: String
    let maximum_absolute_error: Double
    let validated: Bool
    let mismatchedValues: Int
    let firstActual: Double?
    let firstExpected: Double?

    init(elapsed: Int64, values: [Double], expected: [Double]) throws {
        guard values.count == expected.count, values.allSatisfy(\.isFinite), expected.allSatisfy(\.isFinite) else {
            throw BenchmarkFailure("Invalid qualification output count or nonfinite value")
        }
        elapsed_ns = elapsed
        output_sha256 = values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
        do {
            let sample = try BenchmarkSample(elapsed: elapsed, values: values, expected: expected, atol: 1e-5, rtol: 1e-5)
            maximum_absolute_error = sample.maximum_absolute_error
            validated = true
            mismatchedValues = 0
            firstActual = nil
            firstExpected = nil
        } catch {
            let errors = zip(values, expected).map { abs($0 - $1) }
            let failures = values.indices.filter { errors[$0] > 1e-5 + 1e-5 * abs(expected[$0]) }
            guard let first = failures.first else { throw error }
            maximum_absolute_error = errors.max() ?? 0
            validated = false
            mismatchedValues = failures.count
            firstActual = values[first]
            firstExpected = expected[first]
        }
    }
}
