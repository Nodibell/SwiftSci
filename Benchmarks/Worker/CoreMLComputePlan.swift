import CoreML
import Foundation

func plannedDevices(at url: URL, units: MLComputeUnits) async throws -> [String] {
    guard #available(macOS 14.4, *) else { return ["unavailable before macOS 14.4"] }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = units
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
    switch plan.modelStructure {
    case .neuralNetwork(let network):
        return network.layers.map { "\($0.name): \(deviceName(plan.deviceUsage(for: $0)?.preferred))" }
    case .program(let program):
        return program.functions.keys.sorted().flatMap { name in
            programDevices(program.functions[name]!.block, path: name, plan: plan)
        }
    default: return ["unsupported model structure"]
    }
}

@available(macOS 14.4, *)
private func programDevices(_ block: MLModelStructure.Program.Block, path: String,
                            plan: MLComputePlan) -> [String] {
    var devices: [String] = []
    for (index, operation) in block.operations.enumerated() {
        let location = "\(path)/\(index):\(operation.operatorName)"
        let outputs = operation.outputs.map(\.name).joined(separator: ",")
        devices.append("\(location) [\(outputs)]: \(deviceName(plan.deviceUsage(for: operation)?.preferred))")
        for (childIndex, child) in operation.blocks.enumerated() {
            devices += programDevices(child, path: "\(location)/block\(childIndex)", plan: plan)
        }
    }
    return devices
}

@available(macOS 14.4, *)
private func deviceName(_ device: MLComputeDevice?) -> String {
    switch device {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neural-engine"
    case nil: return "unknown"
    @unknown default: return "unknown"
    }
}
