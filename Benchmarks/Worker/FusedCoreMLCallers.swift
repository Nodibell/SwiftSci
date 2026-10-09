import Foundation
import SwiftDataFrame
import SwiftML
import SwiftPreprocessing

func measureFusionCallers(mode: String, plan: FusedPreprocessingPlan, inputs: [PreparedNumericBatch],
    pool: CoreMLMatrixPool, budget: MemoryBudget, contract: CoreMLTrialInputContract) async throws -> (Double, [FusionPrediction]) {
    if mode == "native-owned" {
        // Give the consuming path genuinely unique column buffers. Fixture creation
        // is outside timing, just as shared fixture creation is for the other modes.
        let input = try inputs[0].selectingRows(Array(0..<inputs[0].rowCount))
        let start = ContinuousClock.now
        let result = try await fusionPredict(mode: mode, caller: 0, plan: plan, input: consume input,
            pool: pool, budget: budget, contract: contract)
        try await awaitFusionRelease(budget)
        return (elapsedSeconds(since: start), [result])
    }
    if inputs.count == 1 {
        let start = ContinuousClock.now
        let result = try await fusionPredict(mode: mode, caller: 0, plan: plan, input: inputs[0],
            pool: pool, budget: budget, contract: contract)
        try await awaitFusionRelease(budget)
        return (elapsedSeconds(since: start), [result])
    }
    let start = ContinuousClock.now
    let outputs = try await withThrowingTaskGroup(of: FusionPrediction.self) { group in
        for caller in inputs.indices {
            group.addTask {
                try await fusionPredict(mode: mode, caller: caller, plan: plan,
                    input: inputs[caller], pool: pool, budget: budget, contract: contract)
            }
        }
        var results = [FusionPrediction]()
        for try await result in group { results.append(result) }
        return results.sorted { $0.caller < $1.caller }
    }
    try await awaitFusionRelease(budget)
    return (elapsedSeconds(since: start), outputs)
}
