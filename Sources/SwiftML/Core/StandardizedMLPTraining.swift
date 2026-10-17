import SwiftDataFrame
import SwiftPreprocessing

struct StandardizedMLPTraining: Sendable {
    let plan: StandardPreprocessingPlan
    let rows: [[Double]]
    let validation: (rows: [[Double]], targets: [Double])?

    init(features: PreparedNumericBatch, targets: [Double],
         validation: PreparedSupervisedBatch?, strategy: Imputer.Strategy) throws {
        guard Set(features.columnNames).count == features.columnCount,
              targets.count == features.rowCount, targets.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidParameter("Training requires unique names and aligned finite targets")
        }
        let plan = try StandardPreprocessingPlan(training: features, strategy: strategy)
        guard plan.replacements.allSatisfy(\.isFinite), plan.negativeMeans.allSatisfy(\.isFinite),
              plan.deviations.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw SwiftMLError.invalidInput("Preprocessing statistics must be finite")
        }
        self.plan = plan
        rows = try Self.transform(features, plan: plan)
        if let validation {
            let batch = validation.features, targets = validation.targets
            guard batch.rowCount > 0, targets.allSatisfy(\.isFinite) else {
                throw SwiftMLError.invalidInput("Validation requires nonempty rows and finite targets")
            }
            self.validation = (try Self.transform(batch, plan: plan), targets)
        } else { self.validation = nil }
    }

    private static func transform(_ batch: PreparedNumericBatch,
                                  plan: StandardPreprocessingPlan) throws -> [[Double]] {
        let matrix = try plan.preparedMatrix(batch)
        guard matrix.values.allSatisfy(\.isFinite) else {
            throw SwiftMLError.invalidInput("Standardized training values must be finite")
        }
        return (0..<matrix.rowCount).map { row -> [Double] in
            let start = row * matrix.columnCount
            return Array(matrix.values[start..<(start + matrix.columnCount)])
        }
    }
}
