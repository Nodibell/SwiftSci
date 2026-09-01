// AccuracyBenchmarks.swift
// Evaluates and reports end-to-end model accuracy and forecast quality error metrics:
//   • Time-Series Forecasting: ARIMA & Holt-Winters (RMSE, MAE, MAPE, R²)
//   • Regression Models: GBDT Regressor & LinearRegression (RMSE, MAE, R²)
//   • Classification Models: RandomForest & NaiveBayes (Accuracy, Precision, Recall, F1)

import Foundation
import SwiftML
import SwiftForecast
import SwiftOptimize
import SwiftNLP
import SwiftStats

public struct AccuracyBenchmarks: BenchmarkSuite {
    public let module = "SwiftSci Accuracy & Quality"

    public init() {}

    private static func printRow(_ text: String, width: Int = 80) {
        let trimmed = text.count > width ? String(text.prefix(width)) : text
        let pad = max(0, width - trimmed.count)
        print("  │ " + trimmed + String(repeating: " ", count: pad) + " │")
    }

    public func run() async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        var rng = BenchmarkLCG(seed: 42)

        let lineWidth = 82
        let border = String(repeating: "─", count: lineWidth + 2)
        print("  ┌" + border + "┐")
        let title = "MODEL ACCURACY & FORECAST QUALITY SCORECARD"
        let titlePadLeft = (lineWidth - title.count) / 2
        let titlePadRight = lineWidth - title.count - titlePadLeft
        print("  │ " + String(repeating: " ", count: titlePadLeft) + title + String(repeating: " ", count: titlePadRight) + " │")
        print("  ├" + border + "┤")

        // ── 1. Time-Series Forecasting Accuracy (Holt-Winters on Seasonal Trend) ──
        let nTotal = 500
        let nTrain = 476
        let horizon = 24
        let fullSeries: [Double] = (0..<nTotal).map { t in
            let trend = Double(t) * 0.25
            let seasonal = 8.0 * sin(Double(t) * 2.0 * .pi / 12.0)
            let noise = rng.nextDouble(in: -0.5...0.5)
            return 20.0 + trend + seasonal + noise
        }
        let trainSeries = Array(fullSeries[0..<nTrain])
        let actualFuture = Array(fullSeries[nTrain..<nTotal])

        // Fit Holt-Winters
        let hw = ExponentialSmoothing(
            method: .holtWinters(beta: 0.1, gamma: 0.1, period: 12, seasonal: .additive),
            alpha: 0.2
        )
        _ = try? await hw.fit(series: trainSeries)
        let hwForecast = (try? await hw.forecast(horizon: horizon).predictions) ?? Array(repeating: trainSeries.last ?? 0, count: horizon)

        let hwRMSE = Metrics.rootMeanSquaredError(yTrue: actualFuture, yPred: hwForecast)
        let hwMAE  = Metrics.meanAbsoluteError(yTrue: actualFuture, yPred: hwForecast)
        let hwMAPE = Metrics.mape(yTrue: actualFuture, yPred: hwForecast)
        let hwR2   = Metrics.r2Score(yTrue: actualFuture, yPred: hwForecast)

        Self.printRow(String(format: "[Forecast] Holt-Winters (h=%d) : RMSE=%.3f, MAE=%.3f, MAPE=%.2f%%, R²=%.3f", horizon, hwRMSE, hwMAE, hwMAPE, hwR2), width: lineWidth)

        let hwResult = await BenchmarkRunner.run(
            name: "Holt-Winters Accuracy (RMSE, MAE, MAPE, R²)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let hwModel = ExponentialSmoothing(
                method: .holtWinters(beta: 0.1, gamma: 0.1, period: 12, seasonal: .additive),
                alpha: 0.2
            )
            try await hwModel.fit(series: trainSeries)
            let res = try await hwModel.forecast(horizon: horizon)
            _ = Metrics.rootMeanSquaredError(yTrue: actualFuture, yPred: res.predictions)
            _ = Metrics.meanAbsoluteError(yTrue: actualFuture, yPred: res.predictions)
            _ = Metrics.mape(yTrue: actualFuture, yPred: res.predictions)
            _ = Metrics.r2Score(yTrue: actualFuture, yPred: res.predictions)
        }
        results.append(hwResult)

        // ── 2. Time-Series Forecasting Accuracy (ARIMA on Trend Series) ────────
        let arima = try? ARIMAModel(p: 1, d: 1, q: 1)
        _ = try? await arima?.fit(series: trainSeries)
        let arimaForecast = (try? await arima?.forecast(horizon: horizon).forecast.predictions) ?? Array(repeating: trainSeries.last ?? 0, count: horizon)

        let arimaRMSE = Metrics.rootMeanSquaredError(yTrue: actualFuture, yPred: arimaForecast)
        let arimaMAE  = Metrics.meanAbsoluteError(yTrue: actualFuture, yPred: arimaForecast)
        let arimaMAPE = Metrics.mape(yTrue: actualFuture, yPred: arimaForecast)
        let arimaR2   = Metrics.r2Score(yTrue: actualFuture, yPred: arimaForecast)

        Self.printRow(String(format: "[Forecast] ARIMA(1,1,1) (h=%d) : RMSE=%.3f, MAE=%.3f, MAPE=%.2f%%, R²=%.3f", horizon, arimaRMSE, arimaMAE, arimaMAPE, arimaR2), width: lineWidth)

        let arimaResult = await BenchmarkRunner.run(
            name: "ARIMA(1,1,1) Accuracy (RMSE, MAE, MAPE, R²)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let m = try ARIMAModel(p: 1, d: 1, q: 1)
            try await m.fit(series: trainSeries)
            let res = try await m.forecast(horizon: horizon)
            _ = Metrics.rootMeanSquaredError(yTrue: actualFuture, yPred: res.forecast.predictions)
            _ = Metrics.meanAbsoluteError(yTrue: actualFuture, yPred: res.forecast.predictions)
            _ = Metrics.mape(yTrue: actualFuture, yPred: res.forecast.predictions)
            _ = Metrics.r2Score(yTrue: actualFuture, yPred: res.forecast.predictions)
        }
        results.append(arimaResult)

        // ── 3. GBDT Regressor Accuracy on Synthetic Non-linear Function ───────
        let nReg = 1_000
        let nTrainReg = 800
        var allX: [[Double]] = []
        var allY: [Double] = []
        for _ in 0..<nReg {
            let x1 = rng.nextDouble(in: -3.0...3.0)
            let x2 = rng.nextDouble(in: -3.0...3.0)
            let yVal = 2.0 * x1 + sin(x2) * 3.0 + rng.nextDouble(in: -0.2...0.2)
            allX.append([x1, x2])
            allY.append(yVal)
        }
        let xTrain = Array(allX[0..<nTrainReg])
        let yTrain = Array(allY[0..<nTrainReg])
        let xTest  = Array(allX[nTrainReg..<nReg])
        let yTest  = Array(allY[nTrainReg..<nReg])

        let gbdt = try? GradientBoostedTreesRegressor(nEstimators: 30, learningRate: 0.1, maxDepth: 4)
        _ = try? await gbdt?.fit(features: xTrain, targets: yTrain)
        let gbdtPred = (try? await gbdt?.predict(features: xTest)) ?? Array(repeating: 0.0, count: xTest.count)

        let gbdtRMSE = Metrics.rootMeanSquaredError(yTrue: yTest, yPred: gbdtPred)
        let gbdtMAE  = Metrics.meanAbsoluteError(yTrue: yTest, yPred: gbdtPred)
        let gbdtR2   = Metrics.r2Score(yTrue: yTest, yPred: gbdtPred)

        Self.printRow(String(format: "[ML Reg]   GBDT (30 trees, d=4) : RMSE=%.3f, MAE=%.3f, R²=%.4f", gbdtRMSE, gbdtMAE, gbdtR2), width: lineWidth)

        let gbdtResult = await BenchmarkRunner.run(
            name: "GBDT Regressor Quality (RMSE, MAE, R²)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let g = try GradientBoostedTreesRegressor(nEstimators: 30, learningRate: 0.1, maxDepth: 4)
            try await g.fit(features: xTrain, targets: yTrain)
            let pred = try await g.predict(features: xTest)
            _ = Metrics.rootMeanSquaredError(yTrue: yTest, yPred: pred)
            _ = Metrics.r2Score(yTrue: yTest, yPred: pred)
        }
        results.append(gbdtResult)

        // ── 4. Random Forest Classifier Accuracy & F1 Score ───────────────────
        let nCls = 1_000
        let nTrainCls = 800
        var clsX: [[Double]] = []
        var clsY: [Double] = []
        for _ in 0..<nCls {
            let x1 = rng.nextDouble(in: -2.0...2.0)
            let x2 = rng.nextDouble(in: -2.0...2.0)
            let label = (x1 * 0.8 + x2 * 0.6 > 0.0) ? 1.0 : 0.0
            clsX.append([x1, x2])
            clsY.append(label)
        }
        let clsXTrain = Array(clsX[0..<nTrainCls])
        let clsYTrain = Array(clsY[0..<nTrainCls])
        let clsXTest  = Array(clsX[nTrainCls..<nCls])
        let clsYTest  = Array(clsY[nTrainCls..<nCls])

        let rf = try? RandomForestClassifier(nEstimators: 30, maxDepth: 5, criterion: .gini)
        _ = try? await rf?.fit(features: clsXTrain, targets: clsYTrain)
        let rfPred = (try? await rf?.predict(features: clsXTest)) ?? Array(repeating: 0, count: clsXTest.count)

        let yTestInt = clsYTest.map { Int($0) }

        let rfAcc = Metrics.accuracy(yTrue: yTestInt, yPred: rfPred)
        let rfF1  = Metrics.f1Score(yTrue: yTestInt, yPred: rfPred, label: 1)

        Self.printRow(String(format: "[ML Cls]   RandomForest (30 tr.): Accuracy=%.2f%%, F1=%.3f", rfAcc * 100.0, rfF1), width: lineWidth)

        let rfResult = await BenchmarkRunner.run(
            name: "RandomForest Quality (Accuracy, F1)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let model = try RandomForestClassifier(nEstimators: 30, maxDepth: 5, criterion: .gini)
            try await model.fit(features: clsXTrain, targets: clsYTrain)
            let pred = try await model.predict(features: clsXTest)
            _ = Metrics.accuracy(yTrue: yTestInt, yPred: pred)
            _ = Metrics.f1Score(yTrue: yTestInt, yPred: pred, label: 1)
        }
        results.append(rfResult)

        // ── 5. NaiveBayes Classifier Quality on Multi-class Text Counts ────────
        let nbXTrain = (0..<800).map { i in (0..<50).map { j in Double((i + j) % 5) } }
        let nbYTrain = (0..<800).map { Double($0 % 3) }
        let nbXTest  = (0..<200).map { i in (0..<50).map { j in Double((i + j) % 5) } }
        let nbYTest  = (0..<200).map { Double($0 % 3) }

        let nb = NaiveBayesClassifier(alpha: 1.0)
        _ = try? await nb.fit(features: nbXTrain, targets: nbYTrain)
        let nbPred = (try? await nb.predict(features: nbXTest)) ?? Array(repeating: 0, count: nbXTest.count)
        let nbYTestInt = nbYTest.map { Int($0) }

        let nbReport = Metrics.classificationReport(yTrue: nbYTestInt, yPred: nbPred)
        let nbAcc = nbReport.accuracy
        let nbF1  = nbReport.macroF1

        Self.printRow(String(format: "[NLP Cls]  NaiveBayes (3-class) : Accuracy=%.2f%%, Macro-F1=%.3f", nbAcc * 100.0, nbF1), width: lineWidth)
        print("  └" + border + "┘")

        let nbResult = await BenchmarkRunner.run(
            name: "NaiveBayes Quality (Accuracy, Macro-F1)",
            module: module,
            warmup: 1,
            iterations: 5
        ) {
            let model = NaiveBayesClassifier(alpha: 1.0)
            try await model.fit(features: nbXTrain, targets: nbYTrain)
            let pred = try await model.predict(features: nbXTest)
            _ = Metrics.accuracy(yTrue: nbYTestInt, yPred: pred)
        }
        results.append(nbResult)

        return results
    }
}
