// AccuracyBenchmarks.swift
// Evaluates and reports end-to-end model accuracy and forecast quality error metrics:
//   • Time-Series Forecasting: ARIMA & Holt-Winters (RMSE, MAE, MAPE, R²)
//   • Linear & Regularized Regression: OLS LinearRegression (RMSE, MAE, R²)
//   • Non-linear & Ensemble Regression: GBDT, HistGBDT, DecisionTree (RMSE, MAE, R²)
//   • Classification: LogisticRegression, LinearSVC, RandomForest, DecisionTree, HistGBDT, NaiveBayes (Accuracy, F1)
//   • Unsupervised Clustering & Decomposition: PCA SVD spectrum, KMeans inertia
//   • Feature Preprocessing: StandardScaler, MinMaxScaler (IEEE 754 parity)
//   • Statistical Tests: Welch t, Student t, Paired t, ANOVA, Pearson, Spearman
//   • NLP: VADER Sentiment (compound polarity)

import Foundation
import SwiftML
import SwiftForecast
import SwiftOptimize
import SwiftNLP
import SwiftStats
import SwiftCluster
import SwiftPreprocessing
import Accelerate

public struct AccuracyBenchmarks: BenchmarkSuite {
    public let module = "SwiftSci Accuracy & Quality"

    public init() {}

    private static func printRow(_ text: String, width: Int = 82) {
        let trimmed = text.count > width ? String(text.prefix(width)) : text
        let pad = max(0, width - trimmed.count)
        print("  │ " + trimmed + String(repeating: " ", count: pad) + " │")
    }

    private static func printSection(_ title: String, width: Int = 82) {
        let border = String(repeating: "─", count: width + 2)
        print("  ├" + border + "┤")
        printRow("▶ " + title, width: width)
        print("  ├" + border + "┤")
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

        // ══════════════════════════════════════════════════════════════════
        // ── 1. Time-Series Forecasting (Holt-Winters & ARIMA) ─────────────
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("1. TIME-SERIES FORECASTING (Horizon = 24)", width: lineWidth)

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

        // 1.1 Holt-Winters
        let hw = ExponentialSmoothing(
            method: .holtWinters(beta: 0.1, gamma: 0.1, period: 12, seasonal: .additive)
        )
        _ = try? await hw.fit(series: trainSeries)
        let hwForecast = (try? await hw.forecast(horizon: horizon).predictions) ?? Array(repeating: trainSeries.last ?? 0, count: horizon)

        let hwRMSE = Metrics.rootMeanSquaredError(yTrue: actualFuture, yPred: hwForecast)
        let hwMAE  = Metrics.meanAbsoluteError(yTrue: actualFuture, yPred: hwForecast)
        let hwMAPE = Metrics.mape(yTrue: actualFuture, yPred: hwForecast)
        let hwR2   = Metrics.r2Score(yTrue: actualFuture, yPred: hwForecast)

        Self.printRow(String(format: "[Forecast] Holt-Winters (h=%d) : RMSE=%.3f, MAE=%.3f, MAPE=%.2f%%, R²=%.3f", horizon, hwRMSE, hwMAE, hwMAPE, hwR2), width: lineWidth)

        let hwResult = await BenchmarkRunner.run(
            name: "Holt-Winters Forecast (RMSE, MAE, R²)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = ExponentialSmoothing(method: .holtWinters(beta: 0.1, gamma: 0.1, period: 12, seasonal: .additive))
            try await m.fit(series: trainSeries)
            _ = try await m.forecast(horizon: horizon)
        }
        results.append(hwResult)

        // 1.2 ARIMA(1,1,1)
        let arima = try? ARIMAModel(p: 1, d: 1, q: 1)
        _ = try? await arima?.fit(series: trainSeries)
        let arimaForecast = (try? await arima?.forecast(horizon: horizon).forecast.predictions) ?? Array(repeating: trainSeries.last ?? 0, count: horizon)

        let arimaRMSE = Metrics.rootMeanSquaredError(yTrue: actualFuture, yPred: arimaForecast)
        let arimaMAE  = Metrics.meanAbsoluteError(yTrue: actualFuture, yPred: arimaForecast)
        let arimaMAPE = Metrics.mape(yTrue: actualFuture, yPred: arimaForecast)
        let arimaR2   = Metrics.r2Score(yTrue: actualFuture, yPred: arimaForecast)

        Self.printRow(String(format: "[Forecast] ARIMA(1,1,1) (h=%d) : RMSE=%.3f, MAE=%.3f, MAPE=%.2f%%, R²=%.3f", horizon, arimaRMSE, arimaMAE, arimaMAPE, arimaR2), width: lineWidth)

        let arimaResult = await BenchmarkRunner.run(
            name: "ARIMA(1,1,1) Forecast (RMSE, MAE, R²)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = try ARIMAModel(p: 1, d: 1, q: 1)
            try await m.fit(series: trainSeries)
            _ = try await m.forecast(horizon: horizon)
        }
        results.append(arimaResult)

        // ══════════════════════════════════════════════════════════════════
        // ── 2. Supervised Regression (OLS, GBDT, HistGBDT, DT) ────────────
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("2. SUPERVISED REGRESSION (N = 1000, 80/20 split)", width: lineWidth)

        // 2.1 Linear OLS Dataset (analytical linear function: y = 3 x1 - 2 x2 + 1.5 x3 + 0.5 + noise)
        let nLin = 1_000
        var linX: [[Double]] = []
        var linY: [Double] = []
        for _ in 0..<nLin {
            let x1 = rng.nextDouble(in: -2.0...2.0)
            let x2 = rng.nextDouble(in: -2.0...2.0)
            let x3 = rng.nextDouble(in: -2.0...2.0)
            let y = 3.0 * x1 - 2.0 * x2 + 1.5 * x3 + 0.5 + rng.nextDouble(in: -0.1...0.1)
            linX.append([x1, x2, x3])
            linY.append(y)
        }
        let linXTrain = Array(linX[0..<800])
        let linYTrain = Array(linY[0..<800])
        let linXTest  = Array(linX[800..<nLin])
        let linYTest  = Array(linY[800..<nLin])

        let ols = LinearRegression(device: .cpu)
        _ = try? await ols.fit(features: linXTrain, targets: linYTrain)
        let olsPred = (try? await ols.predict(features: linXTest)) ?? Array(repeating: 0.0, count: 200)
        let olsRMSE = Metrics.rootMeanSquaredError(yTrue: linYTest, yPred: olsPred)
        let olsMAE  = Metrics.meanAbsoluteError(yTrue: linYTest, yPred: olsPred)
        let olsR2   = Metrics.r2Score(yTrue: linYTest, yPred: olsPred)

        Self.printRow(String(format: "[ML Reg]   OLS Linear (LAPACK)  : RMSE=%.4f, MAE=%.4f, R²=%.5f", olsRMSE, olsMAE, olsR2), width: lineWidth)

        let olsResult = await BenchmarkRunner.run(
            name: "OLS Linear Regression (LAPACK)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = LinearRegression(device: .cpu)
            try await m.fit(features: linXTrain, targets: linYTrain)
            _ = try await m.predict(features: linXTest)
        }
        results.append(olsResult)

        // 2.2 Non-linear Synthetic Function: y = 2 x1 + sin(x2) * 3 + noise
        let nReg = 1_000
        var allX: [[Double]] = []
        var allY: [Double] = []
        for _ in 0..<nReg {
            let x1 = rng.nextDouble(in: -3.0...3.0)
            let x2 = rng.nextDouble(in: -3.0...3.0)
            let yVal = 2.0 * x1 + sin(x2) * 3.0 + rng.nextDouble(in: -0.2...0.2)
            allX.append([x1, x2])
            allY.append(yVal)
        }
        let xTrain = Array(allX[0..<800])
        let yTrain = Array(allY[0..<800])
        let xTest  = Array(allX[800..<nReg])
        let yTest  = Array(allY[800..<nReg])

        // GBDT Regressor
        let gbdt = try? GradientBoostedTreesRegressor(nEstimators: 30, learningRate: 0.1, maxDepth: 4)
        _ = try? await gbdt?.fit(features: xTrain, targets: yTrain)
        let gbdtPred = (try? await gbdt?.predict(features: xTest)) ?? Array(repeating: 0.0, count: xTest.count)
        let gbdtRMSE = Metrics.rootMeanSquaredError(yTrue: yTest, yPred: gbdtPred)
        let gbdtR2   = Metrics.r2Score(yTrue: yTest, yPred: gbdtPred)

        Self.printRow(String(format: "[ML Reg]   GBDT (30 tr, d=4)    : RMSE=%.3f, R²=%.4f", gbdtRMSE, gbdtR2), width: lineWidth)

        let gbdtResult = await BenchmarkRunner.run(
            name: "GBDT Regressor Quality (RMSE, R²)",
            module: module, warmup: 1, iterations: 3
        ) {
            let g = try GradientBoostedTreesRegressor(nEstimators: 30, learningRate: 0.1, maxDepth: 4)
            try await g.fit(features: xTrain, targets: yTrain)
            _ = try await g.predict(features: xTest)
        }
        results.append(gbdtResult)

        // HistGradientBoosting Regressor
        let hgbr = try? HistGradientBoostingRegressor(nEstimators: 30, learningRate: 0.1, maxDepth: 4, maxBins: 256)
        _ = try? await hgbr?.fit(features: xTrain, targets: yTrain)
        let hgbrPred = (try? await hgbr?.predict(features: xTest)) ?? Array(repeating: 0.0, count: xTest.count)
        let hgbrRMSE = Metrics.rootMeanSquaredError(yTrue: yTest, yPred: hgbrPred)
        let hgbrR2   = Metrics.r2Score(yTrue: yTest, yPred: hgbrPred)

        Self.printRow(String(format: "[ML Reg]   HistGBDT (30 tr, b256): RMSE=%.3f, R²=%.4f", hgbrRMSE, hgbrR2), width: lineWidth)

        let hgbrResult = await BenchmarkRunner.run(
            name: "HistGBDT Regressor (RMSE, R²)",
            module: module, warmup: 1, iterations: 3
        ) {
            let h = try HistGradientBoostingRegressor(nEstimators: 30, learningRate: 0.1, maxDepth: 4, maxBins: 256)
            try await h.fit(features: xTrain, targets: yTrain)
            _ = try await h.predict(features: xTest)
        }
        results.append(hgbrResult)

        // Decision Tree Regressor
        let dtr = DecisionTreeRegressor(maxDepth: 5, minSamplesSplit: 2)
        _ = try? await dtr.fit(features: xTrain, targets: yTrain)
        let dtrPred = (try? await dtr.predict(features: xTest)) ?? Array(repeating: 0.0, count: xTest.count)
        let dtrRMSE = Metrics.rootMeanSquaredError(yTrue: yTest, yPred: dtrPred)
        let dtrR2   = Metrics.r2Score(yTrue: yTest, yPred: dtrPred)

        Self.printRow(String(format: "[ML Reg]   DecisionTree (d=5)   : RMSE=%.3f, R²=%.4f", dtrRMSE, dtrR2), width: lineWidth)

        let dtrResult = await BenchmarkRunner.run(
            name: "DecisionTree Regressor (RMSE, R²)",
            module: module, warmup: 1, iterations: 3
        ) {
            let d = DecisionTreeRegressor(maxDepth: 5, minSamplesSplit: 2)
            try await d.fit(features: xTrain, targets: yTrain)
            _ = try await d.predict(features: xTest)
        }
        results.append(dtrResult)

        // ══════════════════════════════════════════════════════════════════
        // ── 3. Supervised Classification (LogReg, SVC, RF, HistGB, DT, NB)
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("3. SUPERVISED CLASSIFICATION (N = 1000, 80/20 split)", width: lineWidth)

        let nCls = 1_000
        var clsX: [[Double]] = []
        var clsY: [Double] = []
        for _ in 0..<nCls {
            let x1 = rng.nextDouble(in: -2.0...2.0)
            let x2 = rng.nextDouble(in: -2.0...2.0)
            let label = (x1 * 0.8 + x2 * 0.6 > 0.0) ? 1.0 : 0.0
            clsX.append([x1, x2])
            clsY.append(label)
        }
        let clsXTrain = Array(clsX[0..<800])
        let clsYTrain = Array(clsY[0..<800])
        let clsXTest  = Array(clsX[800..<nCls])
        let clsYTest  = Array(clsY[800..<nCls])
        let yTestInt  = clsYTest.map { Int($0) }

        // 3.1 Logistic Regression
        let logreg = LogisticRegression(device: .cpu)
        _ = try? await logreg.fit(features: clsXTrain, targets: clsYTrain, learningRate: 0.1, epochs: 500)
        let logregPred = (try? await logreg.predict(features: clsXTest)) ?? Array(repeating: 0, count: 200)
        let logregAcc = Metrics.accuracy(yTrue: yTestInt, yPred: logregPred)
        let logregF1  = Metrics.f1Score(yTrue: yTestInt, yPred: logregPred, label: 1)

        Self.printRow(String(format: "[ML Cls]   LogisticRegression   : Accuracy=%.2f%%, F1=%.3f", logregAcc * 100.0, logregF1), width: lineWidth)

        let logregResult = await BenchmarkRunner.run(
            name: "LogisticRegression (Accuracy, F1)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = LogisticRegression(device: .cpu)
            try await m.fit(features: clsXTrain, targets: clsYTrain, learningRate: 0.1, epochs: 500)
            _ = try await m.predict(features: clsXTest)
        }
        results.append(logregResult)

        // 3.2 LinearSVC
        let svc = LinearSVC(C: 1.0, device: .cpu)
        _ = try? await svc.fit(features: clsXTrain, targets: clsYTrain, learningRate: 0.1, epochs: 500)
        let svcPred = (try? await svc.predict(features: clsXTest)) ?? Array(repeating: 0, count: 200)
        let svcAcc = Metrics.accuracy(yTrue: yTestInt, yPred: svcPred)
        let svcF1  = Metrics.f1Score(yTrue: yTestInt, yPred: svcPred, label: 1)

        Self.printRow(String(format: "[ML Cls]   LinearSVC (C=1.0)    : Accuracy=%.2f%%, F1=%.3f", svcAcc * 100.0, svcF1), width: lineWidth)

        let svcResult = await BenchmarkRunner.run(
            name: "LinearSVC (Accuracy, F1)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = LinearSVC(C: 1.0, device: .cpu)
            try await m.fit(features: clsXTrain, targets: clsYTrain, learningRate: 0.1, epochs: 500)
            _ = try await m.predict(features: clsXTest)
        }
        results.append(svcResult)

        // 3.3 Random Forest Classifier
        let rf = try? RandomForestClassifier(nEstimators: 30, maxDepth: 5, criterion: .gini)
        _ = try? await rf?.fit(features: clsXTrain, targets: clsYTrain)
        let rfPred = (try? await rf?.predict(features: clsXTest)) ?? Array(repeating: 0, count: 200)
        let rfAcc = Metrics.accuracy(yTrue: yTestInt, yPred: rfPred)
        let rfF1  = Metrics.f1Score(yTrue: yTestInt, yPred: rfPred, label: 1)

        Self.printRow(String(format: "[ML Cls]   RandomForest (30 tr.): Accuracy=%.2f%%, F1=%.3f", rfAcc * 100.0, rfF1), width: lineWidth)

        let rfResult = await BenchmarkRunner.run(
            name: "RandomForest Classifier (Accuracy, F1)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = try RandomForestClassifier(nEstimators: 30, maxDepth: 5, criterion: .gini)
            try await m.fit(features: clsXTrain, targets: clsYTrain)
            _ = try await m.predict(features: clsXTest)
        }
        results.append(rfResult)

        // 3.4 HistGradientBoosting Classifier
        let hgbc = try? HistGradientBoostingClassifier(nEstimators: 30, learningRate: 0.1, maxDepth: 4, maxBins: 256)
        _ = try? await hgbc?.fit(features: clsXTrain, targets: clsYTrain)
        let hgbcPred = (try? await hgbc?.predict(features: clsXTest)) ?? Array(repeating: 0, count: 200)
        let hgbcAcc = Metrics.accuracy(yTrue: yTestInt, yPred: hgbcPred)
        let hgbcF1  = Metrics.f1Score(yTrue: yTestInt, yPred: hgbcPred, label: 1)

        Self.printRow(String(format: "[ML Cls]   HistGBDT Classifier  : Accuracy=%.2f%%, F1=%.3f", hgbcAcc * 100.0, hgbcF1), width: lineWidth)

        let hgbcResult = await BenchmarkRunner.run(
            name: "HistGBDT Classifier (Accuracy, F1)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = try HistGradientBoostingClassifier(nEstimators: 30, learningRate: 0.1, maxDepth: 4, maxBins: 256)
            try await m.fit(features: clsXTrain, targets: clsYTrain)
            _ = try await m.predict(features: clsXTest)
        }
        results.append(hgbcResult)

        // 3.5 Decision Tree Classifier
        let dtc = DecisionTreeClassifier(maxDepth: 5, minSamplesSplit: 2, criterion: .gini)
        _ = try? await dtc.fit(features: clsXTrain, targets: clsYTrain)
        let dtcPred = (try? await dtc.predict(features: clsXTest)) ?? Array(repeating: 0, count: 200)
        let dtcAcc = Metrics.accuracy(yTrue: yTestInt, yPred: dtcPred)
        let dtcF1  = Metrics.f1Score(yTrue: yTestInt, yPred: dtcPred, label: 1)

        Self.printRow(String(format: "[ML Cls]   DecisionTree (d=5)   : Accuracy=%.2f%%, F1=%.3f", dtcAcc * 100.0, dtcF1), width: lineWidth)

        let dtcResult = await BenchmarkRunner.run(
            name: "DecisionTree Classifier (Accuracy, F1)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = DecisionTreeClassifier(maxDepth: 5, minSamplesSplit: 2, criterion: .gini)
            try await m.fit(features: clsXTrain, targets: clsYTrain)
            _ = try await m.predict(features: clsXTest)
        }
        results.append(dtcResult)

        // 3.6 Naive Bayes (3-class text counts)
        let nbXTrain = (0..<800).map { i in (0..<50).map { j in Double((i + j) % 5) } }
        let nbYTrain = (0..<800).map { Double($0 % 3) }
        let nbXTest  = (0..<200).map { i in (0..<50).map { j in Double((i + j) % 5) } }
        let nbYTest  = (0..<200).map { Double($0 % 3) }
        let nbYTestInt = nbYTest.map { Int($0) }

        let nb = NaiveBayesClassifier(alpha: 1.0)
        _ = try? await nb.fit(features: nbXTrain, targets: nbYTrain)
        let nbPred = (try? await nb.predict(features: nbXTest)) ?? Array(repeating: 0, count: 200)
        let nbReport = Metrics.classificationReport(yTrue: nbYTestInt, yPred: nbPred)
        let nbAcc = nbReport.accuracy
        let nbF1  = nbReport.macroF1

        Self.printRow(String(format: "[NLP Cls]  NaiveBayes (3-class) : Accuracy=%.2f%%, Macro-F1=%.3f", nbAcc * 100.0, nbF1), width: lineWidth)

        let nbResult = await BenchmarkRunner.run(
            name: "NaiveBayes (Accuracy, Macro-F1)",
            module: module, warmup: 1, iterations: 3
        ) {
            let m = NaiveBayesClassifier(alpha: 1.0)
            try await m.fit(features: nbXTrain, targets: nbYTrain)
            _ = try await m.predict(features: nbXTest)
        }
        results.append(nbResult)

        // ══════════════════════════════════════════════════════════════════
        // ── 4. Unsupervised Clustering & SVD (PCA & K-Means) ──────────────
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("4. UNSUPERVISED LEARNING & SPECTRAL DECOMPOSITION", width: lineWidth)

        // 4.1 PCA on 5D Correlated Data (Accelerate LAPACK SVD)
        let nPca = 500
        var pcaData: [[Double]] = []
        for _ in 0..<nPca {
            let z1 = rng.nextDouble(in: -3.0...3.0)
            let z2 = rng.nextDouble(in: -2.0...2.0)
            let x0 = z1 * 2.0 + rng.nextDouble(in: -0.1...0.1)
            let x1 = z1 * 1.5 + z2 * 0.5 + rng.nextDouble(in: -0.1...0.1)
            let x2 = z2 * 3.0 + rng.nextDouble(in: -0.1...0.1)
            let x3 = -z1 + z2 * 0.8 + rng.nextDouble(in: -0.1...0.1)
            let x4 = rng.nextDouble(in: -0.5...0.5)
            pcaData.append([x0, x1, x2, x3, x4])
        }

        let pca = try? PCA(nComponents: 2, svdSolver: .full, device: .cpu)
        _ = try? await pca?.fit(pcaData)
        let evr = (await pca?.explainedVarianceRatio) ?? [0.0, 0.0]
        let evr1 = evr.count > 0 ? evr[0] : 0.0
        let evr2 = evr.count > 1 ? evr[1] : 0.0
        let totalEVR = evr1 + evr2

        Self.printRow(String(format: "[Cluster]  PCA SVD (5D → 2D)    : EVR=[%.4f, %.4f], Total=%.2f%%", evr1, evr2, totalEVR * 100.0), width: lineWidth)

        let pcaResult = await BenchmarkRunner.run(
            name: "PCA SVD Spectrum (EVR Parity)",
            module: module, warmup: 1, iterations: 3
        ) {
            let p = try PCA(nComponents: 2, svdSolver: .full, device: .cpu)
            try await p.fit(pcaData)
        }
        results.append(pcaResult)

        // 4.2 K-Means Clustering on 3 Synthetic Blobs
        let nKm = 600
        var kmData: [[Double]] = []
        for i in 0..<nKm {
            let clusterIdx = i % 3
            let cx = Double(clusterIdx) * 5.0
            let cy = Double(clusterIdx) * -3.0
            let x = cx + rng.nextDouble(in: -1.0...1.0)
            let y = cy + rng.nextDouble(in: -1.0...1.0)
            kmData.append([x, y])
        }

        let kmeans = try? KMeans(nClusters: 3, maxIterations: 100, seed: 42, device: .cpu)
        _ = try? await kmeans?.fit(features: kmData)
        let kmLabels = (try? await kmeans?.predict(features: kmData)) ?? []
        let cents = (await kmeans?.getCentroids()) ?? []
        let inertia = ClusteringMetrics.inertia(features: kmData, labels: kmLabels, centroids: cents)

        Self.printRow(String(format: "[Cluster]  K-Means (k=3, N=600) : Inertia (WCSS)=%.2f, Centroids=3", inertia), width: lineWidth)

        let kmResult = await BenchmarkRunner.run(
            name: "K-Means Convergence (WCSS Inertia)",
            module: module, warmup: 1, iterations: 3
        ) {
            let km = try KMeans(nClusters: 3, maxIterations: 100, seed: 42, device: .cpu)
            try await km.fit(features: kmData)
            _ = try await km.predict(features: kmData)
        }
        results.append(kmResult)

        // ══════════════════════════════════════════════════════════════════
        // ── 5. Feature Preprocessing (StandardScaler & MinMaxScaler) ──────
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("5. PREPROCESSING & IEEE 754 PRECISION (N = 1000)", width: lineWidth)

        let nScale = 1_000
        var scaleData: [[Double]] = []
        for _ in 0..<nScale {
            let f1 = rng.nextDouble(in: 10.0...50.0)
            let f2 = rng.nextDouble(in: -100.0...100.0)
            let f3 = rng.nextDouble(in: 0.0...1.0)
            scaleData.append([f1, f2, f3])
        }

        // StandardScaler
        var stdScaler = StandardScaler()
        _ = try? stdScaler.fit(scaleData)
        let stdTransformed = (try? stdScaler.transform(scaleData)) ?? []
        let meanCol0 = stdScaler.mean?[0] ?? 0.0
        let stdCol0  = stdScaler.std?[0] ?? 1.0
        let tMean0 = stdTransformed.isEmpty ? 0.0 : stdTransformed.map { $0[0] }.reduce(0, +) / Double(nScale)

        Self.printRow(String(format: "[Preproc]  StandardScaler       : col0 μ=%.4f, σ=%.4f (Post-scaled μ=%.2e)", meanCol0, stdCol0, tMean0), width: lineWidth)

        let stdResult = await BenchmarkRunner.run(
            name: "StandardScaler Parity (vDSP)",
            module: module, warmup: 1, iterations: 3
        ) {
            var s = StandardScaler()
            try s.fit(scaleData)
            _ = try s.transform(scaleData)
        }
        results.append(stdResult)

        // MinMaxScaler
        var mmScaler = MinMaxScaler(range: (0.0, 1.0))
        _ = try? mmScaler.fit(scaleData)
        let mmTransformed = (try? mmScaler.transform(scaleData)) ?? []
        let minCol0 = mmScaler.dataMin?[0] ?? 0.0
        let maxCol0 = mmScaler.dataMax?[0] ?? 1.0
        let tMin0 = mmTransformed.isEmpty ? 0.0 : mmTransformed.map { $0[0] }.min() ?? 0.0
        let tMax0 = mmTransformed.isEmpty ? 1.0 : mmTransformed.map { $0[0] }.max() ?? 1.0

        Self.printRow(String(format: "[Preproc]  MinMaxScaler         : col0 [%.2f, %.2f] → Transformed [%.4f, %.4f]", minCol0, maxCol0, tMin0, tMax0), width: lineWidth)

        let mmResult = await BenchmarkRunner.run(
            name: "MinMaxScaler Parity (vDSP)",
            module: module, warmup: 1, iterations: 3
        ) {
            var s = MinMaxScaler(range: (0.0, 1.0))
            try s.fit(scaleData)
            _ = try s.transform(scaleData)
        }
        results.append(mmResult)

        // ══════════════════════════════════════════════════════════════════
        // ── 6. Inferential Statistics & Hypothesis Tests ──────────────────
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("6. INFERENTIAL STATISTICS & HYPOTHESIS TESTING", width: lineWidth)

        let samp1 = (0..<1000).map { _ in rng.nextDouble(in: 4.5...5.5) }
        let samp2 = (0..<1000).map { _ in rng.nextDouble(in: 4.3...5.7) }
        let samp3 = (0..<1000).map { _ in rng.nextDouble(in: 4.8...5.8) }

        // Welch's t-test
        let welch = try? Stats.tTest(sample1: samp1, sample2: samp2, equalVariances: false)
        let wT = welch?.statistic ?? 0.0
        let wP = welch?.pValue ?? 1.0
        let wDf = welch?.degreesOfFreedom ?? 0.0
        Self.printRow(String(format: "[Stats]    Welch's t-test       : t=%.4f, p=%.6e, df=%.1f", wT, wP, wDf), width: lineWidth)

        // Student's t-test
        let student = try? Stats.tTest(sample1: samp1, sample2: samp2, equalVariances: true)
        let sT = student?.statistic ?? 0.0
        let sP = student?.pValue ?? 1.0
        Self.printRow(String(format: "[Stats]    Student's t-test     : t=%.4f, p=%.6e", sT, sP), width: lineWidth)

        // Paired t-test
        let paired = try? Stats.pairedTTest(before: samp1, after: samp2)
        let pT = paired?.statistic ?? 0.0
        let pP = paired?.pValue ?? 1.0
        Self.printRow(String(format: "[Stats]    Paired t-test        : t=%.4f, p=%.6e", pT, pP), width: lineWidth)

        // One-way ANOVA
        let anova = try? Stats.oneWayANOVA(groups: [samp1, samp2, samp3])
        let aF = anova?.fStatistic ?? 0.0
        let aP = anova?.pValue ?? 1.0
        Self.printRow(String(format: "[Stats]    One-Way ANOVA        : F=%.4f, p=%.6e", aF, aP), width: lineWidth)

        // Pearson & Spearman Correlation
        let rVal = (try? Stats.pearsonCorrelation(samp1, samp2)) ?? 0.0
        let rhoVal = (try? Stats.spearmanCorrelation(samp1, samp2)) ?? 0.0
        Self.printRow(String(format: "[Stats]    Correlation          : Pearson r=%.5f, Spearman ρ=%.5f", rVal, rhoVal), width: lineWidth)

        let statsResult = await BenchmarkRunner.run(
            name: "Hypothesis Tests & Correlation Suite",
            module: module, warmup: 1, iterations: 3
        ) {
            _ = try Stats.tTest(sample1: samp1, sample2: samp2, equalVariances: false)
            _ = try Stats.oneWayANOVA(groups: [samp1, samp2, samp3])
            _ = try Stats.pearsonCorrelation(samp1, samp2)
            _ = try Stats.spearmanCorrelation(samp1, samp2)
        }
        results.append(statsResult)

        // ══════════════════════════════════════════════════════════════════
        // ── 7. NLP: Sentiment Analysis & Text Processing ──────────────────
        // ══════════════════════════════════════════════════════════════════
        Self.printSection("7. NLP: VADER SENTIMENT POLARITY SCORING", width: lineWidth)

        let vader = VADERSentimentAnalyzer()
        let sentPos = "SwiftSci 3.8.1 is incredibly fast, robust and accurate!"
        let sentNeg = "The algorithm failed completely with disastrous and horrible errors."
        let sentNeu = "The dataset contains standard numerical observations and measurements."

        let scPos = vader.polarityScores(text: sentPos)
        let scNeg = vader.polarityScores(text: sentNeg)
        let scNeu = vader.polarityScores(text: sentNeu)

        Self.printRow(String(format: "[NLP]      VADER (Positive)     : Compound=%.4f, Pos=%.3f, Neg=%.3f", scPos.compound, scPos.pos, scPos.neg), width: lineWidth)
        Self.printRow(String(format: "[NLP]      VADER (Negative)     : Compound=%.4f, Pos=%.3f, Neg=%.3f", scNeg.compound, scNeg.pos, scNeg.neg), width: lineWidth)
        Self.printRow(String(format: "[NLP]      VADER (Neutral)      : Compound=%.4f, Neu=%.3f", scNeu.compound, scNeu.neu), width: lineWidth)

        print("  └" + border + "┘")

        let vaderResult = await BenchmarkRunner.run(
            name: "VADER Sentiment Polarity Analysis",
            module: module, warmup: 1, iterations: 3
        ) {
            _ = vader.polarityScores(text: sentPos)
            _ = vader.polarityScores(text: sentNeg)
            _ = vader.polarityScores(text: sentNeu)
        }
        results.append(vaderResult)

        return results
    }
}
