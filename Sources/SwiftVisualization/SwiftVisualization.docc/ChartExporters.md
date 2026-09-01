# Native SwiftUI Charting & Interactive Plotly Exporters

Render charts with zero external dependencies using `SwiftSciChartView` in SwiftUI, or export interactive HTML reports with `ChartExporter`.

## Overview

`SwiftVisualization` provides two complementary visualization engines:
1. **Native SwiftUI Engine (`SwiftSciChartView`)**: Zero-dependency `@MainActor` SwiftUI view rendering line charts, bar charts, scatter plots, histograms, and 2D correlation heatmaps on a high-performance hardware-accelerated `Canvas`.
2. **Interactive HTML Exporter (`ChartExporter`)**: Standalone browser-ready HTML diagnostic reports (ROC Curves with trapezoidal Area Under Curve / AUC computation, Feature Importances, Confusion Matrices).

---

## 1. Native SwiftUI Charting (`SwiftSciChartView`)

Embed responsive, animated charts in iOS and macOS SwiftUI user interfaces:

```swift
import SwiftUI
import SwiftVisualization

struct ModelDiagnosticsView: View {
    let lossHistory: [Double] = [0.85, 0.62, 0.45, 0.38, 0.29, 0.22, 0.18, 0.15]
    let epochLabels: [String] = (1...8).map { "Ep \($0)" }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Training Loss Convergence")
                .font(.headline)
            
            // 1. Line Chart with custom options
            SwiftSciChartView(
                type: .line,
                data: lossHistory,
                labels: epochLabels,
                options: ChartOptions(
                    accentColor: .blue,
                    showAxes: true,
                    lineWidth: 3.0,
                    customYAxisLabel: "Loss (Cross-Entropy)"
                )
            )
            .frame(height: 250)
            .padding()
            .background(Color(.windowBackgroundColor).opacity(0.5))
            .cornerRadius(12)
        }
        .padding()
    }
}
```

---

## 2. 2D Correlation Heatmaps in SwiftUI

Render 2D covariance and correlation matrices with automatic color gradients:

```swift
import SwiftUI
import SwiftVisualization

struct CorrelationMatrixView: View {
    // 3x3 Correlation Grid
    let matrix2D: [[Double]] = [
        [ 1.00,  0.82, -0.45],
        [ 0.82,  1.00, -0.30],
        [-0.45, -0.30,  1.00]
    ]
    let featureNames = ["Revenue", "Profit", "Expenses"]
    
    var body: some View {
        SwiftSciChartView(
            type: .heatmap2D(matrix2D),
            data: [],
            labels: featureNames,
            options: ChartOptions(showAxes: true)
        )
        .frame(width: 320, height: 320)
    }
}
```

---

## 3. Interactive Plotly HTML Reports with AUC

Generate standalone HTML diagnostics viewable in any modern browser:

```swift
import Foundation
import SwiftVisualization

// 1. Export ROC Curve with exact trapezoidal AUC computation
let yTrue = [0, 0, 1, 1, 0, 1]
let yScores = [0.1, 0.35, 0.8, 0.92, 0.2, 0.75]

let rocHTML = ChartExporter.plotROCCurve(yScores: yScores, yTrue: yTrue)
let outputURL = URL(fileURLWithPath: "roc_curve_report.html")
try rocHTML.write(to: outputURL, atomically: true, encoding: .utf8)

print("Saved interactive ROC report with trapezoidal AUC title.")

// 2. Export Feature Importance bar chart
let featureImportances = ["age": 0.35, "income": 0.45, "education": 0.20]
let featHTML = ChartExporter.plotFeatureImportances(featureImportances)
```
