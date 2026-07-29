# Interactive HTML Chart Exporters

Generate standalone interactive HTML visualizations for heatmaps, ROC curves, confusion matrices, and feature importances.

## Overview

Export visual diagnostic plots directly to HTML files viewable in any browser.

### 1. Correlation Heatmap Export

```swift
import SwiftVisualization
import SwiftDataFrame

let df = try await DataFrame(csv: fileURL)
let html = try df.exportCorrelationHeatmapHTML()

try html.write(toFile: "/tmp/correlation.html", atomically: true, encoding: .utf8)
```

### 2. ROC Curve Visualization

```swift
let rocHTML = Visualization.exportROCCurve(yTrue: yTest, yProbs: probs)
```
