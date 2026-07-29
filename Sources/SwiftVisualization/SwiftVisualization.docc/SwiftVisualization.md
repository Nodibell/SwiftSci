# ``SwiftVisualization``

Standalone Interactive HTML Chart Exporters.

## Overview

`SwiftVisualization` generates standalone, interactive HTML chart diagnostic visualizers viewable directly in web browsers.

### Key Capabilities

- **Correlation Heatmaps**: Interactive correlation matrix heatmaps exported directly from `DataFrame`.
- **Evaluation Visuals**: `ROCCurve` plots with AUC metrics and `ConfusionMatrix` displays.
- **Feature Diagnostics**: `FeatureImportance` bar charts and partial dependence visualizers.

### Example Usage

```swift
import SwiftVisualization

let html = try df.exportCorrelationHeatmapHTML()
```

## Topics

### Guides & Tutorials
- <doc:ChartExporters>
