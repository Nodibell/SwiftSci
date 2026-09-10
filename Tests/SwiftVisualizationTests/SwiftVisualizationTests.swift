import Testing
import SwiftDataFrame
@testable import SwiftVisualization

@Suite("SwiftVisualization Tests")
struct SwiftVisualizationTests {
    
    @Test("ChartExporter generates HTML strings with Plotly script tags")
    func testChartExporterHTML() throws {
        let colA = TypedColumn<Double>(name: "A", values: [1.0, 2.0, 3.0])
        let colB = TypedColumn<Double>(name: "B", values: [2.0, 4.0, 6.0])
        let df = try DataFrame(columns: [colA, colB])
        
        let heatmapHTML = try ChartExporter.plotCorrelationHeatmap(df: df)
        #expect(heatmapHTML.contains("<!DOCTYPE html>"))
        #expect(heatmapHTML.contains("Plotly.newPlot"))
        
        let rocHTML = ChartExporter.plotROCCurve(yTrue: [1, 0, 1], yScores: [0.9, 0.1, 0.8])
        #expect(rocHTML.contains("ROC Curve"))
        
        let featHTML = ChartExporter.plotFeatureImportances(featureNames: ["F1", "F2"], importances: [0.7, 0.3])
        #expect(featHTML.contains("Feature Importances"))
        
        let cmHTML = ChartExporter.plotConfusionMatrix(matrix: [[10, 2], [1, 15]], labels: ["Class 0", "Class 1"])
        #expect(cmHTML.contains("Confusion Matrix"))
    }

    @Test("ChartExporter computes real AUC for perfect ROC classifier")
    func testROCAUCComputation() {
        let yTrue = [1, 1, 0, 0]
        let yScores = [0.9, 0.8, 0.2, 0.1]
        let html = ChartExporter.plotROCCurve(yTrue: yTrue, yScores: yScores)
        #expect(html.contains("AUC = 1.0000"))
    }

    @Test("ChartExporter sanitizes HTML and escapes quotes to prevent XSS and syntax breakage")
    func testXSSSanitization() throws {
        let maliciousTitle = "</title><script>alert('xss')</script>"
        let maliciousCol = TypedColumn<Double>(name: "bad\"<col>", values: [1.0, 2.0])
        let df = try DataFrame(columns: [maliciousCol])

        let heatmap = try ChartExporter.plotCorrelationHeatmap(df: df, title: maliciousTitle)
        // Title tag must not contain unescaped closing tag or script injection
        #expect(heatmap.contains("&lt;/title&gt;&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;"))
        #expect(!heatmap.contains("<title></title><script>"))

        // Feature importances with quotes and script injection
        let featNames = ["col\"with'quotes", "<script>bad()</script>"]
        let featHTML = ChartExporter.plotFeatureImportances(featureNames: featNames, importances: [0.5, 0.5], title: maliciousTitle)
        #expect(!featHTML.contains("<title></title><script>"))
        #expect(featHTML.contains("col\\\"with'quotes"))

        // Confusion matrix with payload
        let cmHTML = ChartExporter.plotConfusionMatrix(matrix: [[1, 0], [0, 1]], labels: featNames, title: maliciousTitle)
        #expect(!cmHTML.contains("<title></title><script>"))
        #expect(cmHTML.contains("col\\\"with'quotes"))
    }
}
