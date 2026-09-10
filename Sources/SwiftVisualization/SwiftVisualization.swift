import Foundation
@_exported import SwiftDataFrame
@_exported import SwiftStats

/// Exporters for generating interactive standalone HTML/SVG chart visualizations.
public enum ChartExporter {
    
    /// Generates HTML file with an interactive Correlation Heatmap.
    /// - Parameters:
    ///   - df: <#description#>
    ///   - title: <#description#>
    /// - Throws: <#error description#>
    /// - Returns: <#description#>
    public static func plotCorrelationHeatmap(df: DataFrame, title: String = "Correlation Heatmap") throws -> String {
        let numericCols = df.columns.compactMap { $0 as? TypedColumn<Double> }
        let names = numericCols.map { $0.name }
        guard !names.isEmpty else { return "<div>No numeric columns found.</div>" }
        
        var matrix = [[Double]]()
        for c1 in numericCols {
            var row = [Double]()
            let v1 = c1.values.compactMap { $0 }
            for c2 in numericCols {
                let v2 = c2.values.compactMap { $0 }
                let corr = (v1.count == v2.count && !v1.isEmpty) ? (try? Stats.pearsonCorrelation(v1, v2)) ?? 0.0 : 0.0
                row.append(corr)
            }
            matrix.append(row)
        }
        
        let zJSON = "[" + matrix.map { "[" + $0.map { String(format: "%.3f", $0) }.joined(separator: ",") + "]" }.joined(separator: ",") + "]"
        let xJSON = jsonStrings(names)
        let yJSON = xJSON
        let safeTitle = escapeHTML(title)
        let titleJSON = jsonString(title)
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(safeTitle)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:600px;"></div>
          <script>
            var data = [{
              z: \(zJSON),
              x: \(xJSON),
              y: \(yJSON),
              type: 'heatmap',
              colorscale: 'Viridis'
            }];
            var layout = { title: \(titleJSON) };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
    
    /// Generates HTML file with an interactive ROC Curve.
    ///
    /// ## Security & Sanitization
    /// All titles and user strings are properly escaped via HTML entities and `JSONEncoder` serialization,
    /// preventing Cross-Site Scripting (XSS) and JavaScript syntax breakages.
    /// - Parameters:
    ///   - yTrue: <#description#>
    ///   - yScores: <#description#>
    ///   - title: <#description#>
    /// - Returns: <#description#>
    public static func plotROCCurve(yTrue: [Int], yScores: [Double], title: String = "ROC Curve") -> String {
        var fpr: [Double] = [0.0, 1.0]
        var tpr: [Double] = [0.0, 1.0]
        var auc = 0.5

        if !yTrue.isEmpty && yTrue.count == yScores.count {
            let paired = zip(yScores, yTrue).sorted { $0.0 > $1.0 }
            let totalPositives = Double(yTrue.filter { $0 == 1 }.count)
            let totalNegatives = Double(yTrue.count) - totalPositives

            if totalPositives > 0 && totalNegatives > 0 {
                var computedFPR: [Double] = [0.0]
                var computedTPR: [Double] = [0.0]
                var tp = 0.0
                var fp = 0.0

                for (_, label) in paired {
                    if label == 1 {
                        tp += 1.0
                    } else {
                        fp += 1.0
                    }
                    computedTPR.append(tp / totalPositives)
                    computedFPR.append(fp / totalNegatives)
                }

                // Trapezoidal rule for AUC
                var computedAUC = 0.0
                for i in 1..<computedFPR.count {
                    let dx = computedFPR[i] - computedFPR[i - 1]
                    let avgY = (computedTPR[i] + computedTPR[i - 1]) / 2.0
                    computedAUC += dx * avgY
                }

                fpr = computedFPR
                tpr = computedTPR
                auc = computedAUC
            }
        }

        let displayTitle = String(format: "%@ (AUC = %.4f)", title, auc)
        let safeDisplayTitle = escapeHTML(displayTitle)
        let displayTitleJSON = jsonString(displayTitle)
        let xJSON = "[" + fpr.map { String($0) }.joined(separator: ",") + "]"
        let yJSON = "[" + tpr.map { String($0) }.joined(separator: ",") + "]"
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(safeDisplayTitle)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:500px;"></div>
          <script>
            var data = [{
              x: \(xJSON),
              y: \(yJSON),
              mode: 'lines+markers',
              name: 'Model ROC',
              line: {color: '#1f77b4', width: 3}
            }, {
              x: [0, 1],
              y: [0, 1],
              mode: 'lines',
              name: 'Random Chance',
              line: {dash: 'dash', color: '#7f7f7f'}
            }];
            var layout = { title: \(displayTitleJSON), xaxis: {title: 'False Positive Rate'}, yaxis: {title: 'True Positive Rate'} };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
    
    /// Generates HTML file with Feature Importances horizontal bar chart.
    ///
    /// ## Security & Sanitization
    /// Feature names and chart titles are safely encoded via `JSONEncoder` and HTML entity escaping.
    /// - Parameters:
    ///   - featureNames: <#description#>
    ///   - importances: <#description#>
    ///   - title: <#description#>
    /// - Returns: <#description#>
    public static func plotFeatureImportances(featureNames: [String], importances: [Double], title: String = "Feature Importances") -> String {
        let xJSON = "[" + importances.map { String(format: "%.4f", $0) }.joined(separator: ",") + "]"
        let yJSON = jsonStrings(featureNames)
        let safeTitle = escapeHTML(title)
        let titleJSON = jsonString(title)
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(safeTitle)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:500px;"></div>
          <script>
            var data = [{
              type: 'bar',
              x: \(xJSON),
              y: \(yJSON),
              orientation: 'h',
              marker: {color: '#2ca02c'}
            }];
            var layout = { title: \(titleJSON), xaxis: {title: 'Importance Score'} };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }
    
    /// Generates HTML file with Confusion Matrix heatmap.
    ///
    /// ## Security & Sanitization
    /// Labels and chart titles are safely encoded via `JSONEncoder` and HTML entity escaping.
    /// - Parameters:
    ///   - matrix: <#description#>
    ///   - labels: <#description#>
    ///   - title: <#description#>
    /// - Returns: <#description#>
    public static func plotConfusionMatrix(matrix: [[Int]], labels: [String], title: String = "Confusion Matrix") -> String {
        let zJSON = "[" + matrix.map { "[" + $0.map { String($0) }.joined(separator: ",") + "]" }.joined(separator: ",") + "]"
        let labelsJSON = jsonStrings(labels)
        let safeTitle = escapeHTML(title)
        let titleJSON = jsonString(title)
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>\(safeTitle)</title>
          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        </head>
        <body>
          <div id="chart" style="width:100%;height:500px;"></div>
          <script>
            var data = [{
              z: \(zJSON),
              x: \(labelsJSON),
              y: \(labelsJSON),
              type: 'heatmap',
              colorscale: 'Blues'
            }];
            var layout = { title: \(titleJSON), xaxis: {title: 'Predicted'}, yaxis: {title: 'Actual'} };
            Plotly.newPlot('chart', data, layout);
          </script>
        </body>
        </html>
        """
    }

    // MARK: - Sanitization Helpers

    private static func escapeHTML(_ string: String) -> String {
        string
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
    }

    private static func jsonString(_ string: String) -> String {
        guard let data = try? JSONEncoder().encode(string),
              let str = String(data: data, encoding: .utf8) else {
            return "\"\(string)\""
        }
        return str
    }

    private static func jsonStrings(_ strings: [String]) -> String {
        guard let data = try? JSONEncoder().encode(strings),
              let str = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return str
    }
}
