import Testing
import SwiftDataFrame
@testable import SwiftVisualization

@Suite("SwiftVisualization EDA Charts Tests")
struct EDAChartsTests {
    
    @Test("plotClassDistribution produces interactive HTML with counts and percentages")
    func testPlotClassDistributionCounts() {
        let counts = ["business": 40, "tech": 30, "sport": 30]
        let html = ChartExporter.plotClassDistribution(counts: counts, title: "News Topics")
        
        #expect(html.contains("<!DOCTYPE html>"))
        #expect(html.contains("Plotly.newPlot"))
        #expect(html.contains("News Topics"))
        #expect(html.contains("business"))
        #expect(html.contains("40.0%"))
        #expect(html.contains("30.0%"))
        #expect(html.contains("type: 'bar'"))
    }
    
    @Test("DataFrame.plotClassDistribution extracts counts from column")
    func testDataFramePlotClassDistribution() throws {
        let labels = ["cat", "cat", "dog", "cat", "bird", "dog"]
        let df = try DataFrame(columns: [
            TypedColumn<String>(name: "category", values: labels)
        ])
        
        let html = df.plotClassDistribution(targetColumn: "category", title: "Animal Classes")
        #expect(html.contains("cat"))
        #expect(html.contains("dog"))
        #expect(html.contains("bird"))
        #expect(html.contains("Animal Classes"))
    }
    
    @Test("plotBoxPlot generates valid plotly box traces")
    func testPlotBoxPlot() throws {
        let series = [
            "Group A": [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0],
            "Group B": [5.0, 5.5, 6.0, 6.2, 6.8, 7.0]
        ]
        let html = ChartExporter.plotBoxPlot(series: series, title: "Feature Outliers")
        
        #expect(html.contains("<!DOCTYPE html>"))
        #expect(html.contains("type: 'box'"))
        #expect(html.contains("boxpoints: 'outliers'"))
        #expect(html.contains("Group A"))
        #expect(html.contains("Group B"))
        #expect(html.contains("Feature Outliers"))
    }
    
    @Test("DataFrame.plotBoxPlot extracts numerical columns")
    func testDataFramePlotBoxPlot() throws {
        let df = try DataFrame(columns: [
            TypedColumn<Double>(name: "salary", values: [50000, 60000, 75000, 120000]),
            TypedColumn<Double>(name: "age", values: [25, 30, 35, 50]),
            TypedColumn<String>(name: "department", values: ["Eng", "Sales", "HR", "Eng"])
        ])
        
        let html = df.plotBoxPlot(title: "Employee Stats")
        #expect(html.contains("salary"))
        #expect(html.contains("age"))
        // Department is categorical, shouldn't be in numeric box traces
        #expect(!html.contains("department"))
    }
    
    @Test("EDA charts prevent XSS injections")
    func testEDAXSSSanitization() {
        let malicious = "<script>alert('xss')</script>"
        let html1 = ChartExporter.plotClassDistribution(counts: [malicious: 10], title: malicious)
        #expect(!html1.contains("<title></title><script>"))
        #expect(html1.contains("&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;"))
        
        let html2 = ChartExporter.plotBoxPlot(series: [malicious: [1.0, 2.0]], title: malicious)
        #expect(!html2.contains("<title></title><script>"))
    }
}
