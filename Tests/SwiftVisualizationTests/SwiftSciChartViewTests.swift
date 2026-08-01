#if canImport(SwiftUI)
import Testing
import SwiftUI
@testable import SwiftVisualization

@Suite("SwiftSciChartView Native Rendering Tests")
@MainActor
struct SwiftSciChartViewTests {

    @Test("SwiftSciChartView line chart initialization")
    func testLineChartInitialization() {
        let series = ChartSeries(
            id: "series_1",
            x: [1.0, 2.0, 3.0, 4.0],
            y: [10.0, 25.0, 15.0, 30.0],
            color: .blue
        )

        let view = SwiftSciChartView(title: "Sample Line Chart", type: .line, series: [series])

        #expect(view.title == "Sample Line Chart")
        #expect(view.type == .line)
        #expect(view.series.count == 1)
        #expect(view.series[0].x.count == 4)
    }

    @Test("SwiftSciChartView bar chart initialization")
    func testBarChartInitialization() {
        let series = ChartSeries(
            id: "bar_1",
            x: [0.0, 1.0, 2.0],
            y: [5.0, 10.0, 15.0],
            color: .green
        )

        let view = SwiftSciChartView(title: "Bar Chart", type: .bar, series: [series])

        #expect(view.type == .bar)
        #expect(view.series[0].y == [5.0, 10.0, 15.0])
    }
}
#endif
