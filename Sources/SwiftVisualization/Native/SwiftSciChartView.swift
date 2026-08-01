#if canImport(SwiftUI)
import SwiftUI

/// Chart type for native `SwiftSciChartView` rendering.
public enum ChartType: Sendable {
    case line
    case bar
    case heatmap
}

/// Data series for native SwiftUI rendering.
public struct ChartSeries: Sendable, Identifiable {
    public let id: String
    public let x: [Double]
    public let y: [Double]
    public let color: Color

    public init(id: String = UUID().uuidString, x: [Double], y: [Double], color: Color = .blue) {
        self.id = id
        self.x = x
        self.y = y
        self.color = color
    }
}

/// A native SwiftUI `Canvas` chart component for SwiftSci datasets.
@MainActor
public struct SwiftSciChartView: View {
    public let title: String
    public let type: ChartType
    public let series: [ChartSeries]

    public init(title: String, type: ChartType = .line, series: [ChartSeries]) {
        self.title = title
        self.type = type
        self.series = series
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)

            Canvas { context, size in
                guard !series.isEmpty else { return }

                let padding: CGFloat = 30.0
                let width = size.width - 2 * padding
                let height = size.height - 2 * padding

                let allX = series.flatMap { $0.x }
                let allY = series.flatMap { $0.y }

                guard let minX = allX.min(), let maxX = allX.max(),
                      let minY = allY.min(), let maxY = allY.max() else { return }

                let rangeX = maxX > minX ? maxX - minX : 1.0
                let rangeY = maxY > minY ? maxY - minY : 1.0

                // Draw axes
                var axisPath = Path()
                axisPath.move(to: CGPoint(x: padding, y: padding))
                axisPath.addLine(to: CGPoint(x: padding, y: size.height - padding))
                axisPath.addLine(to: CGPoint(x: size.width - padding, y: size.height - padding))
                context.stroke(axisPath, with: .color(.gray), lineWidth: 1)

                // Render series based on type
                for s in series {
                    guard s.x.count == s.y.count, !s.x.isEmpty else { continue }

                    switch type {
                    case .line:
                        var linePath = Path()
                        for (idx, (xVal, yVal)) in zip(s.x, s.y).enumerated() {
                            let px = padding + CGFloat((xVal - minX) / rangeX) * width
                            let py = size.height - padding - CGFloat((yVal - minY) / rangeY) * height

                            if idx == 0 {
                                linePath.move(to: CGPoint(x: px, y: py))
                            } else {
                                linePath.addLine(to: CGPoint(x: px, y: py))
                            }
                        }
                        context.stroke(linePath, with: .color(s.color), lineWidth: 2)

                    case .bar:
                        let barWidth = width / CGFloat(s.x.count) * 0.8
                        for (xVal, yVal) in zip(s.x, s.y) {
                            let px = padding + CGFloat((xVal - minX) / rangeX) * width
                            let py = size.height - padding - CGFloat((yVal - minY) / rangeY) * height
                            let rect = CGRect(x: px - barWidth / 2, y: py, width: barWidth, height: (size.height - padding) - py)
                            context.fill(Path(rect), with: .color(s.color))
                        }

                    case .heatmap:
                        let count = CGFloat(s.x.count)
                        let side = min(width, height) / count
                        for (idx, yVal) in s.y.enumerated() {
                            let normVal = CGFloat((yVal - minY) / rangeY)
                            let rect = CGRect(x: padding + CGFloat(idx) * side, y: padding, width: side, height: side)
                            context.fill(Path(rect), with: .color(s.color.opacity(Double(normVal))))
                        }
                    }
                }
            }
            .frame(minHeight: 200)
        }
        .padding()
    }
}
#endif
