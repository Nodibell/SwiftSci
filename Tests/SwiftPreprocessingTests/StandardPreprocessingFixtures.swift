func preprocessingTrainingColumns(width: Int) -> [[Double]] {
    (0..<width).map { column -> [Double] in
        (0..<37).map { row -> Double in
            if (row + column) % 11 == 0 { return .nan }
            let seed: Int = (row * 13 + column) % 47
            let value: Double = Double(seed) / 8.0
            return value
        }
    }
}

func preprocessingIncomingColumns(width: Int, rows: Int) -> [[Double]] {
    (0..<width).map { column -> [Double] in
        (0..<rows).map { row -> Double in
            let kind: Int = (row + column) % 17
            switch kind {
            case 0: return .nan
            case 1: return -0.0
            case 2: return .infinity
            case 3: return -.infinity
            default:
                let seed: Int = (row * 7 + column * 3) % 97
                let value: Double = Double(seed) / 16.0
                return value
            }
        }
    }
}
