// Compiler reproduction of the original helper, independent of SwiftSci.
import Darwin
func sortedIndices<T: Comparable>(_ values: [T?], ascending: Bool) -> [Int] {
    var indices = Array(values.indices)
    sortIndices(&indices, ascending: ascending) { values[$0] }
    return indices
}
private func sortIndices<C: Comparable>(_ indices: inout [Int], ascending: Bool, key: (Int) -> C?) {
    if ascending {
        indices.sort { i, j in
            switch (key(i), key(j)) {
            case (nil, nil): return false
            case (nil, _):   return false
            case (_, nil):   return true
            case let (l?, r?): return l < r
            }
        }
    } else {
        indices.sort { i, j in
            switch (key(i), key(j)) {
            case (nil, nil): return false
            case (nil, _):   return false
            case (_, nil):   return true
            case let (l?, r?): return l > r
            }
        }
    }
}

let values: [Double?] = [2, nil, 1]
var failures = 0
for ascending in [true, false] {
    let actual = sortedIndices(values, ascending: ascending)
    let expected = ascending ? [2, 0, 1] : [0, 2, 1]
    print("ascending=\(ascending), indices=\(actual), expected=\(expected)")
    if actual != expected { failures += 1 }
}
exit(failures == 0 ? 0 : 1)
