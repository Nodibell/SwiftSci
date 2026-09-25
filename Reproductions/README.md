# Descending sort compiler reproduction

`DescendingNullSort.swift` preserves the original optional-value sorting helper in a standalone program. It does not import SwiftSci or any third-party package. It intentionally retains the compiler trigger after the library workaround.

On Apple Swift 6.4, swiftlang-6.4.0.34.1, targeting arm64 macOS with Xcode 27.0:

| Compilation | Ascending result | Descending result | Exit |
|---|---|---|---|
| `-Onone` | `[2, 0, 1]` | `[0, 2, 1]` | 0 |
| `-O` | `[2, 0, 1]` | `[2, 0, 1]` | 1 |

The input is `[2, nil, 1]`. Both directions should keep the missing value last. Descending order must put 2 before 1.

## Run

From the repository root:

```sh
swiftc -swift-version 6 -Onone Reproductions/DescendingNullSort.swift -o /tmp/swiftsci-sort-debug
/tmp/swiftsci-sort-debug
swiftc -swift-version 6 -O Reproductions/DescendingNullSort.swift -o /tmp/swiftsci-sort-release
/tmp/swiftsci-sort-release
```

The optimized program exits 1 on the affected toolchain. It prints actual and expected indices rather than crashing. Other compiler versions may behave differently.

## Isolation

The original upstream tests and a new three-element regression fail in Release and pass in Debug. The standalone program reproduces the failure without the package, its dependencies, XCTest, or Swift Testing. A standalone variant with one comparison closure passes with optimization enabled.

Together these observations point to a compiler optimization defect triggered by separate generic comparison closures. They do not identify the particular compiler pass or establish which other Swift versions are affected. No compiler issue has been filed as part of this investigation.

## Library workaround

The library now sorts with one closure and chooses `<` or `>` inside its non-missing comparison case. Missing-value ordering remains independent of direction. The workaround preserves optimization and the existing API.

`Tests/SwiftDataFrameTests/DescendingNullSortTests.swift` checks the numeric reproduction, String and Bool callers, and dataframe row association. All three tests were run and failed before the production change. All three pass with the workaround in Release mode. The original upstream assertions remain unchanged.
