/// SwiftStats — statistical analysis module backed by Accelerate vDSP + LAPACK.
///
/// All functions are grouped as static methods on the `Stats` enum namespace.
/// Usage:
/// ```swift
/// let m = try Stats.mean([1.0, 2.0, 3.0])
/// let r = try Stats.tTest(sample: data, populationMean: 0)
/// print(r.isSignificant)
/// ```
public enum Stats {}
