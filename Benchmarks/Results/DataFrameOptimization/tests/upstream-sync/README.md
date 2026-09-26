# Validation after upstream merge

Tested merge commit: `a2853b1a918a005dd95ac8649d6cedc030b902de`.
Merged upstream main: `f6320c1e9f852e49c44569119e362b00b4d550b9`.
Date: September 26, 2026.

| Full suite | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Debug | 891 | 0 | 0 |
| Release | 891 | 0 | 0 |

Both native arm64 Xcode runs include MLX-dependent targets. The configuration remains Xcode 27.0, Swift 6.4, macOS 27.0 on Apple M4 Max. Release retains optimization with testability enabled. Dependency pins and CI configuration were unchanged by the merge.

[Debug summary](debug.json) and [Release summary](release.json) preserve the result counts, timings and execution details. Device identifiers are omitted. Original and published summary hashes appear in the evidence manifest.

The subsequent evidence-only commit adds these summaries and this note. It changes no production source. The existing performance measurements remain attributed to their original recorded commits; this validation did not rerun benchmarks or establish remote CI status.
