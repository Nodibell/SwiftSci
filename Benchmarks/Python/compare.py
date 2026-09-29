#!/usr/bin/env python3
"""Retired legacy comparison entry point."""

import sys


def main():
    print(
        "Legacy automatic comparison is retired. Legacy JSON does not establish "
        "matched workload identities or validated outputs. Generate standardized "
        "runs, then use: python3 Benchmarks/Tools/bench.py compare "
        "BASELINE_RUN_DIR CANDIDATE_RUN_DIR. See Benchmarks/README.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
