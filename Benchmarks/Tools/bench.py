#!/usr/bin/env python3
"""Entry point for preparation, certified runs and strict comparisons."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import uuid
from contracts import ContractError, digest, load_profile, read_json, require
from datasets import load_manifest, prepare
from runner import build, plan, run
from reporting import comparison, summarize, audit


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "run", "certify"]:
        p = commands.add_parser(name)
        p.add_argument(
            "--profile", default="certification" if name == "certify" else "smoke"
        )
        if name != "prepare":
            p.add_argument("--engines", default="swiftsci")
            p.add_argument("--swift-worker")
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--output", type=Path)
    p = commands.add_parser("build")
    p.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / "Library/Caches/SwiftSci/standardized-benchmarks",
    )
    p.add_argument(
        "--packages",
        type=Path,
        default=Path.home() / "Library/Caches/SwiftSci/xcode-packages",
    )
    p = commands.add_parser("audit")
    p.add_argument("run_directory", type=Path)
    p = commands.add_parser("report")
    p.add_argument("run_directory", type=Path)
    p = commands.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        evidence = read_json(args.run_directory / "run.json")
        certificate = read_json(args.run_directory / "certificate.json")
        require(
            certificate["run_sha256"]
            == digest((args.run_directory / "run.json").read_bytes()),
            "Run certificate checksum mismatch",
        )
        require(
            certificate["status"] == "passed"
            and certificate["contract_hash"] == evidence["plan"]["contract_hash"],
            "Certificate mismatch",
        )
        audit(evidence)
        print("PASS: complete validated run with matching certificate")
        return 0
    if args.command == "report":
        print(
            json.dumps(summarize(read_json(args.run_directory / "run.json")), indent=2)
        )
        return 0
    if args.command == "compare":
        print(
            json.dumps(
                comparison(
                    read_json(args.baseline / "run.json"),
                    read_json(args.candidate / "run.json"),
                ),
                indent=2,
            )
        )
        return 0
    lock = root / "Benchmarks/Runs/.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError("Another benchmark or build is active")
        if args.command == "build":
            print(build(root, args.cache / "derived", args.packages))
            return 0
        profile = load_profile(root, args.profile)
        if args.command == "prepare":
            for name in sorted({c["dataset"] for c in profile["cases"]}):
                print(prepare(root, load_manifest(root, name)))
            return 0
        resolved = plan(
            root, profile, args.engines.split(","), args.swift_worker, args.python
        )
        result = run(
            root,
            resolved,
            args.output or root / "Benchmarks/Runs" / str(uuid.uuid4()),
            purpose=args.command,
        )
        print("Certificate:", result["status"])
        return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
