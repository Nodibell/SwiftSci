"""Export validated LCOV from every macOS XCTest bundle in a SwiftPM build."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REQUIRED_MODULES = {"SwiftAgent", "SwiftDataFrame", "SwiftNLP"}


def export_coverage(build_dir, profile):
    if not profile.is_file():
        sys.stderr.write(f"Coverage profile not found: {profile}\n")
        return 1

    bundles = sorted(build_dir.rglob("*.xctest"))
    if not bundles:
        sys.stderr.write(f"No .xctest bundles found in {build_dir}\n")
        return 1
    binaries = [bundle / "Contents" / "MacOS" / bundle.stem for bundle in bundles]
    for binary in binaries:
        if not binary.is_file():
            sys.stderr.write(f"Test bundle executable not found: {binary}\n")
            return 1

    # llvm-cov treats additional positional arguments as source filters.
    command = ["xcrun", "llvm-cov", "export", "-format=lcov",
               "-instr-profile=" + str(profile), str(binaries[0])]
    for binary in binaries[1:]:
        command.extend(["-object", str(binary)])

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as coverage:
        result = subprocess.run(command, stdout=coverage)
        if result.returncode:
            return result.returncode

        coverage.seek(0)
        covered = set()
        module = None
        for line in coverage:
            if line.startswith("SF:"):
                module = next((name for name in REQUIRED_MODULES
                               if f"/Sources/{name}/" in "/" + line[3:].strip()), None)
            elif line.startswith("DA:") and module and int(line.split(",")[1]) > 0:
                covered.add(module)
            elif line.strip() == "end_of_record":
                module = None
        missing = REQUIRED_MODULES - covered
        if missing:
            sys.stderr.write("No executed source coverage for: " + ", ".join(sorted(missing)) + "\n")
            return 1

        coverage.seek(0)
        shutil.copyfileobj(coverage, sys.stdout)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="SwiftPM debug binary directory")
    parser.add_argument("profile", type=Path, help="Merged safe-test .profdata file")
    args = parser.parse_args()
    try:
        return export_coverage(args.build_dir, args.profile)
    except OSError as error:
        sys.stderr.write(f"Coverage export failed: {error}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
