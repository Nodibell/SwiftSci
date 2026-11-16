"""Reconcile every legacy registration with a replacement or research disposition."""

import ast
from collections import Counter
import re
from contracts import fields, load_profile, load_workload, read_json, require

INVENTORY_PATH = "Benchmarks/Specs/legacy-inventory.json"
LEGACY_DISCOVERY_SOURCES = [
    "Benchmarks/Swift",
    "Benchmarks/Python/benchmarks.py",
    "Benchmarks/Python/accuracy_benchmarks.py",
]
LEGACY_RETIRED_PATHS = [
    *LEGACY_DISCOVERY_SOURCES,
    "Benchmarks/Python/compare.py",
    "Benchmarks/generate_fixtures.py",
    "Benchmarks/CSVAcceleration",
    "python_accuracy.json",
    "Benchmarks/Results/python_results.json",
    "Benchmarks/Results/swift_results.json",
]
EXPERIMENTAL_REFERENCES = [
    {
        "id": "kiraa",
        "role": "experimental-reference",
        "optional": True,
        "official": False,
        "known_buggy": True,
        "correctness_oracle": False,
        "certification_engine": False,
    }
]


def has_active_legacy_sources(root):
    """Return True when legacy registration files exist under root."""
    return any((root / rel).exists() for rel in LEGACY_DISCOVERY_SOURCES)


def retired_legacy_stragglers(root):
    """List retired legacy benchmark paths that still exist under root."""
    return [rel for rel in LEGACY_RETIRED_PATHS if (root / rel).exists()]


def discover(root):
    """Read registrations without importing optional legacy dependencies."""
    if not has_active_legacy_sources(root):
        return []
    found = []
    swift_dir = root / "Benchmarks/Swift"
    if swift_dir.exists():
        for path in sorted(swift_dir.glob("*Benchmarks.swift")):
            source = str(path.relative_to(root))
            text = path.read_text()
            calls = list(re.finditer(r"BenchmarkRunner\.run\s*\(", text))
            names = re.findall(r'BenchmarkRunner\.run\s*\(\s*name:\s*"([^"\n]+)"', text)
            require(len(calls) == len(names), f"Unrecognized Swift registration: {source}")
            found.extend({"source": source, "name": n, "kind": "timed"} for n in names)
    source = "Benchmarks/Python/benchmarks.py"
    benchmarks_py = root / source
    if benchmarks_py.exists():
        for node in ast.walk(ast.parse(benchmarks_py.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_benchmark"
            ):
                name = (
                    node.args[0]
                    if node.args
                    else next((k.value for k in node.keywords if k.arg == "name"), None)
                )
                require(
                    isinstance(name, ast.Constant) and isinstance(name.value, str),
                    "Dynamic Python benchmark name",
                )
                found.append({"source": source, "name": name.value, "kind": "timed"})
    source = "Benchmarks/Python/accuracy_benchmarks.py"
    accuracy_py = root / source
    if accuracy_py.exists():
        for node in ast.walk(ast.parse(accuracy_py.read_text())):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "results"
                    ):
                        require(
                            isinstance(target.slice, ast.Constant)
                            and isinstance(target.slice.value, str),
                            "Dynamic accuracy result key",
                        )
                        found.append(
                            {
                                "source": source,
                                "name": target.slice.value,
                                "kind": "accuracy-result",
                            }
                        )
    source = "Benchmarks/Swift/AccuracyBenchmarks.swift"
    accuracy_swift = root / source
    if accuracy_swift.exists():
        text = accuracy_swift.read_text()
        for name, pattern in [
            ("Student t-test", r"Stats\.tTest\([^\n]*equalVariances:\s*true"),
            ("Paired t-test", r"Stats\.pairedTTest\("),
        ]:
            if re.search(pattern, text):
                found.append({"source": source, "name": name, "kind": "untimed-diagnostic"})
    keys = [(x["source"], x["name"]) for x in found]
    require(len(keys) == len(set(keys)), "Duplicate legacy registration")
    return sorted(found, key=lambda x: (x["source"], x["name"]))


def reconcile(root, inventory=None):
    inventory = inventory if inventory is not None else read_json(root / INVENTORY_PATH)
    fields(inventory, ["schema_version", "scope", "experimental_references", "entries"])
    require(inventory["schema_version"] == 1, "Inventory version mismatch")
    require(
        inventory["experimental_references"] == EXPERIMENTAL_REFERENCES,
        "Kiraa must remain an optional unofficial, known-buggy experimental reference, never an oracle or certification engine",
    )
    require(isinstance(inventory["entries"], list), "Inventory entries must be a list")
    if has_active_legacy_sources(root):
        expected = {(x["source"], x["name"]): x["kind"] for x in discover(root)}
        legacy_sources_retired = False
    else:
        stragglers = retired_legacy_stragglers(root)
        require(
            not stragglers,
            f"Retired legacy benchmark paths must remain removed: {stragglers}",
        )
        canonical = read_json(root / INVENTORY_PATH) if (root / INVENTORY_PATH).exists() else inventory
        expected = {}
        for entry in canonical["entries"]:
            fields(entry, ["source", "name", "kind", "suite", "reason", "replacements"])
            key = (entry["source"], entry["name"])
            require(key not in expected, f"Duplicate canonical inventory entry: {key}")
            expected[key] = entry["kind"]
        legacy_sources_retired = True
    seen = set()
    workloads = set()
    for entry in inventory["entries"]:
        fields(entry, ["source", "name", "kind", "suite", "reason", "replacements"])
        key = (entry["source"], entry["name"])
        require(key not in seen, f"Duplicate inventory entry: {key}")
        seen.add(key)
        require(key in expected, f"Dangling legacy entry: {key}")
        require(entry["kind"] == expected[key], f"Registration kind mismatch: {key}")
        require(
            entry["suite"] in ["standardized-core", "research"], f"Invalid suite: {key}"
        )
        require(
            isinstance(entry["reason"], str) and len(entry["reason"].strip()) >= 25,
            f"Missing disposition reason: {key}",
        )
        require(isinstance(entry["replacements"], list), f"Invalid replacements: {key}")
        require(
            bool(entry["replacements"]) == (entry["suite"] == "standardized-core"),
            f"Suite/replacement mismatch: {key}",
        )
        mapped = set()
        for replacement in entry["replacements"]:
            fields(replacement, ["operation", "workload", "profiles"])
            workload = load_workload(root, replacement["workload"])
            require(
                workload["operation"] == replacement["operation"],
                f"Replacement operation mismatch: {key}",
            )
            require(workload["id"] not in mapped, f"Duplicate replacement: {key}")
            mapped.add(workload["id"])
            workloads.add(workload["id"])
            profiles = replacement["profiles"]
            require(
                isinstance(profiles, list)
                and profiles
                and len(profiles) == len(set(profiles)),
                f"Missing or duplicate replacement profiles: {key}",
            )
            for name in profiles:
                profile = load_profile(root, name)
                require(
                    any(
                        case["workload"] == workload["id"] for case in profile["cases"]
                    ),
                    f"Replacement absent from profile {name}: {key}",
                )
    require(
        seen == set(expected),
        f"Unclassified legacy entries: {sorted(set(expected) - seen)}",
    )
    return {
        "status": "passed",
        "entries": len(seen),
        "by_kind": dict(
            sorted(Counter(x["kind"] for x in inventory["entries"]).items())
        ),
        "by_suite": dict(
            sorted(Counter(x["suite"] for x in inventory["entries"]).items())
        ),
        "replacement_workloads": len(workloads),
        "experimental_references": inventory["experimental_references"],
        "legacy_sources_retired": legacy_sources_retired,
    }
