"""Reconcile every legacy registration with a replacement or research disposition."""

import ast
from collections import Counter
import re
from contracts import fields, load_profile, load_workload, read_json, require

INVENTORY_PATH = "Benchmarks/Specs/legacy-inventory.json"
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


def discover(root):
    """Read registrations without importing optional legacy dependencies."""
    found = []
    for path in sorted((root / "Benchmarks/Swift").glob("*Benchmarks.swift")):
        source = str(path.relative_to(root))
        text = path.read_text()
        calls = list(re.finditer(r"BenchmarkRunner\.run\s*\(", text))
        names = re.findall(r'BenchmarkRunner\.run\s*\(\s*name:\s*"([^"\n]+)"', text)
        require(len(calls) == len(names), f"Unrecognized Swift registration: {source}")
        found.extend({"source": source, "name": n, "kind": "timed"} for n in names)
    source = "Benchmarks/Python/benchmarks.py"
    for node in ast.walk(ast.parse((root / source).read_text())):
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
    for node in ast.walk(ast.parse((root / source).read_text())):
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
    text = (root / source).read_text()
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
    expected = {(x["source"], x["name"]): x["kind"] for x in discover(root)}
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
    }
