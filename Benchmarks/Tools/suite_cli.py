"""Unified CLI orchestration, catalog, environment diagnostics, and release-gate helpers for SwiftSci benchmarks."""

from datetime import datetime, timezone
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from contracts import read_json, write_json
from runner import environment, power_state, source_identity


DEFAULT_CACHE_DIR = Path.home() / "Library/Caches/SwiftSci/standardized-benchmarks"
DEFAULT_PACKAGES_DIR = Path.home() / "Library/Caches/SwiftSci/xcode-packages"

STANDARDIZED_MODULES = ("numpy", "pandas", "scipy", "pyarrow", "mpmath")
COMPARISON_MODULES = ("polars", "duckdb")
MLX_MODULES = ("mlx", "mlx.core")
SCIENTIFIC_MODULES = ("sklearn", "statsmodels", "shap")
LLM_MODULES = ("mlx_lm", "transformers", "safetensors", "huggingface_hub")

TRIAL_SCRIPTS = {
    "coreml": "Benchmarks/CoreMLInference/run.py",
    "bounded": "Benchmarks/BoundedInference/run.py",
    "fused": "Benchmarks/FusedPreprocessingTrial/run.py",
    "fused-coreml": "Benchmarks/FusedPreprocessingTrial/run_coreml.py",
    "mixed": "Benchmarks/MixedWorkloads/run.py",
    "persistent": "Benchmarks/PersistentModelTrial/run.py",
}

PRETRAINED_TOOLS = {
    "pretrained-controls": "Benchmarks/Tests/check_pretrained_inspection.py",
    "pretrained-llama": "Benchmarks/Tools/pretrained_llama.py",
    "generation": "Benchmarks/Tools/llama_generation_benchmark.py",
    "precision": "Benchmarks/Tools/llama_precision_benchmark.py",
    "gguf-verify": "Benchmarks/Tools/verify_llama_gguf.py",
    "llama-cpp": "Benchmarks/Tools/llama_cpp_compare.py",
}


def default_worker_path(cache_dir=DEFAULT_CACHE_DIR):
    """Return the default Release SwiftSciBenchmarkWorker path created by `bench.py build`."""
    return Path(cache_dir) / "derived/Build/Products/Release/SwiftSciBenchmarkWorker"


def resolve_worker(worker_arg=None, cache_dir=DEFAULT_CACHE_DIR):
    """Resolve an explicit `--swift-worker` argument or fall back to the default build product."""
    if worker_arg:
        return Path(worker_arg).resolve()
    return default_worker_path(cache_dir)


def timestamped_run_dir(root, prefix):
    """Return a fresh UTC timestamped directory under `Benchmarks/Runs/`."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return Path(root) / "Benchmarks/Runs" / f"{prefix}-{stamp}"


def can_import_modules(python_exe, modules):
    """Return True when `python_exe` exists and can import every module in `modules`."""
    py = Path(python_exe)
    if not py.is_file():
        return False
    if os.path.abspath(str(py)) == os.path.abspath(sys.executable):
        return all(importlib.util.find_spec(mod) is not None for mod in modules)
    stmt = "import " + ", ".join(modules)
    proc = subprocess.run(
        [str(py), "-c", stmt],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _candidate_python_paths(root, explicit, env_var, extra_candidates):
    seen = set()
    candidates = []
    if explicit:
        candidates.append(Path(os.path.abspath(str(explicit))))
    elif env_var and os.environ.get(env_var):
        candidates.append(Path(os.path.abspath(os.environ[env_var])))
    if not explicit:
        candidates.append(Path(os.path.abspath(sys.executable)))
        candidates.extend(extra_candidates)
    for item in candidates:
        key = os.path.abspath(str(item))
        if key not in seen:
            seen.add(key)
            yield Path(key)


def resolve_scientific_python(root, explicit=None):
    """Locate a Python interpreter capable of importing `sklearn`, `statsmodels`, and `shap`."""
    extras = [
        Path(root) / "Benchmarks/.venv/bin/python",
        Path(root) / "Benchmarks/.venv-scientific/bin/python",
        Path.home() / ".pyenv/versions/3.11.9/bin/python3",
    ]
    for candidate in _candidate_python_paths(
        root, explicit, "SWIFTSCI_SCIENTIFIC_PYTHON", extras
    ):
        if can_import_modules(candidate, SCIENTIFIC_MODULES):
            return candidate
    return None


def resolve_llm_python(root, explicit=None):
    """Locate an optional Python interpreter with `mlx_lm`, `transformers`, and `safetensors`."""
    extras = [
        Path(root) / "Benchmarks/.venv-llm/bin/python",
    ]
    for candidate in _candidate_python_paths(root, explicit, "SWIFTSCI_LLM_PYTHON", extras):
        if can_import_modules(candidate, LLM_MODULES):
            return candidate
    return None


def _installed_package_versions(python_exe, packages):
    py = Path(python_exe)
    if not py.is_file():
        return {}
    if os.path.abspath(str(py)) == os.path.abspath(sys.executable):
        versions = {}
        for pkg in packages:
            try:
                versions[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                versions[pkg] = None
        return versions
    script = (
        "import importlib.metadata as m, json, sys;"
        "pkgs = sys.argv[1:];"
        "out = {};"
        "\nfor p in pkgs:\n"
        "    try: out[p] = m.version(p)\n"
        "    except m.PackageNotFoundError: out[p] = None\n"
        "print(json.dumps(out))"
    )
    proc = subprocess.run(
        [str(py), "-c", script, *packages],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}
    return json.loads(proc.stdout)


def build_catalog(root):
    """Return a structured inventory of all active SwiftSci benchmark suites and profiles."""
    root = Path(root)
    acc = read_json(root / "Benchmarks/Specs/acceptance.json")
    tiers = {"cpu": [], "apple": [], "sweep": []}
    for entry in acc["profiles"]:
        prof = read_json(root / "Benchmarks/Specs/profiles" / f"{entry['profile']}.json")
        tiers[entry["tier"]].append(
            {
                "profile": entry["profile"],
                "cases": len(prof.get("cases", [])),
                "warmups": prof.get("warmups"),
                "samples": prof.get("samples"),
                "batches": prof.get("batches"),
                "purpose": entry["purpose"],
            }
        )

    datasets_count = len(list((root / "Benchmarks/Specs/datasets").glob("*.json")))
    workloads_count = len(list((root / "Benchmarks/Specs/workloads").glob("*.json")))
    return {
        "schema_version": 1,
        "standardized": {
            "profiles_count": sum(len(v) for v in tiers.values()),
            "datasets_count": datasets_count,
            "workloads_count": workloads_count,
            "tiers": tiers,
        },
        "scientific_validation": {
            "script": "Benchmarks/Tools/scientific_validation_suite.py",
            "command": "bench.py scientific",
            "workloads_count": 39,
            "rolling_origin_backtests": 3,
        },
        "controls": {
            "pretrained": "Benchmarks/Tests/check_pretrained_inspection.py",
            "workflow": "Benchmarks/Tests/check_public_workflow_workers.py",
            "command": "bench.py controls",
        },
        "pretrained_llama": PRETRAINED_TOOLS,
        "trials": TRIAL_SCRIPTS,
    }


def _format_catalog_text(catalog):
    std = catalog["standardized"]
    lines = [
        f"SwiftSci Benchmarks Catalog ({std['profiles_count']} profiles, "
        f"{std['datasets_count']} datasets, {std['workloads_count']} workloads)",
        "",
    ]
    for tier in ("cpu", "apple", "sweep"):
        items = std["tiers"][tier]
        lines.append(f"[{tier}] ({len(items)} profiles)")
        for item in items:
            lines.append(
                f"  - {item['profile']:32s} cases={item['cases']:2d} "
                f"w/s/b={item['warmups']}/{item['samples']}/{item['batches']}  "
                f"{item['purpose']}"
            )
        lines.append("")
    sci = catalog["scientific_validation"]
    lines.append(
        f"[scientific] {sci['workloads_count']} workloads + "
        f"{sci['rolling_origin_backtests']} rolling-origin backtests ({sci['command']})"
    )
    lines.append(f"[controls] pretrained, workflow ({catalog['controls']['command']})")
    lines.append("[trials] " + ", ".join(sorted(catalog["trials"])) + " (bench.py trial <name>)")
    lines.append("[pretrained-llama] " + ", ".join(sorted(catalog["pretrained_llama"])))
    return "\n".join(lines)


def cmd_list(root, as_json=False):
    """Print available benchmark tiers, profiles, scientific suite, controls, and trials."""
    catalog = build_catalog(root)
    if as_json:
        print(json.dumps(catalog, indent=2))
    else:
        print(_format_catalog_text(catalog))
    return 0


def collect_doctor_report(
    root,
    worker_path=None,
    python_exe=None,
    scientific_python=None,
    llm_python=None,
):
    """Inspect source fingerprint, Release worker build freshness, Python packages, and power state."""
    root = Path(root)
    src = source_identity(root)
    worker = resolve_worker(worker_arg=worker_path)
    build_json_path = Path(f"{worker}.build.json")

    worker_info = {
        "path": str(worker),
        "exists": worker.is_file(),
        "build_record_path": str(build_json_path),
        "build_record_exists": build_json_path.is_file(),
        "matches_current_source": False,
        "coverage_instrumentation": None,
        "binary_sha256": None,
    }
    if build_json_path.is_file():
        try:
            record = read_json(build_json_path)
            worker_info["coverage_instrumentation"] = record.get("coverage_instrumentation")
            worker_info["binary_sha256"] = record.get("binary_sha256")
            worker_info["matches_current_source"] = (
                record.get("source", {}).get("tree_sha256") == src["tree_sha256"]
            )
        except Exception as error:
            worker_info["build_record_error"] = f"{type(error).__name__}: {error}"

    py = Path(os.path.abspath(str(python_exe or sys.executable)))
    tracked_pkgs = (
        "numpy",
        "pandas",
        "scipy",
        "pyarrow",
        "mpmath",
        "polars",
        "duckdb",
        "mlx",
        "mlx-lm",
        "scikit-learn",
        "statsmodels",
        "shap",
        "transformers",
    )
    versions = _installed_package_versions(py, tracked_pkgs)
    groups = {
        "standardized": all(versions.get(p) is not None for p in STANDARDIZED_MODULES),
        "comparison": all(versions.get(p) is not None for p in COMPARISON_MODULES),
        "mlx": versions.get("mlx") is not None,
        "scientific": all(
            versions.get(p) is not None for p in ("scikit-learn", "statsmodels", "shap")
        ),
        "llm": all(
            versions.get(p) is not None
            for p in ("mlx-lm", "transformers", "safetensors")
        ),
    }

    sci_py = resolve_scientific_python(root, explicit=scientific_python)
    llm_py = resolve_llm_python(root, explicit=llm_python)
    uv_path = shutil.which("uv")
    uv_version = None
    if uv_path:
        proc = subprocess.run([uv_path, "--version"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            uv_version = proc.stdout.strip()

    worker_fresh = bool(
        worker_info["exists"]
        and worker_info["build_record_exists"]
        and worker_info["matches_current_source"]
        and worker_info["coverage_instrumentation"] == "absent"
    )
    return {
        "schema_version": 1,
        "source": {
            "commit": src["commit"],
            "dirty": src["dirty"],
            "tree_sha256": src["tree_sha256"],
        },
        "worker": worker_info,
        "python": {
            "executable": str(py),
            "version": sys.version.split()[0],
            "groups": groups,
            "packages": versions,
        },
        "environments": {
            "uv": uv_version,
            "venv_default_exists": (root / "Benchmarks/.venv/bin/python").is_file(),
            "venv_standardized_exists": (root / "Benchmarks/.venv-standardized/bin/python").is_file(),
            "venv_scientific_exists": (root / "Benchmarks/.venv-scientific/bin/python").is_file(),
            "venv_llm_exists": (root / "Benchmarks/.venv-llm/bin/python").is_file(),
            "resolved_scientific_python": str(sci_py) if sci_py else None,
            "resolved_llm_python": str(llm_py) if llm_py else None,
        },
        "power": power_state(),
        "ready": {
            "standardized_cpu": groups["standardized"],
            "comparison_engines": groups["standardized"] and groups["comparison"],
            "mlx": groups["standardized"] and groups["mlx"],
            "scientific": sci_py is not None,
            "llm_pretrained": llm_py is not None,
            "worker_fresh": worker_fresh,
        },
    }


def _format_doctor_text(report):
    src = report["source"]
    wrk = report["worker"]
    env = report["environments"]
    rdy = report["ready"]
    lines = [
        f"Source commit: {src['commit']} (dirty={src['dirty']}, tree={src['tree_sha256'][:12]})",
        f"Power: {report['power'].get('source')}",
        f"uv: {env.get('uv') or 'not found'}",
        f"Python: {report['python']['executable']}",
        f"Scientific Python: {env.get('resolved_scientific_python') or 'not configured'}",
        f"LLM Python: {env.get('resolved_llm_python') or 'not configured'}",
        f"Worker: {wrk['path']} (exists={wrk['exists']}, fresh={rdy['worker_fresh']})",
        "Readiness: " + ", ".join(f"{k}={v}" for k, v in rdy.items()),
    ]
    return "\n".join(lines)


def cmd_doctor(root, args):
    """Print environment and worker readiness diagnostics."""
    report = collect_doctor_report(
        root,
        worker_path=args.swift_worker,
        python_exe=args.python,
        scientific_python=args.python_engine,
        llm_python=args.llm_python,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_doctor_text(report))
    if args.strict and not (
        report["ready"]["standardized_cpu"] and report["ready"]["worker_fresh"]
    ):
        return 1
    return 0


def build_trial_command(root, trial_name, swift_worker, output_dir, extra_args=None, python_exe=sys.executable):
    """Construct the subprocess command list for a Core ML or memory-budget trial harness."""
    if trial_name not in TRIAL_SCRIPTS:
        raise ValueError(f"Unknown trial '{trial_name}'. Expected one of: {', '.join(sorted(TRIAL_SCRIPTS))}")
    script = Path(root) / TRIAL_SCRIPTS[trial_name]
    cmd = [
        str(python_exe),
        str(script),
        "--worker",
        str(swift_worker),
        "--output",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _default_step_executor(cmd, cwd, env):
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return proc.returncode, time.monotonic() - started


def plan_release_gate_steps(
    root,
    *,
    tier="cpu",
    swift_worker=None,
    python_exe=sys.executable,
    scientific_python=None,
    output_dir=None,
    include_build=True,
    include_tests=True,
    include_acceptance=True,
    include_controls=True,
    include_scientific=True,
    include_comparison=False,
    status_label="LOCAL",
    run_label="release-gate",
):
    """Build ordered step specifications for `bench.py release-gate`."""
    root = Path(root)
    out = Path(output_dir) if output_dir else timestamped_run_dir(root, "release-gate")
    worker = resolve_worker(swift_worker)
    py = str(python_exe)
    steps = []

    if include_build:
        steps.append(
            {
                "name": "build",
                "cmd": [py, str(root / "Benchmarks/Tools/bench.py"), "build"],
                "artifact": str(worker),
                "env": None,
            }
        )

    steps.append(
        {
            "name": "inventory",
            "cmd": [py, str(root / "Benchmarks/Tools/bench.py"), "inventory"],
            "artifact": None,
            "env": None,
        }
    )

    if include_tests:
        test_env = os.environ.copy()
        test_env["POLARS_MAX_THREADS"] = "1"
        steps.append(
            {
                "name": "unittest",
                "cmd": [py, "-m", "unittest", "discover", "-s", "Benchmarks/Tests"],
                "artifact": None,
                "env": test_env,
            }
        )

    acc_dir = out / "acceptance"
    if include_acceptance:
        acc_cmd = [
            py,
            str(root / "Benchmarks/Tools/acceptance.py"),
            "--tier",
            tier,
            "--swift-worker",
            str(worker),
            "--python",
            py,
            "--output",
            str(acc_dir),
        ]
        if tier == "cpu":
            acc_cmd.append("--check-baseline")
        steps.append(
            {
                "name": f"acceptance-{tier}",
                "cmd": acc_cmd,
                "artifact": str(acc_dir / "acceptance.json"),
                "env": None,
            }
        )

    if include_controls:
        pre_dir = out / "pretrained-inspection"
        steps.append(
            {
                "name": "pretrained-controls",
                "cmd": [
                    py,
                    str(root / "Benchmarks/Tests/check_pretrained_inspection.py"),
                    "--swift-worker",
                    str(worker),
                    "--output",
                    str(pre_dir),
                ],
                "artifact": str(pre_dir),
                "env": None,
            }
        )
        if include_acceptance and tier in ("cpu", "all"):
            wf_dir = out / "workflow-controls"
            steps.append(
                {
                    "name": "workflow-controls",
                    "cmd": [
                        py,
                        str(root / "Benchmarks/Tests/check_public_workflow_workers.py"),
                        "--run",
                        str(acc_dir / "scientific-workflow"),
                        "--run",
                        str(acc_dir / "persisted-workflow"),
                        "--output",
                        str(wf_dir),
                    ],
                    "artifact": str(wf_dir),
                    "env": None,
                }
            )

    if include_scientific:
        sci_py = resolve_scientific_python(root, explicit=scientific_python)
        if sci_py is None:
            raise ValueError(
                "Scientific Python environment not found (requires scikit-learn, statsmodels, shap). "
                "Run `uv sync --project Benchmarks --extra scientific` or pass --python-engine."
            )
        sci_json = out / "scientific_validation.json"
        steps.append(
            {
                "name": "scientific-validation",
                "cmd": [
                    py,
                    str(root / "Benchmarks/Tools/scientific_validation_suite.py"),
                    "--swift-worker",
                    str(worker),
                    "--python-engine",
                    str(sci_py),
                    "--status",
                    status_label,
                    "--label",
                    run_label,
                    "--output",
                    str(sci_json),
                ],
                "artifact": str(sci_json),
                "env": None,
            }
        )

    if include_comparison:
        cmp_dir = out / "comparison"
        steps.append(
            {
                "name": "comparison-suite",
                "cmd": [
                    py,
                    str(root / "Benchmarks/Tools/comparison_suite.py"),
                    "--swift-worker",
                    str(worker),
                    "--python",
                    py,
                    "--output",
                    str(cmp_dir),
                ],
                "artifact": str(cmp_dir),
                "env": None,
            }
        )

    return out, worker, steps


def run_release_gate(
    root,
    *,
    tier="cpu",
    swift_worker=None,
    python_exe=sys.executable,
    scientific_python=None,
    output_dir=None,
    include_build=True,
    include_tests=True,
    include_acceptance=True,
    include_controls=True,
    include_scientific=True,
    include_comparison=False,
    status_label="LOCAL",
    run_label="release-gate",
    dry_run=False,
    step_executor=None,
):
    """Run the unified release-gate pipeline and write `<output>/summary.json`."""
    root = Path(root)
    out, worker, planned = plan_release_gate_steps(
        root,
        tier=tier,
        swift_worker=swift_worker,
        python_exe=python_exe,
        scientific_python=scientific_python,
        output_dir=output_dir,
        include_build=include_build,
        include_tests=include_tests,
        include_acceptance=include_acceptance,
        include_controls=include_controls,
        include_scientific=include_scientific,
        include_comparison=include_comparison,
        status_label=status_label,
        run_label=run_label,
    )
    if dry_run:
        payload = {
            "schema_version": 1,
            "command": "release-gate",
            "dry_run": True,
            "output_dir": str(out),
            "worker": str(worker),
            "steps": [{"name": s["name"], "command": s["cmd"], "artifact": s["artifact"]} for s in planned],
        }
        return payload, 0

    out.mkdir(parents=True, exist_ok=False)
    executor = step_executor or _default_step_executor
    results = []
    for step in planned:
        rc, elapsed = executor(step["cmd"], root, step["env"])
        results.append(
            {
                "name": step["name"],
                "status": "passed" if rc == 0 else "failed",
                "exit_code": rc,
                "duration_seconds": round(elapsed, 3),
                "artifact": step["artifact"],
                "command": step["cmd"],
            }
        )

    all_passed = bool(results) and all(item["exit_code"] == 0 for item in results)
    src = source_identity(root)
    summary = {
        "schema_version": 1,
        "command": "release-gate",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tier": tier,
        "status": "passed" if all_passed else "failed",
        "all_passed": all_passed,
        "source": {
            "commit": src["commit"],
            "dirty": src["dirty"],
            "tree_sha256": src["tree_sha256"],
        },
        "worker": str(worker),
        "environment": environment(),
        "power_state": power_state(),
        "steps": results,
    }
    write_json(out / "summary.json", summary)
    return summary, (0 if all_passed else 1)


def register_subcommands(commands):
    """Register unified benchmark orchestration subcommands on the `bench.py` parser."""
    p = commands.add_parser("list", help="List all benchmark profiles, suites, controls, and trials")
    p.add_argument("--json", action="store_true")

    p = commands.add_parser("doctor", help="Inspect worker freshness, Python dependencies, and power state")
    p.add_argument("--swift-worker")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--python-engine")
    p.add_argument("--llm-python")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true")

    p = commands.add_parser("accept", help="Run acceptance tier (`cpu`, `apple`, `sweep`, `all`)")
    p.add_argument("--tier", choices=["cpu", "apple", "sweep", "all"], default="cpu")
    p.add_argument("--swift-worker")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--check-baseline", action="store_true")
    p.add_argument("--output", type=Path)

    p = commands.add_parser("scientific", help="Run the 39-workload accuracy-gated scientific validation suite")
    p.add_argument("--swift-worker")
    p.add_argument("--python-engine")
    p.add_argument("--output", type=Path)
    p.add_argument("--processes", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--status", default="LOCAL")
    p.add_argument("--label", default="scientific-validation")

    p = commands.add_parser("compare-engines", help="Run the 4-engine tabular comparison driver")
    p.add_argument("--swift-worker")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--mode", choices=["legacy-capped", "production-default"], default="production-default")
    p.add_argument("--output", type=Path)

    p = commands.add_parser("controls", help="Run synthetic pretrained inspection and workflow mutation controls")
    p.add_argument("--kind", choices=["all", "pretrained", "workflow"], default="all")
    p.add_argument("--swift-worker")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workflow-run", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path)

    p = commands.add_parser("trial", help="Run an opt-in Core ML or memory-budget trial harness")
    p.add_argument("name", choices=sorted(TRIAL_SCRIPTS))
    p.add_argument("--swift-worker", "--worker", dest="swift_worker")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output", type=Path)

    p = commands.add_parser("release-gate", help="Run the end-to-end verification pipeline and emit summary.json")
    p.add_argument("--tier", choices=["cpu", "apple", "sweep", "all"], default="cpu")
    p.add_argument("--swift-worker")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--python-engine")
    p.add_argument("--output", type=Path)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-acceptance", action="store_true")
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--skip-scientific", action="store_true")
    p.add_argument("--with-comparison", action="store_true")
    p.add_argument("--status", default="LOCAL")
    p.add_argument("--label", default="release-gate")
    p.add_argument("--dry-run", action="store_true")


def dispatch(root, args, extra_args=None):
    """Execute a registered unified CLI subcommand without holding the checkout `.lock`."""
    root = Path(root)
    cmd_name = args.command
    if cmd_name == "list":
        return cmd_list(root, as_json=args.json)
    if cmd_name == "doctor":
        return cmd_doctor(root, args)
    if cmd_name == "accept":
        worker = resolve_worker(args.swift_worker)
        out = args.output or timestamped_run_dir(root, f"acceptance-{args.tier}")
        cmd = [
            args.python,
            str(root / "Benchmarks/Tools/acceptance.py"),
            "--tier",
            args.tier,
            "--swift-worker",
            str(worker),
            "--python",
            args.python,
            "--output",
            str(out),
        ]
        if args.check_baseline:
            cmd.append("--check-baseline")
        return subprocess.run(cmd, cwd=str(root), check=False).returncode
    if cmd_name == "scientific":
        worker = resolve_worker(args.swift_worker)
        sci_py = resolve_scientific_python(root, explicit=args.python_engine)
        if sci_py is None:
            raise ValueError(
                "Scientific Python environment not found (requires scikit-learn, statsmodels, shap). "
                "Run `uv sync --project Benchmarks --extra scientific` or pass --python-engine."
            )
        out = args.output or (timestamped_run_dir(root, "scientific") / "scientific_validation.json")
        cmd = [
            sys.executable,
            str(root / "Benchmarks/Tools/scientific_validation_suite.py"),
            "--swift-worker",
            str(worker),
            "--python-engine",
            str(sci_py),
            "--processes",
            str(args.processes),
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
            "--status",
            args.status,
            "--label",
            args.label,
            "--output",
            str(out),
        ]
        return subprocess.run(cmd, cwd=str(root), check=False).returncode
    if cmd_name == "compare-engines":
        worker = resolve_worker(args.swift_worker)
        out = args.output or timestamped_run_dir(root, "comparison")
        cmd = [
            args.python,
            str(root / "Benchmarks/Tools/comparison_suite.py"),
            "--swift-worker",
            str(worker),
            "--python",
            args.python,
            "--mode",
            args.mode,
            "--output",
            str(out),
        ]
        return subprocess.run(cmd, cwd=str(root), check=False).returncode
    if cmd_name == "controls":
        worker = resolve_worker(args.swift_worker)
        out = args.output or timestamped_run_dir(root, "controls")
        out.mkdir(parents=True, exist_ok=True)
        results = {}
        rc = 0
        if args.kind in ("all", "pretrained"):
            pre_cmd = [
                args.python,
                str(root / "Benchmarks/Tests/check_pretrained_inspection.py"),
                "--swift-worker",
                str(worker),
                "--output",
                str(out / "pretrained"),
            ]
            pre_rc = subprocess.run(pre_cmd, cwd=str(root), check=False).returncode
            results["pretrained"] = "passed" if pre_rc == 0 else "failed"
            rc = max(rc, pre_rc)
        if args.kind in ("all", "workflow"):
            if not args.workflow_run:
                if args.kind == "workflow":
                    raise ValueError("--workflow-run is required when --kind workflow is selected")
            else:
                wf_cmd = [
                    args.python,
                    str(root / "Benchmarks/Tests/check_public_workflow_workers.py"),
                    "--output",
                    str(out / "workflow"),
                ]
                for run_dir in args.workflow_run:
                    wf_cmd.extend(["--run", str(run_dir)])
                wf_rc = subprocess.run(wf_cmd, cwd=str(root), check=False).returncode
                results["workflow"] = "passed" if wf_rc == 0 else "failed"
                rc = max(rc, wf_rc)
        write_json(out / "summary.json", {"schema_version": 1, "controls": results, "status": "passed" if rc == 0 else "failed"})
        return rc
    if cmd_name == "trial":
        worker = resolve_worker(args.swift_worker)
        out = args.output or timestamped_run_dir(root, f"trial-{args.name}")
        cmd = build_trial_command(
            root,
            args.name,
            worker,
            out,
            extra_args=extra_args or [],
            python_exe=args.python,
        )
        return subprocess.run(cmd, cwd=str(root), check=False).returncode
    if cmd_name == "release-gate":
        summary, rc = run_release_gate(
            root,
            tier=args.tier,
            swift_worker=args.swift_worker,
            python_exe=args.python,
            scientific_python=args.python_engine,
            output_dir=args.output,
            include_build=not args.skip_build,
            include_tests=not args.skip_tests,
            include_acceptance=not args.skip_acceptance,
            include_controls=not args.skip_controls,
            include_scientific=not args.skip_scientific,
            include_comparison=args.with_comparison,
            status_label=args.status,
            run_label=args.label,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return rc
    raise ValueError(f"Unsupported command: {cmd_name}")
