"""Serial fresh-process execution with durable validation and provenance."""

import json
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from contracts import (
    digest,
    identity,
    load_workload,
    read_json,
    require,
    verified_file,
    write_json,
    validate_values,
)
from datasets import load_manifest, cache_path, reference, binary, resolve_workload
from reporting import validate_worker, summarize

METAL_BUILD_SETTINGS = {
    "MTL_FAST_MATH": "NO",
    "MTL_MATH_MODE": "SAFE",
    "MTL_MATH_FP32_FUNCTIONS": "PRECISE",
}

THREAD_ENV = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def source_identity(root):
    names = command(["git", "-C", str(root), "ls-files"]).splitlines()
    sources = {
        name: digest((root / name).read_bytes())
        for name in names
        if (root / name).is_file()
        and (
            name.startswith(
                (
                    "Sources/",
                    "Benchmarks/Tools/",
                    "Benchmarks/Worker/",
                    "Benchmarks/Support/",
                    "Benchmarks/Python/",
                )
            )
            or name in ["Package.swift", "Package.resolved"]
        )
    }
    return dict(
        commit=command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        files=sources,
        tree_sha256=identity(sources),
    )


def environment():
    result = dict(
        platform=platform.platform(),
        host=platform.node(),
        machine=platform.machine(),
        python=sys.version,
        threads=THREAD_ENV,
    )
    if sys.platform == "darwin":
        result.update(
            chip=command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            memory_bytes=command(["sysctl", "-n", "hw.memsize"]),
            os_build=command(["sw_vers", "-buildVersion"]),
            swift=command(["swift", "--version"]),
            xcode=command(["xcodebuild", "-version"]),
        )
    return result


def verify_uninstrumented(worker):
    sections = re.findall(
        r"^\s*sectname\s+(\S+)",
        command(["xcrun", "otool", "-l", str(worker)]),
        re.MULTILINE,
    )
    require(sections, "Cannot inspect benchmark Mach-O sections")
    coverage = [
        name for name in sections if name.startswith(("__llvm_prf", "__llvm_cov"))
    ]
    require(
        not coverage,
        f"Benchmark contains coverage instrumentation: {coverage}; rebuild with coverage disabled",
    )


def metal_build_record(worker):
    """Fingerprint the precompiled Metal resources beside the executable."""
    products = worker.parent
    paths = sorted(set(products.glob("*.metallib")) |
                   set(products.glob("*.bundle/**/*.metallib")))
    libraries = {
        str(path.relative_to(products)): digest(path.read_bytes())
        for path in paths if path.is_file()
    }
    require(libraries, "Metal libraries are missing; rebuild the worker")
    return dict(build_settings=dict(METAL_BUILD_SETTINGS), libraries=libraries)


def build(root, products, packages):
    """Build a manifest-verified tracked snapshot outside cloud-synced source."""
    source = source_identity(root)
    products.parent.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="source-", dir=products.parent))
    for name in command(["git", "-C", str(root), "ls-files"]).splitlines():
        p = root / name
        if not p.is_file():
            continue
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    args = [
        "xcodebuild",
        "build-for-testing",
        "-scheme",
        "SwiftSci-Package",
        "-configuration",
        "Release",
        "-destination",
        "platform=macOS,arch=arm64",
        "-derivedDataPath",
        str(products),
        "-clonedSourcePackagesDirPath",
        str(packages),
        "-onlyUsePackageVersionsFromResolvedFile",
        "-skipPackagePluginValidation",
        "-enableCodeCoverage",
        "NO",
        "SWIFT_ENABLE_CODE_COVERAGE=NO",
        "ENABLE_TESTABILITY=YES",
        "CLANG_ENABLE_CODE_COVERAGE=NO",
    ]
    args.extend(f"{name}={value}" for name, value in METAL_BUILD_SETTINGS.items())
    log = products.parent / "build.log"
    with log.open("w") as stream:
        subprocess.run(
            args, cwd=snapshot, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    worker = products / "Build/Products/Release/SwiftSciBenchmarkWorker"
    require(worker.is_file(), "Worker build did not produce executable")
    verify_uninstrumented(worker)
    require(
        source_identity(root) == source,
        "Source changed during build; rebuild before reporting",
    )
    write_json(
        Path(str(worker) + ".build.json"),
        dict(
            schema_version=1,
            source=source,
            command=args,
            binary_sha256=digest(worker.read_bytes()),
            coverage_instrumentation="absent",
            metal=metal_build_record(worker),
            log=str(log),
            snapshot=str(snapshot),
            environment=environment(),
        ),
    )
    return worker


def plan(root, profile, engines, swift_worker, python):
    require(
        len(set(engines)) == len(engines)
        and engines
        and set(engines) <= {"swiftsci", "pandas", "mlx"},
        "Supported engines: swiftsci,pandas,mlx",
    )
    engine_records = {}
    for engine in engines:
        if engine == "swiftsci":
            require(
                swift_worker is not None, "Use --swift-worker with a recorded build"
            )
            worker = Path(swift_worker).resolve()
            record = read_json(Path(str(worker) + ".build.json"))
            verified_file(worker, record["binary_sha256"])
            require(
                record.get("coverage_instrumentation") == "absent",
                "Worker has no coverage-instrumentation check; rebuild",
            )
            verify_uninstrumented(worker)
            require(
                record.get("metal") == metal_build_record(worker),
                "Metal build settings or library fingerprints differ; rebuild the worker",
            )
            require(
                record["environment"] == environment(),
                "Build and run toolchains differ; rebuild",
            )
            require(
                record["source"]["tree_sha256"] == source_identity(root)["tree_sha256"],
                "Swift worker source is stale; rebuild",
            )
            engine_records[engine] = dict(command=[str(worker)], build=record)
        elif engine == "mlx":
            version = command([python, "-c", "import importlib.metadata as m,numpy,pandas,sys;print(m.version('mlx'),m.version('mlx-metal'),numpy.__version__,pandas.__version__,sys.version)"])
            worker = root / "Benchmarks/Python/mlx_worker.py"
            engine_records[engine] = dict(command=[python, str(worker)], version=version,
                worker_sha256=digest(worker.read_bytes()),
                settings=dict(device="fixture-explicit", fallback=False, output="materialized",
                              decoder_initialization="direct-fixed-arrays"))
        else:
            version = command(
                [
                    python,
                    "-c",
                    "import pandas,numpy,scipy,pyarrow,sys;print(pandas.__version__,numpy.__version__,scipy.__version__,pyarrow.__version__,sys.version)",
                ]
            )
            engine_records[engine] = dict(
                command=[python, str(root / "Benchmarks/Python/standard_worker.py")],
                version=version,
                worker_sha256=digest(
                    (root / "Benchmarks/Python/standard_worker.py").read_bytes()
                ),
            )
    cases = []
    for case in profile["cases"]:
        dataset = load_manifest(root, case["dataset"])
        workload = resolve_workload(dataset, load_workload(root, case["workload"]))
        verified_file(
            cache_path(root, dataset), dataset["sha256"], dataset["size_bytes"]
        )
        spec = dict(case=case, dataset=dataset, workload=workload)
        cases.append(dict(**spec, case_key=identity(spec)))
    if "mlx" in engines:
        sys.path.insert(0, str(root / "Benchmarks/Python"))
        from mlx_workloads import OPERATIONS, check_supported
        for case in cases:
            require(case['workload']['operation'] in OPERATIONS,
                    'MLX comparison does not support: ' + case['case']['id'])
            check_supported(read_json(root / case['dataset']['fixture']))
    contract = dict(
        profile=profile,
        cases=cases,
        threads=THREAD_ENV,
        measurement="materialized-output-alive-v5",
        oracle_sha256=identity(
            {
                name: digest((root / "Benchmarks/Tools" / name).read_bytes())
                for name in ["workflow_fixtures.py", "workflow_reference.py", "datasets.py", "numerical_reference.py", "public_datasets.py", "numerical_fixtures.py", "controlled_fixtures.py", "supervised_fixtures.py", "classification_reference.py", "dataframe_fixtures.py", "neural_fixtures.py", "neural_reference.py", "vision_fixtures.py", "vision_reference.py", "boundary_fixtures.py", "boundary_reference.py", "boundary_sweep.py"]
            }
        ),
    )
    return dict(
        schema_version=1,
        profile=profile,
        cases=cases,
        engines=engine_records,
        contract_hash=identity(contract),
        environment=environment(),
        source=source_identity(root),
    )


def validate_parquet_artifacts(response_path, request, expected):
    import pyarrow as pa
    import pyarrow.parquet as pq

    require(
        pa.__version__ == "25.0.1", "Artifact validation requires pinned PyArrow 25.0.1"
    )
    records = []
    for index in range(request["warmups"], request["warmups"] + request["samples"]):
        path = Path(str(response_path) + f".sample{index}.parquet")
        data = path.read_bytes()
        table = pq.read_table(path)
        require(
            table.column_names == ["id", "group", "x", "y", "flag"],
            "Parquet schema mismatch",
        )
        require(
            pa.types.is_boolean(table.schema.field("flag").type),
            "Parquet Boolean type lost",
        )
        rows = table.to_pylist()
        require(len(rows) == request["rows"], "Parquet row count mismatch")
        actual = []
        for row in rows:
            key = row["group"]
            require(
                isinstance(key, str)
                and key.startswith("category")
                and key[8:].isdigit(),
                "Malformed Parquet category",
            )
            actual.extend(
                [row["id"], int(key[8:]), row["x"], row["y"], float(row["flag"])]
            )
        validate_values(actual, expected, request["atol"], request["rtol"])
        records.append(
            dict(
                file=path.name,
                sha256=digest(data),
                bytes=len(data),
                independently_validated=True,
            )
        )
    return records


def run(root, resolved, destination, purpose="benchmark"):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    run = dict(
        schema_version=1,
        id=destination.name,
        purpose=purpose,
        status="running",
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan=resolved,
        events=[],
    )
    write_json(destination / "run.json", run)
    profile = resolved["profile"]
    engines = list(resolved["engines"])
    for case in resolved["cases"]:
        dataset = case["dataset"]
        workload = case["workload"]
        expected_values = reference(root, dataset, workload)
        expected = binary(expected_values)
        expected_path = destination / (case["case"]["id"] + ".expected.f64")
        expected_path.write_bytes(expected)
        for batch in range(profile["batches"]):
            for engine in engines if batch % 2 == 0 else list(reversed(engines)):
                token = f"{case['case']['id']}-{engine}-{batch}"
                request = dict(
                    schema_version=1,
                    case_key=case["case_key"],
                    operation=workload["operation"],
                    dataset_kind=dataset["kind"],
                    input_path=str(cache_path(root, dataset)),
                    input_sha256=dataset["sha256"],
                    input_bytes=dataset["size_bytes"],
                    expected_path=str(expected_path),
                    expected_sha256=digest(expected),
                    rows=dataset["rows"],
                    warmups=profile["warmups"],
                    samples=profile["samples"],
                    atol=workload["atol"],
                    rtol=workload["rtol"],
                )
                if dataset["kind"] == "nist-univariate-v1":
                    request["input_skip_rows"] = dataset["data_start_line"] - 1
                request_path = destination / (token + ".request.json")
                write_json(request_path, request)
                response_path = destination / (token + ".response.json")
                event = dict(
                    case_id=case["case"]["id"],
                    case_key=case["case_key"],
                    engine=engine,
                    batch=batch,
                    status="failed",
                )
                try:
                    with (destination / (token + ".log")).open("w") as log:
                        completed = subprocess.run(
                            resolved["engines"][engine]["command"]
                            + [str(request_path), str(response_path)],
                            env=dict(os.environ, **THREAD_ENV),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=profile["timeout_seconds"],
                        )
                    require(
                        completed.returncode == 0,
                        f"Worker exited {completed.returncode}; see {token}.log",
                    )
                    result = read_json(response_path)
                    validate_worker(result, request)
                    if workload["operation"] == "parquet-write":
                        event["artifacts"] = validate_parquet_artifacts(
                            response_path, request, expected_values
                        )
                    event.update(status="passed", result=result)
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    event["error"] = str(error)
                    if response_path.is_file():
                        try:
                            event["failed_result"] = read_json(response_path)
                        except (OSError, ValueError):
                            pass
                run["events"].append(event)
                with (destination / "events.jsonl").open("a") as log:
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                write_json(destination / "run.json", run)
                print(f"{token}: {event['status']}", flush=True)
    run["status"] = (
        "passed" if all(e["status"] == "passed" for e in run["events"]) else "failed"
    )
    run["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run["summary"] = summarize(run)
    write_json(destination / "run.json", run)
    certificate = dict(
        schema_version=1,
        kind="SwiftSci workload conformance; not third-party accreditation",
        status=run["status"],
        run_sha256=digest((destination / "run.json").read_bytes()),
        contract_hash=resolved["contract_hash"],
        source=resolved["source"],
        cases=[dict(id=c["case"]["id"], key=c["case_key"]) for c in resolved["cases"]],
        validated_samples=sum(
            len(e["result"]["samples"])
            for e in run["events"]
            if e["status"] == "passed"
        ),
    )
    write_json(destination / "certificate.json", certificate)
    return run
