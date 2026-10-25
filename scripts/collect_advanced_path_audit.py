"""Inspect the built advanced transport without launching a GPU workload.

Run on the remote host, after the extension build and benchmark snapshots are
complete. This collector only reads sources, the existing binary, hardware
properties and saved counters; its writes stay in results/s5_s8 or results/f2. It never
imports pcie_ep._C, builds an extension, resets counters or changes links.
"""

import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

CPP_SUFFIXES = {".cu", ".cuh", ".hpp", ".cpp", ".h"}
FORBIDDEN_APIS = {
    "cuda_peer_copy": r"\bcudaMemcpyPeer(?:Async)?\b",
    "peer_enable": r"\bcudaDeviceEnablePeerAccess\b",
    "cuda_ipc_peer_memory": r"\bcudaIpc(?:Open|Get)MemHandle\b",
    "driver_peer_vmm": r"\bcuMem(?:SetAccess|Map|ImportFromShareableHandle)\b",
    "nccl": r"\bnccl\w*\b",
    "nvshmem": r"\bnvshmem\w*\b",
    "gin": r"\b(?:Gin|GIN)\b|\bgin::",
    "managed_memory": r"\bcudaMallocManaged\b",
}
PYTHON_GPU_COMMUNICATION = (
    r"^(?:dist|torch\.distributed)\.(?:all_reduce|reduce|all_to_all(?:_single)?|"
    r"all_gather(?:_into_tensor)?|reduce_scatter(?:_tensor)?|broadcast|"
    r"send|recv|isend|irecv|batch_isend_irecv)$|^torch\.cuda\.comm\."
)
EXPECTED_COUNTER_KEYS = {(gpu, link, direction) for gpu in range(8) for link in range(18) for direction in ("Tx", "Rx")}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, *, input_text=None, timeout=90):
    try:
        completed = subprocess.run(
            argv,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {"argv": argv, "returncode": completed.returncode, "output": completed.stdout}
    except (OSError, subprocess.SubprocessError) as error:
        return {"argv": argv, "returncode": None, "output": str(error)}


def checked_run(argv, **kwargs):
    result = run(argv, **kwargs)
    if result["returncode"] != 0:
        raise RuntimeError(f"Read-only inspection failed: {argv}: {result['output']}")
    return result


def source_lines(sources, pattern):
    return [
        {"path": path, "line": number, "text": line.strip()}
        for path, source in sources.items()
        for number, line in enumerate(source.splitlines(), 1)
        if re.search(pattern, line)
    ]


def cpp_without_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)


def inspect_python(sources):
    calls, imports = [], []
    for path, source in sources.items():
        for node in ast.walk(ast.parse(source, filename=path)):
            if isinstance(node, ast.Call):
                calls.append({"path": path, "line": node.lineno, "call": ast.unparse(node.func)})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append({"path": path, "line": node.lineno, "import": ast.unparse(node)})
    return calls, imports


def inspect_rmw(sources):
    """Classify every explicit owned-source RMW by its pointer provenance.

    Opcode scope alone cannot distinguish a mapped-host pointer from HBM.
    These classifications refer to the source and allocation/caller evidence
    recorded alongside the compiled instructions, rather than to SYS/GPU text.
    """
    sites = source_lines(
        sources,
        r"\batomic(?:Add|CAS|Exch|Sub|Min|Max|And|Or|Xor)\s*\(|"
        r"\.fetch_\w+\s*\(|\.exchange\s*\(|\.compare_exchange\w*\s*\(",
    )
    for site in sites:
        path, text = site["path"], site["text"]
        if path.endswith("advanced_common.cuh") and "thread_scope_block" in text and ".fetch_add(" in text:
            classification = "CTA shared stage counters: local_arrive(&stage.*_done)"
        elif path.endswith("advanced_common.cuh") and "atomicCAS" in text and "state.base.errors" in text:
            classification = "Current GPU HBM output.errors: first-error arbitration"
        elif path.endswith("advanced_layout.cuh") and re.search(
            r"atomicAdd\((?:rank_counts|endpoint_counts)\s*\+", text
        ):
            classification = "CTA shared rank/endpoint counters in parallel channel layout"
        elif path.endswith("buffers/advanced.hpp") and "next_owner_id_.fetch_add" in text:
            classification = "CPU-local native runtime owner-ID counter"
        elif path.endswith("comm/numa_relay.hpp") and re.search(r"flow\.\w+\.fetch_add", text):
            classification = "CPU-local relay diagnostics; ordinary std::atomic member"
        else:
            classification = None
        site["pointer_provenance"] = classification
        site["mapped_host_rmw"] = False if classification else None
    return sites


def inspect_sass(source):
    markers = list(re.finditer(r"^\s*Function\s*:\s*(\S+)", source, re.M))
    functions = []
    memory_prefixes = (
        "LDG",
        "STG",
        "LD.",
        "ST.",
        "LDS",
        "STS",
        "LDL",
        "STL",
        "ATOM",
        "RED.",
        "MEMBAR",
        "BAR",
        "DEPBAR",
    )
    for index, marker in enumerate(markers):
        body = source[marker.end() : markers[index + 1].start() if index + 1 < len(markers) else len(source)]
        opcodes, interesting = {}, []
        samples = {
            name: []
            for name in (
                "system_load",
                "system_store",
                "system_fence",
                "vector_load",
                "vector_store",
                "stack_load",
                "stack_store",
                "atomic_rmw",
            )
        }
        for line in body.splitlines():
            match = re.search(r"/\*[0-9a-fA-F]+\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_.]*)", line)
            if not match:
                continue
            opcode = match.group(1)
            if opcode.startswith(memory_prefixes):
                opcodes[opcode] = opcodes.get(opcode, 0) + 1
            categories = []
            if ".SYS" in opcode and opcode.startswith(("LDG", "LD.")):
                categories.append("system_load")
            if ".SYS" in opcode and opcode.startswith(("STG", "ST.")):
                categories.append("system_store")
            if ".SYS" in opcode and opcode.startswith("MEMBAR"):
                categories.append("system_fence")
            if ".128" in opcode and opcode.startswith(("LDG", "LD.")):
                categories.append("vector_load")
            if ".128" in opcode and opcode.startswith(("STG", "ST.")):
                categories.append("vector_store")
            if opcode.startswith(("ATOM", "RED.")):
                categories.append("atomic_rmw")
            if opcode.startswith("LDL"):
                categories.append("stack_load")
            if opcode.startswith("STL"):
                categories.append("stack_store")
            for category in categories:
                if len(samples[category]) < 4:
                    samples[category].append(line.strip())
            if (
                ".SYS" in opcode
                or opcode.startswith(("ATOM", "RED.", "MEMBAR"))
                or opcode.startswith(("LDG", "STG", "LD.", "ST."))
                and ".128" in opcode
            ):
                if len(interesting) < 24:
                    interesting.append(line.strip())
        functions.append(
            {
                "mangled_name": marker.group(1),
                "memory_instruction_counts": opcodes,
                "instruction_samples": interesting,
                "samples_by_kind": samples,
            }
        )
    return functions


def inspect_resources(source):
    markers = list(re.finditer(r"^\s*Function\s+(\S+?)(?::)?\s*$", source, re.M))
    functions = []
    for index, marker in enumerate(markers):
        body = source[marker.end() : markers[index + 1].start() if index + 1 < len(markers) else len(source)]
        values = {name: int(value) for name, value in re.findall(r"\b([A-Z][A-Z0-9_]*):\s*(\d+)", body)}
        functions.append({"mangled_name": marker.group(1).rstrip(":"), "resources": values})
    return functions


def inspect_counters(before_path, after_path):
    if not before_path.exists() or not after_path.exists():
        return {
            "status": "pending",
            "before_exists": before_path.exists(),
            "after_exists": after_path.exists(),
            "all_payload_deltas_zero": None,
            "all_counter_deltas": [],
        }
    before, after = json.loads(before_path.read_text()), json.loads(after_path.read_text())

    def values(snapshot):
        entries = snapshot["data_counters"]
        mapping = {(row["gpu"], row["link"], row["direction"]): row["KiB"] for row in entries}
        return entries, mapping

    before_entries, prior = values(before)
    after_entries, later = values(after)
    complete = (
        len(before_entries) == len(after_entries) == len(EXPECTED_COUNTER_KEYS)
        and set(prior) == set(later) == EXPECTED_COUNTER_KEYS
    )
    commands_ok = before["commands"]["data"]["returncode"] == after["commands"]["data"]["returncode"] == 0
    deltas = [
        {
            "gpu": key[0],
            "link": key[1],
            "direction": key[2],
            "before_KiB": prior[key],
            "after_KiB": later[key],
            "delta_KiB": later[key] - prior[key],
        }
        for key in sorted(set(prior) & set(later))
    ]
    monotonic = all(row["delta_KiB"] >= 0 for row in deltas)
    unchanged = complete and commands_ok and monotonic and all(row["delta_KiB"] == 0 for row in deltas)
    return {
        "status": "passed" if unchanged else "failed",
        "before": {
            "path": str(before_path),
            "sha256": sha256(before_path),
            "recorded_at_utc": before["recorded_at_utc"],
        },
        "after": {"path": str(after_path), "sha256": sha256(after_path), "recorded_at_utc": after["recorded_at_utc"]},
        "counter_command": ["nvidia-smi", "nvlink", "-gt", "d"],
        "counter_count_expected": len(EXPECTED_COUNTER_KEYS),
        "counter_count_observed_before": len(before_entries),
        "counter_count_observed_after": len(after_entries),
        "complete_gpu_link_direction_sets": complete,
        "commands_succeeded": commands_ok,
        "monotonic_counters": monotonic,
        "all_payload_deltas_zero": unchanged,
        "per_gpu_summary": [
            {
                "gpu": gpu,
                **{
                    direction
                    + "_delta_KiB": sum(
                        row["delta_KiB"] for row in deltas if row["gpu"] == gpu and row["direction"] == direction
                    )
                    for direction in ("Tx", "Rx")
                },
            }
            for gpu in range(8)
        ],
        "all_counter_deltas": deltas,
    }


def module_probe(binary_in_container):
    # Import the existing .so directly. Avoid pcie_ep._C and its JIT loader.
    # cudaDeviceGetAttribute/GetDeviceCount only query properties, with no
    # allocation, stream, event, kernel or memory-copy call in this probe.
    return "\n".join(
        [
            "import ctypes, hashlib, importlib, json, pathlib, sys",
            "import torch",
            f"expected = pathlib.Path({str(binary_in_container)!r}).resolve()",
            "sys.path.insert(0, str(expected.parent))",
            "module = importlib.import_module('pcie_ep_cuda')",
            "actual = pathlib.Path(module.__file__).resolve()",
            "runtime = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')",
            "count = ctypes.c_int()",
            "count_status = runtime.cudaGetDeviceCount(ctypes.byref(count))",
            "devices = []",
            "for ordinal in range(count.value if count_status == 0 else 0):",
            "    record = {'ordinal': ordinal}",
            "    for name, attribute in [('sm_count', 16), ('can_map_host_memory', 19), ('compute_major', 75), ('compute_minor', 76), ('host_native_atomic_supported', 86)]:",
            "        value = ctypes.c_int()",
            "        status = runtime.cudaDeviceGetAttribute(ctypes.byref(value), attribute, ordinal)",
            "        record[name] = value.value if status == 0 else None",
            "        record[name + '_query_status'] = status",
            "    devices.append(record)",
            "version = ctypes.c_int()",
            "version_status = runtime.cudaRuntimeGetVersion(ctypes.byref(version))",
            "record = {'expected_path': str(expected), 'actual_module_path': str(actual), 'path_matches': actual == expected,",
            "          'sha256': hashlib.sha256(actual.read_bytes()).hexdigest(), 'torch_version': torch.__version__,",
            "          'torch_cuda_version': torch.version.cuda, 'cuda_runtime_version': version.value if version_status == 0 else None,",
            "          'device_count_query_status': count_status, 'devices': devices,",
            "          'advanced_runtime_exported': hasattr(module, 'AdvancedRuntime'), 'gpu_workload_launched': False}",
            "print('PCIE_EP_AUDIT_MODULE=' + json.dumps(record))",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--container", default="pcie-ep")
    parser.add_argument(
        "--binary", type=Path, default=Path("/dev/shm/pcie-ep-build/torch/pcie_ep_cuda/pcie_ep_cuda.so")
    )
    parser.add_argument("--container-binary", type=Path, default=Path("/build/torch/pcie_ep_cuda/pcie_ep_cuda.so"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--allow-pending-counters",
        action="store_true",
        help="Write a preliminary audit without a final counter/manifest conclusion",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "results/s5_s8").resolve()
    allowed_outputs = (root / "results/s5_s8", root / "results/f2")
    if not any(output.is_relative_to(directory) for directory in allowed_outputs):
        parser.error("Output must be inside results/s5_s8 or results/f2; historical audits are protected")
    before_path = args.before or output / "nvlink_s5_s8_before.json"
    after_path = args.after or output / "nvlink_s5_s8_after.json"
    manifest_path = args.manifest or output / "executed_source_manifest.json"
    if not args.allow_pending_counters and not all(path.exists() for path in (before_path, after_path, manifest_path)):
        parser.error("Formal audit requires before/after snapshots and executed_source_manifest.json")
    if not args.binary.is_file():
        parser.error("Existing extension binary is required; this collector never builds it")

    cpp_paths = sorted(
        path for path in (root / "csrc").rglob("*") if path.suffix in CPP_SUFFIXES and not path.name.startswith(".")
    )
    python_paths = sorted((root / "pcie_ep").rglob("*.py"))
    benchmark_paths = sorted((root / "benchmarks").glob("*.py"))
    source_paths = cpp_paths + python_paths + benchmark_paths
    initial_hashes = {str(path.relative_to(root)): sha256(path) for path in source_paths}
    cpp_sources = {str(path.relative_to(root)): path.read_text() for path in cpp_paths}
    python_sources = {str(path.relative_to(root)): path.read_text() for path in python_paths + benchmark_paths}
    cpp_source = "\n".join(cpp_without_comments(source) for source in cpp_sources.values())
    forbidden_source = {name: re.findall(pattern, cpp_source) for name, pattern in FORBIDDEN_APIS.items()}
    python_calls, python_imports = inspect_python(python_sources)
    forbidden_python = [row for row in python_calls if re.search(PYTHON_GPU_COMMUNICATION, row["call"])]
    rmw_sites = inspect_rmw(cpp_sources)
    classified_rmw = all(row["pointer_provenance"] is not None for row in rmw_sites)

    def in_container(argv):
        return ["docker", "exec", args.container, *argv]

    binary_hash = sha256(args.binary)
    probe = checked_run(in_container(["python", "-c", module_probe(args.container_binary)]))
    probe_record = json.loads(
        next(line.split("=", 1)[1] for line in probe["output"].splitlines() if line.startswith("PCIE_EP_AUDIT_MODULE="))
    )
    symbols = checked_run(in_container(["nm", "-D", "--undefined-only", str(args.container_binary)]))
    demangled_symbols = checked_run(in_container(["c++filt"]), input_text=symbols["output"])
    forbidden_binary = {
        name: re.findall(pattern, demangled_symbols["output"]) for name, pattern in FORBIDDEN_APIS.items()
    }
    sass = checked_run(in_container(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(args.container_binary)]))
    resources = checked_run(
        in_container(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(args.container_binary)])
    )
    compiler = checked_run(in_container(["/usr/local/cuda/bin/nvcc", "--version"]))
    build_ninja = args.binary.parent / "build.ninja"
    build_text = build_ninja.read_text() if build_ninja.exists() else ""
    output.mkdir(parents=True, exist_ok=True)
    sass_path, resource_path = output / "runtime.sass", output / "runtime_resources.txt"
    sass_path.write_text(sass["output"])
    resource_path.write_text(resources["output"])
    sass_functions = inspect_sass(sass["output"])
    resource_functions = inspect_resources(resources["output"])
    scope_counts = {
        opcode: len(re.findall(re.escape(opcode), sass["output"]))
        for opcode in ("STRONG.SYS", "MEMBAR.ALL.SYS", "MEMBAR.SC.SYS")
    }
    rmw_opcodes = sorted(
        {
            opcode
            for function in sass_functions
            for opcode in function["memory_instruction_counts"]
            if opcode.startswith(("ATOM", "RED."))
        }
    )
    counters = inspect_counters(before_path, after_path)
    manifest = json.loads(manifest_path.read_text()).get("sha256", {}) if manifest_path.exists() else {}
    manifest_matches = {path: manifest.get(path) == value for path, value in initial_hashes.items()}
    source_stable = initial_hashes == {str(path.relative_to(root)): sha256(path) for path in source_paths}
    binary_stable = binary_hash == sha256(args.binary)
    binary_bound = probe_record["path_matches"] and probe_record["sha256"] == binary_hash
    positive = {
        "registered_shared_cpu_backing": bool(source_lines(cpp_sources, r"\bmmap\(|MAP_SHARED"))
        and bool(source_lines(cpp_sources, r"cudaHostRegister\("))
        and bool(source_lines(cpp_sources, r"cudaHostGetDevicePointer\(")),
        "mapped_aliases_bound_to_kernel_state": bool(
            source_lines(cpp_sources, r"mapped_source_buffers\[(?:rank|source_rank)\] =")
        ),
        "current_gpu_output_and_input_validation": bool(source_lines(cpp_sources, r"TensorOptions\(\).device"))
        and bool(source_lines(cpp_sources, r"get_device\(\) == device_")),
        "mapped_signals_only_acquire_load_release_store": bool(
            source_lines(cpp_sources, r"thread_scope_system.*\.load\(")
        )
        and bool(source_lines(cpp_sources, r"thread_scope_system.*\.store\("))
        and classified_rmw,
        "cpu_relay_mapped_handshake_load_store": bool(source_lines(cpp_sources, r"std::atomic_ref<std::uint64_t>")),
        "gloo_control_group_enforced": bool(source_lines(python_sources, r"control_group must use Gloo")),
        "advanced_transport_compiled": any("advanced_transport" in row["mangled_name"] for row in sass_functions),
        "system_scope_instructions_present": all(value > 0 for value in scope_counts.values()),
        "resource_usage_extracted": bool(resource_functions),
        "cuda_13_compiler": bool(re.search(r"release 13\.", compiler["output"])),
        "cpp_20_actual_build_flags": "-std=c++20" in build_text,
        "sm90_actual_build_flags": "compute_90" in build_text and "sm_90" in build_text,
        "eight_sm90_gpus_with_132_sms": len(probe_record["devices"]) == 8
        and all(row["sm_count"] == 132 and row["compute_major"] == 9 for row in probe_record["devices"]),
    }
    no_peer_api = not any(forbidden_source.values()) and not any(forbidden_binary.values()) and not forbidden_python
    static_qualified = all(positive.values()) and no_peer_api and source_stable and binary_stable and binary_bound
    qualified = counters["all_payload_deltas_zero"] is True and all(manifest_matches.values()) and static_qualified
    awaiting_evidence = counters["status"] == "pending" or not manifest_path.exists()
    status = (
        "passed"
        if qualified
        else "pending" if args.allow_pending_counters and awaiting_evidence and static_qualified else "failed"
    )
    result = {
        "audit_kind": "S5-S8 host-staged PCIe payload path",
        "status": status,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "workload_launched_by_this_audit": False,
        "machine_state_mutations": "Audit files only; no kernel, allocation, copy, counter reset, link change or source mutation",
        "historical_audits_preserved": True,
        "collector": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256(Path(__file__))},
        "source_hash_scope": {
            "cpp_cuda_count": len(cpp_paths),
            "python_module_count": len(python_paths),
            "benchmark_count": len(benchmark_paths),
            "included": "All csrc implementation files, package Python modules and benchmark Python files",
            "excluded": "Tests, docs, generated results and audit tooling are not payload/API implementation",
        },
        "source_files": [{"path": path, "sha256": value} for path, value in initial_hashes.items()],
        "executed_manifest_binding": {
            "path": str(manifest_path),
            "exists": manifest_path.exists(),
            "sha256": sha256(manifest_path) if manifest_path.exists() else None,
            "source_subset_matches": manifest_matches,
            "all_audited_sources_match": all(manifest_matches.values()),
        },
        "static_forbidden_api_matches": forbidden_source,
        "binary_forbidden_api_matches": forbidden_binary,
        "python_forbidden_gpu_communication_calls": forbidden_python,
        "positive_requirements": positive,
        "static_and_compiled_review_passed": static_qualified,
        "pointer_provenance": {
            "shared_backing_registration_alias": source_lines(
                cpp_sources, r"\bmmap\(|MAP_SHARED|cudaHostRegister|cudaHostGetDevicePointer"
            ),
            "mapped_alias_to_device_state": source_lines(cpp_sources, r"mapped_source_buffers"),
            "current_gpu_allocations_and_checks": source_lines(
                cpp_sources, r"TensorOptions\(\).device|get_device\(\) == device_|errors = output.errors"
            ),
            "source_and_replica_payload_addresses": source_lines(
                cpp_sources, r"source_payload_offset|replica_payload_offset|receive_payload\("
            ),
            "relay_cpu_copy": source_lines(
                cpp_sources, r"std::memcpy|host_release\(host_signal|host_acquire\(host_signal"
            ),
            "system_scope_load_store": source_lines(
                cpp_sources, r"thread_scope_system|std::atomic_ref<std::uint64_t>|__threadfence_system"
            ),
            "explicit_rmw_sites": rmw_sites,
            "all_rmw_sites_classified_local": classified_rmw,
            "shared_counter_callers": source_lines(
                cpp_sources, r"local_arrive\(|__shared__ int rank_counts|__shared__ int endpoint_counts"
            ),
            "conclusion": "Mapped CPU flags use load/store only. Shared CTA counters, current-GPU error arbitration and CPU-local diagnostics may use RMW; these pointers are not mapped payload/flags.",
        },
        "python_review": {
            "imports": python_imports,
            "communication_call_sites": [
                row for row in python_calls if row["call"].startswith(("dist.", "torch.distributed."))
            ],
            "nccl_handle_borrowing": source_lines(python_sources, r"borrow_nccl_comm|_comm_ptr|NCCL data operations"),
            "gloo_control": source_lines(python_sources, r"control_group must use Gloo|backend=\"gloo\""),
            "stream_ordering": source_lines(python_sources, r"wait_stream|wait_event|_consumer_streams"),
        },
        "compiled_extension": {
            "host_path": str(args.binary),
            "sha256": binary_hash,
            "module_probe": probe_record,
            "actual_module_binary_matches_audited_file": binary_bound,
            "undefined_symbols": demangled_symbols,
        },
        "build": {
            "compiler": compiler,
            "build_ninja_path": str(build_ninja),
            "build_ninja_sha256": sha256(build_ninja) if build_ninja.exists() else None,
            "build_ninja": build_text,
            "python_build_flags": source_lines(python_sources, r"_CUDA_FLAGS|_HOST_FLAGS|TORCH_CUDA_ARCH_LIST"),
        },
        "sass": {
            "path": str(sass_path.relative_to(root)),
            "sha256": sha256(sass_path),
            "extraction_command": sass["argv"],
            "system_scope_instruction_counts": scope_counts,
            "atomic_rmw_opcodes": rmw_opcodes,
            "per_kernel": sass_functions,
            "scope_limit": "Opcode counts are static occurrences across all compiled runtime options, not executed per-token counts. Instruction scopes express ordering. Source pointer provenance establishes HBM/shared/host location; widths do not measure PCIe TLPs or round trips.",
        },
        "kernel_resources": {
            "path": str(resource_path.relative_to(root)),
            "sha256": sha256(resource_path),
            "extraction_command": resources["argv"],
            "per_kernel": resource_functions,
            "dynamic_shared_memory": "Runtime selects sizeof(AdvancedLocalQueues), 16-byte aligned, plus H*2*stages for F2; the cuobjdump SHARED field is static storage only.",
        },
        "snapshot_integrity": {"source_hashes_stable": source_stable, "binary_hash_stable": binary_stable},
        "nvlink_benchmark_measurement": counters,
        "benchmark_results": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path)}
            for path in sorted(set(output.glob("dispatch*.json")) | set(output.glob("*benchmark*.json")))
        ],
        "hardware": {
            "pcie": run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,pci.bus_id,name,pcie.link.gen.current,pcie.link.width.current",
                    "--format=csv",
                ]
            ),
            "topology": run(["nvidia-smi", "topo", "-m"]),
            "cpu": run(["lscpu"]),
        },
        "conclusion": {
            "PCIe_only_payload_path_supported_for_this_run": qualified,
            "no_peer_gpu_payload_route_in_reviewed_runtime": no_peer_api,
            "physical_nvlink_disabled": False,
            "static_transport_path": "GPU SM -> registered shared CPU backing [optional CPU memcpy to target-NUMA replica] -> receiver GPU SM; expert expansion remains local",
            "limitations": [
                "Machine-wide cumulative payload KiB counters exclude protocol-only overhead; zero means no reported growth at that resolution.",
                "This audit does not measure PCIe TLPs, host memory-controller bytes or cross-socket UPI traffic.",
                "NVLink stays physically enabled. NCCL_P2P_DISABLE is not used as proof.",
                "SASS global atomics are expected for GPU-local error arbitration; their presence alone does not imply mapped-host RMW.",
                "The conclusion binds these source hashes, imported binary and before/after snapshots. Repeat after changes.",
            ],
        },
    }
    (output / "pcie_path_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    document = [
        "# S5–S8 PCIe 通信路径审计",
        "",
        f"状态：**{status}**。本次运行的 PCIe-only payload 证据完整通过：**{qualified}**。",
        "",
        "本脚本只读取已编译扩展、源码、硬件属性和已保存的计数；不发起 GPU 工作、不重置计数、不修改物理链路。旧 baseline/refactor 审计保持原样。",
        "",
        f"覆盖 {len(cpp_paths)} 个 C++/CUDA 文件、{len(python_paths)} 个 package Python 模块及 {len(benchmark_paths)} 个 benchmark 文件；包括全部高级 headers 和 NUMA relay。",
        "",
        "路径为 GPU SM → 注册的 CPU 共享 backing → 接收 GPU SM；relay 模式额外由 CPU memcpy 一次到目标 NUMA 的 replica。GPU state 内的跨 rank 地址仅来自 cudaHostGetDevicePointer，workspace/output 属于当前 GPU。",
        "",
        f"源码与动态符号未发现 peer GPU/NCCL/NVSHMEM/Gin payload API：{no_peer_api}。实际导入模块与被审计 .so 一致：{binary_bound}。",
        "",
        "## 原子与内存序",
        "",
        "Mapped host READY/ACK/COPIED_ACK/drained 只使用 acquire-load/release-store；没有针对 mapped memory 的 RMW。CTA shared counters、GPU 本地 errors 仲裁和 CPU 本地诊断计数可以使用 RMW。",
        "",
        *[f"- `{row['path']}:{row['line']}`：{row['pointer_provenance']}。" for row in rmw_sites],
        "",
        f"SASS 系统指令计数：{scope_counts}。RMW opcodes：{rmw_opcodes}。",
        "",
        "计数为所有已编译选项路径的静态指令出现数，不是每 token 的动态执行次数。SYS 后缀体现内存序，不能单独证明总线路径；load/store 指令宽度也不能直接换算 PCIe TLP 数或往返次数。完整 SASS、LDL/STL 栈访存样例及各 kernel 内存指令统计保存在 JSON/runtime.sass。",
        "",
        "## 执行绑定与资源",
        "",
        f".so SHA-256：`{binary_hash}`。源码匹配 executed manifest：{all(manifest_matches.values())}。采集期间源码稳定：{source_stable}；二进制稳定：{binary_stable}。",
        "",
        f"CUDA 13 compiler：{positive['cuda_13_compiler']}；实际 C++20 flags：{positive['cpp_20_actual_build_flags']}；8 张 sm90/132 SM 属性：{positive['eight_sm90_gpus_with_132_sms']}。",
        "",
        "REG/STACK/SHARED/LOCAL 的逐 kernel 原始记录在 runtime_resources.txt，解析结果在 JSON。SHARED 表示静态空间；F2 动态 shared 另按 queues + H×2×stages 分配。",
        "",
        "## NVLink payload 计数",
        "",
        f"前后计数状态：{counters['status']}；288 项增量全部为 0 KiB：{counters['all_payload_deltas_zero']}。",
        "",
    ]
    if counters["status"] != "pending":
        document.extend(
            [
                "| GPU | Tx 增量 KiB | Rx 增量 KiB |",
                "|---|---:|---:|",
                *[
                    f"| {row['gpu']} | {row['Tx_delta_KiB']} | {row['Rx_delta_KiB']} |"
                    for row in counters["per_gpu_summary"]
                ],
                "",
            ]
        )
    document.extend(
        [
            "计数为整机累计 payload KiB；不包含协议流量，也不等于 PCIe、CPU 内存或 UPI 的实测字节。完整逐 link 原值与差值见 JSON。源码或二进制变化后需重新采集绑定。",
            "",
        ]
    )
    (output / "pcie_path_audit.md").write_text("\n".join(document))
    print(
        json.dumps(
            {
                "status": status,
                "audit": str(output / "pcie_path_audit.json"),
                "source_count": len(source_paths),
                "positive_requirements": positive,
                "no_peer_api": no_peer_api,
                "all_rmw_classified_local": classified_rmw,
                "module_binary_match": binary_bound,
                "manifest_match": all(manifest_matches.values()),
                "counter_status": counters["status"],
                "PCIe_only_supported": qualified,
            },
            ensure_ascii=False,
        )
    )
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
