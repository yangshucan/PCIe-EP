"""CPU-only checks for the exact NCCL-reference path audit exception."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script("collect_advanced_path_audit")
manifest = load_script("source_manifest")


class NCCLAuditScopeTests(unittest.TestCase):
    def test_only_exact_reference_entry_is_exempt(self):
        paths = (
            "benchmarks/benchmark_nccl_shm.py",
            "pcie_ep/benchmark_nccl_shm.py",
            "pcie_ep/runtime.py",
            "benchmarks/benchmark.py",
            "benchmarks/nccl_expand.py",
            "benchmarks/benchmark_nccl_shm_copy.py",
            "benchmarks/nested/benchmark_nccl_shm.py",
        )
        calls, _ = audit.inspect_python({path: "dist.all_gather_into_tensor(output, value)" for path in paths})
        forbidden = audit.forbidden_python_calls(calls)
        self.assertEqual({row["path"] for row in forbidden}, set(paths[1:]))
        self.assertEqual(len(forbidden), len(paths) - 1)

    def test_other_forbidden_calls_keep_original_policy(self):
        sources = {
            "pcie_ep/runtime.py": "dist.all_reduce(x)\ntorch.cuda.comm.broadcast(x)\ndist.all_gather_object(items, value)",
            "benchmarks/benchmark.py": "torch.distributed.all_gather_into_tensor(output, value)",
        }
        calls, _ = audit.inspect_python(sources)
        self.assertEqual(
            {row["call"] for row in audit.forbidden_python_calls(calls)},
            {"dist.all_reduce", "torch.cuda.comm.broadcast", "torch.distributed.all_gather_into_tensor"},
        )

    def test_production_reference_imports_are_rejected(self):
        imports = (
            "import benchmark_nccl_shm",
            "import benchmarks.benchmark_nccl_shm as reference",
            "import nccl_expand as expansion",
            "import benchmarks.nccl_expand",
            "from benchmarks import nccl_expand as expansion",
            "from benchmarks import benchmark_nccl_shm",
            "from benchmarks.nccl_expand import NCCLExpand",
            "from . import nccl_expand",
            "from .nccl_expand import NCCLExpand",
            "from .benchmark_nccl_shm import run",
        )
        for source in imports:
            with self.subTest(source=source):
                violations = audit.inspect_production_reference_imports({"pcie_ep/runtime.py": source})
                self.assertEqual(len(violations), 1)
                self.assertEqual(violations[0]["path"], "pcie_ep/runtime.py")
                self.assertEqual(violations[0]["line"], 1)

    def test_literal_dynamic_reference_imports_are_rejected(self):
        imports = (
            "importlib.import_module('benchmarks.nccl_expand')",
            "import importlib as loader\nloader.import_module('nccl_expand')",
            "from importlib import import_module as load\nload('benchmarks.benchmark_nccl_shm')",
            "__import__('benchmark_nccl_shm')",
            "import builtins as loader\nloader.__import__('benchmarks.nccl_expand')",
        )
        for source in imports:
            with self.subTest(source=source):
                self.assertEqual(len(audit.inspect_production_reference_imports({"pcie_ep/runtime.py": source})), 1)

    def test_unrelated_import_names_and_reference_owned_imports_are_allowed(self):
        sources = {
            "pcie_ep/runtime.py": (
                "import torch\nfrom . import runtime\nfrom benchmarks import helper as nccl_expand\n"
                "import nccl_expand_other\nimportlib.import_module('pcie_ep.runtime')"
            ),
            "benchmarks/benchmark_nccl_shm.py": "from nccl_expand import NCCLExpand",
            "benchmarks/nccl_expand.py": "import torch",
        }
        self.assertEqual(audit.inspect_production_reference_imports(sources), [])

    def test_exact_exception_records_hash_and_limited_policy(self):
        hashes = {path: hashlib.sha256(path.encode()).hexdigest() for path in audit.REFERENCE_ARTIFACTS}
        scope = audit.python_policy_scope(hashes)
        exclusions = scope["reference_only_exclusions"]
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions[0]["path"], "benchmarks/benchmark_nccl_shm.py")
        self.assertEqual(exclusions[0]["sha256"], hashes[exclusions[0]["path"]])
        self.assertIn("Independent NCCL SHM", exclusions[0]["reason"])
        self.assertTrue(scope["reference_sources_hashed_and_manifest_bound"])
        self.assertFalse(scope["production_reference_imports_allowed"])
        self.assertIn("exact path", scope["exclusion_applies_to"])

    def test_helper_and_exempt_entry_remain_hashed_and_manifest_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "csrc/runtime.cu": "// production runtime\n",
                "pcie_ep/_C.py": "sources = ['runtime.cu']\n",
                "benchmarks/benchmark.py": "# production benchmark\n",
                "benchmarks/benchmark_nccl_shm.py": "dist.all_gather_into_tensor(output, value)\n",
                "benchmarks/nccl_expand.py": "# independent loader\n",
                "benchmarks/csrc/nccl_expand.cu": "// independent local GPU helper\n",
            }
            for name, source in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            cpp_paths, _, _, source_paths = audit.collect_source_paths(root)
            self.assertEqual([str(path.relative_to(root)) for path in cpp_paths], ["csrc/runtime.cu"])
            hashes = {str(path.relative_to(root)): audit.sha256(path) for path in source_paths}
            self.assertEqual(set(hashes), set(files))
            self.assertEqual(hashes, manifest.source_manifest(root))
            previous = dict(hashes)
            helper = root / "benchmarks/csrc/nccl_expand.cu"
            helper.write_text("// changed independent helper\n")
            self.assertNotEqual(audit.sha256(helper), previous[str(helper.relative_to(root))])
            self.assertNotEqual(manifest.source_manifest(root), previous)

    def test_real_production_package_does_not_import_reference_modules(self):
        sources = {str(path.relative_to(ROOT)): path.read_text() for path in (ROOT / "pcie_ep").rglob("*.py")}
        self.assertEqual(audit.inspect_production_reference_imports(sources), [])


if __name__ == "__main__":
    unittest.main()
