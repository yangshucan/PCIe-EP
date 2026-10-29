"""CPU-only orchestration checks; all GPU commands are isolated stubs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NCCLSHMDriverTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("scripts", "tests", "benchmarks/data", "bin"):
            (self.root / name).mkdir(parents=True)
        shutil.copy(ROOT / "scripts/benchmark_nccl_shm.sh", self.root / "scripts")
        (self.root / "benchmarks/data/nccl_shm_routes.json").write_text('{"legacy": true}\n')
        self.log = self.root / "commands.jsonl"
        self.env = dict(os.environ, PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                        DRIVER_TEST_LOG=str(self.log))
        self.write_executable("bin/nvidia-smi", """
import sys
if any('--query-gpu=' in value for value in sys.argv):
    print('\\n'.join(f'{rank}, 0, 0' for rank in range(8)))
""")
        self.write_executable("bin/torchrun", """
import json
import os
from pathlib import Path
import sys
with Path(os.environ['DRIVER_TEST_LOG']).open('a') as output:
    output.write(json.dumps(dict(tool='torchrun', args=sys.argv[1:]))+'\\n')
destination = Path(sys.argv[sys.argv.index('--output')+1])
destination.with_suffix('.json').write_text('{}')
if destination.name == 'round1' and os.environ.get('DRIVER_MUTATE_PATH'):
    with Path(os.environ['DRIVER_MUTATE_PATH']).open('ab') as output:
        output.write(b' ')
log = Path(os.environ['NCCL_DEBUG_FILE'].replace('%p', str(os.getpid())))
log.write_text('isolated test stub')
""")
        (self.root / "tests/capture_path_evidence.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        self.write_executable("scripts/source_manifest.py", """
from pathlib import Path
import sys
Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')
""")
        command_stub = """
import json
import os
from pathlib import Path
import sys
with Path(os.environ['DRIVER_TEST_LOG']).open('a') as output:
    output.write(json.dumps(dict(tool=Path(sys.argv[0]).name, args=sys.argv[1:]))+'\\n')
"""
        for name in ("scripts/check_path_counters.py", "scripts/check_nccl_shm_logs.py",
                     "benchmarks/report_nccl_shm.py"):
            self.write_executable(name, command_stub)

    def write_executable(self, name, source):
        path = self.root / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)

    def fixture(self):
        directory = self.root / "fixture with spaces"
        directory.mkdir()
        tensors = directory / "routes.npz"
        tensors.write_bytes(b"fixed offline routing tensors")
        manifest = directory / "routes.json"
        manifest.write_text(json.dumps(dict(schema_version=2, data_file=tensors.name,
                                           tensors_sha256=hashlib.sha256(tensors.read_bytes()).hexdigest())))
        return manifest, tensors

    def run_driver(self, *arguments):
        return subprocess.run(["bash", str(self.root / "scripts/benchmark_nccl_shm.sh"), *map(str, arguments)],
                              cwd=self.root, env=self.env, text=True, capture_output=True, timeout=20)

    def commands(self, tool):
        records = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return [record["args"] for record in records if record["tool"] == tool]

    def test_legacy_default_topk_eight_and_explicit_report_fixture(self):
        result = self.run_driver("results/legacy")
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = self.commands("torchrun")
        self.assertEqual(len(launches), 2)
        for arguments in launches:
            self.assertEqual(arguments[arguments.index("--topk")+1], "8")
            self.assertNotIn("--routing-fixture", arguments)
        report = self.commands("report_nccl_shm.py")[0]
        self.assertEqual(report[report.index("--route-fixture")+1],
                         str(self.root / "benchmarks/data/nccl_shm_routes.json"))
        manifest = json.loads((self.root / "results/legacy/run_manifest.json").read_text())
        self.assertEqual(manifest["routing_selection"], "legacy_seeded")
        self.assertTrue((self.root / "results/legacy/executed_source_manifest.json").is_file())

    def test_topk_four_binds_both_rounds_and_report_to_exact_manifest(self):
        fixture, tensors = self.fixture()
        result = self.run_driver("results/four", "--topk", "4", "--routing-fixture", fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        for arguments in self.commands("torchrun"):
            self.assertEqual(arguments[arguments.index("--topk")+1], "4")
            self.assertEqual(arguments[arguments.index("--routing-fixture")+1], str(fixture.resolve()))
        report = self.commands("report_nccl_shm.py")[0]
        self.assertEqual(report[report.index("--route-fixture")+1], str(fixture.resolve()))
        manifest = json.loads((self.root / "results/four/run_manifest.json").read_text())
        self.assertEqual(manifest["route_fixture_sha256"], hashlib.sha256(fixture.read_bytes()).hexdigest())
        self.assertEqual(manifest["route_tensors_sha256"], hashlib.sha256(tensors.read_bytes()).hexdigest())
        self.assertEqual(manifest["topk"], 4)

    def test_missing_fixture_and_invalid_arguments_fail_before_launch(self):
        for arguments in (("--topk4",), ("--topk", "4"), ("--topk",),
                          ("--topk", "3"), ("--routing-fixture",), ("--unknown",)):
            with self.subTest(arguments=arguments):
                result = self.run_driver(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.commands("torchrun"), [])

    def test_fixture_mutation_after_first_round_prevents_report(self):
        for item in ("manifest", "tensors"):
            with self.subTest(item=item), tempfile.TemporaryDirectory() as fixture_directory:
                tensors = Path(fixture_directory) / "routes.npz"
                tensors.write_bytes(b"immutable")
                manifest = Path(fixture_directory) / "routes.json"
                manifest.write_text(json.dumps(dict(data_file=tensors.name,
                                                   tensors_sha256=hashlib.sha256(tensors.read_bytes()).hexdigest())))
                self.env["DRIVER_MUTATE_PATH"] = str(manifest if item == "manifest" else tensors)
                if self.log.exists():
                    self.log.unlink()
                result = self.run_driver("results/changed-"+item, "--topk4", "--routing-fixture", manifest)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("changed during the comparison", result.stderr)
                self.assertEqual(len(self.commands("torchrun")), 1)
                self.assertEqual(self.commands("report_nccl_shm.py"), [])

    def test_bad_tensor_digest_fails_before_launch(self):
        manifest, tensors = self.fixture()
        tensors.write_bytes(b"changed before run")
        result = self.run_driver("results/bad", "--topk4", "--routing-fixture", manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tensor hash does not match", result.stderr)
        self.assertEqual(self.commands("torchrun"), [])


if __name__ == "__main__":
    unittest.main()
