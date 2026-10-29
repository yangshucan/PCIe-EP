"""CPU-only anonymous NCCL 2.29 transport-log validation checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_nccl_shm_logs", ROOT / "scripts/check_nccl_shm_logs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def rank_log(rank, transport="SHM/direct/direct", comm="0xabc", cuda_device=None):
    device = rank if cuda_device is None else cuda_device
    prefix = f"private-host:123{rank}:456{rank} [{device}] NCCL INFO "
    lines = [prefix + "NCCL version 2.29.7+cuda13.0",
             prefix + "Bootstrap : Using private-net:192.0.2.123<0>",
             prefix + "Using network Socket",
             prefix + "NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)",
             prefix + f"Channel 00 : {rank}[{rank}] -> {(rank + 1) % 8}[{(rank + 1) % 8}] via {transport}",
             prefix + f"ncclCommInitRankConfig comm 0x123 rank {rank} nranks 8 cudaDev {device} "
                      f"nvmlDev {rank} busId 1000 commId {comm} - Init COMPLETE"]
    return ("\n".join(lines) + "\n").encode()


def valid_logs():
    return [rank_log(rank) for rank in range(8)]


class NCCLSHMLogsTests(unittest.TestCase):
    def test_all_four_official_shm_suffixes_pass(self):
        for transport in ("SHM/direct/direct", "SHM/direct/CE", "SHM/CE/direct", "SHM/CE/CE"):
            with self.subTest(transport=transport):
                logs = [rank_log(rank, transport) for rank in range(8)]
                result = checker.analyze_logs(logs)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(len(result["connections"]), 8)
                self.assertTrue(result["all_connection_candidates_parsed_and_bound"])
                self.assertEqual(result["logs"][0]["sha256"], hashlib.sha256(logs[0]).hexdigest())

    def test_cuda_device_prefix_is_not_mistaken_for_rank(self):
        result = checker.analyze_logs([rank_log(rank, cuda_device=0) for rank in range(8)])
        self.assertEqual(result["status"], "passed")
        self.assertEqual([row["rank"] for row in result["connections"]], list(range(8)))

    def test_p2p_net_nvls_and_unknown_transports_fail(self):
        for transport in ("P2P/IPC", "P2P/CUMEM/read/CE", "P2P/direct pointer", "NET/Socket/0", "NET/IB/0/GDRDMA",
                          "NVLS", "SHM/unknown/direct", "SHM", "other/private-host:192.0.2.123"):
            with self.subTest(transport=transport):
                logs = valid_logs()
                logs[3] = rank_log(3, transport)
                result = checker.analyze_logs(logs)
                self.assertEqual(result["status"], "failed")
                self.assertIn("non_shm_connection", {row["code"] for row in result["errors"]})

    def test_net_connection_index_and_receive_tag_are_parsed_then_rejected(self):
        logs = valid_logs()
        logs[3] = rank_log(3, "NET/Socket/0").replace(b"Channel 00 :", b"Channel 00/1 :").replace(b" via NET", b" [send] via NET")
        result = checker.analyze_logs(logs)
        row = result["connections"][3]
        self.assertEqual((row["connection_index"], row["direction"], row["transport"]), (1, "send", "NET"))
        self.assertEqual(result["status"], "failed")

    def test_missing_rank_connection_and_init_fail(self):
        cases = [valid_logs()[:-1], valid_logs(), valid_logs()]
        cases[1][3] = b"\n".join(line for line in cases[1][3].splitlines() if b"Channel" not in line)
        cases[2][3] = b"\n".join(line for line in cases[2][3].splitlines() if b"Init COMPLETE" not in line)
        for logs in cases:
            with self.subTest(logs=len(logs)):
                self.assertEqual(checker.analyze_logs(logs)["status"], "failed")

    def test_each_communicator_requires_its_own_eight_ranks(self):
        logs = valid_logs() + [rank_log(rank, comm="0xdef") for rank in range(7)]
        result = checker.analyze_logs(logs)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["communicators"]), 2)
        self.assertIn("communicator_requires_one_init_per_rank", {row["code"] for row in result["errors"]})
        complete = checker.analyze_logs(logs + [rank_log(7, comm="0xdef")])
        self.assertEqual(complete["status"], "passed")

    def test_duplicate_rank_or_ambiguous_process_log_fails(self):
        logs = valid_logs()
        logs[7] = rank_log(6)
        self.assertEqual(checker.analyze_logs(logs)["status"], "failed")
        logs = valid_logs()
        logs[0] += rank_log(0, comm="0xdef")
        self.assertIn("one_initialized_communicator_per_log_required",
                      {row["code"] for row in checker.analyze_logs(logs)["errors"]})

    def test_unrecognized_channel_record_and_active_nvls_fail(self):
        logs = valid_logs()
        logs[0] += b"private-host:1230:4560 [0] NCCL INFO Channel malformed via SHM/direct/direct\n"
        self.assertIn("unrecognized_connection", {row["code"] for row in checker.analyze_logs(logs)["errors"]})
        logs = valid_logs()
        logs[0] += b"private-host:1230:4560 [0] NCCL INFO NVLS Creating Multicast group nranks 8 size 1024 on rank 0\n"
        self.assertIn("active_nvls", {row["code"] for row in checker.analyze_logs(logs)["errors"]})

    def test_anonymous_output_keeps_no_host_address_pid_path_or_comm_id(self):
        result = json.dumps(checker.analyze_logs(valid_logs()))
        for private in ("private-host", "private-net", "192.0.2.123", "1230", "0xabc", "0x123", "/root/"):
            self.assertNotIn(private, result)

    def test_cli_writes_anonymous_result_and_returns_failure_for_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for rank, blob in enumerate(valid_logs()):
                path = root / ("private-host-rank" + str(rank) + ".log")
                path.write_bytes(blob)
                paths.append(str(path))
            output = root / "summary.json"
            command = [sys.executable, str(ROOT / "scripts/check_nccl_shm_logs.py"), *paths, "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn(str(root), completed.stdout)
            self.assertEqual(json.loads(output.read_text())["status"], "passed")
            missing = root / "private-missing.log"
            command = [sys.executable, str(ROOT / "scripts/check_nccl_shm_logs.py"), str(missing), "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertNotIn(str(missing), completed.stdout + completed.stderr + output.read_text())


if __name__ == "__main__":
    unittest.main()
