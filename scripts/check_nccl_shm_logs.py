"""Validate and anonymize NCCL 2.29 per-process INFO transport logs.

The formats below are verified against NVIDIA/nccl tag v2.29.7-1:
  src/debug.cc:349-362 (prefix contains CUDA device, not communicator rank)
  src/init.cc:1581 (rank/nranks/commId and Init COMPLETE)
  src/transport/shm.cc:118 (send-side Channel line, four CE/direct variants)
  src/transport/net.cc:287-328 and p2p.cc:373-393 (rejected transports)

SHM connection lines do not carry a communicator ID. This collector therefore
requires one initialized communicator per process log, and rejects ambiguous
logs. Multiple runs with separate log files and communicator IDs are supported.
No raw line, hostname, PID, address, communicator ID or path enters the result.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


SOURCE_TAG = "v2.29.7-1"
SOURCE_URLS = [
    "https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/" + path
    for path in ("debug.cc", "init.cc", "transport/shm.cc", "transport/net.cc", "transport/p2p.cc", "transport/nvls.cc")
]
INFO = re.compile(r"(?P<host>[^\s:]+):(?P<pid>\d+):\d+\s+\[\d+\]\s+NCCL INFO\s+(?P<message>.*)$")
VERSION = re.compile(r"\bNCCL version\s+(\d+\.\d+\.\d+)(?:\+[^\s]+)?")
INIT = re.compile(
    r"\bncclComm\w+\s+comm\s+\S+\s+rank\s+(?P<rank>\d+)\s+nranks\s+(?P<nranks>\d+)\s+"
    r".*\bcommId\s+(?P<comm>0x[0-9a-fA-F]+)\s+-\s+Init COMPLETE\s*$"
)
CONNECTION = re.compile(
    r"^Channel\s+(?P<channel>\d+)(?:/(?P<index>\d+))?\s*:\s*"
    r"(?P<source>\d+)\[\d+\]\s*->\s*(?P<destination>\d+)\[\d+\]"
    r"(?:\s+\[(?P<direction>send|receive)\])?\s+via\s+(?P<transport>.+?)\s*$"
)
SHM_TRANSPORT = re.compile(r"SHM/(?:CE|direct)/(?:CE|direct)")
ACTIVE_NVLS = re.compile(
    r"\bNVLS (?:Creating Multicast group|Created Multicast group|importing shareableHandle|comm\s|rank .* alloc done)"
    r"|\bConnected NVLS tree\b"
)


def transport_name(value):
    """Never copy an arbitrary transport tail (which may contain addresses)."""
    if SHM_TRANSPORT.fullmatch(value):
        return value
    family = value.split("/", 1)[0].split()[0] if value.split() else "UNKNOWN"
    return family if family in {"SHM", "P2P", "NET", "NVLS", "COLLNET", "GIN"} else "UNKNOWN"


def analyze_logs(blobs, world=8):
    """Return anonymous evidence for byte strings, without reading GPU state."""
    if world < 2:
        raise ValueError("world must be at least two")
    errors, logs, pending, init_records = [], [], [], []
    communicator_ids = []

    def fail(code, **fields):
        errors.append({"code": code, **fields})

    for log_index, blob in enumerate(blobs):
        text = blob.decode("utf-8", errors="replace")
        identities, versions, local_inits, local_connections = set(), set(VERSION.findall(text)), [], []
        candidate_count = 0
        for line_number, line in enumerate(text.splitlines(), 1):
            is_connection = "NCCL INFO" in line and bool(re.search(r"\bChannel\b.*\bvia\b", line))
            is_init = "NCCL INFO" in line and "Init COMPLETE" in line
            if not (is_connection or is_init or "NCCL INFO" in line and ACTIVE_NVLS.search(line)):
                continue
            match = INFO.search(line)
            if is_connection:
                candidate_count += 1
            if match is None:
                fail("unrecognized_logger_prefix", log_index=log_index, line=line_number)
                continue
            identities.add((match["host"], match["pid"]))
            message = match["message"]
            if ACTIVE_NVLS.search(message):
                fail("active_nvls", log_index=log_index, line=line_number)
            if is_init:
                initialized = INIT.search(message)
                if initialized is None:
                    fail("unrecognized_init_complete", log_index=log_index, line=line_number)
                else:
                    record = {"rank": int(initialized["rank"]), "nranks": int(initialized["nranks"]),
                              "comm": initialized["comm"].lower(), "log_index": log_index, "line": line_number}
                    local_inits.append(record)
                    init_records.append(record)
                    if record["comm"] not in communicator_ids:
                        communicator_ids.append(record["comm"])
            if is_connection:
                connection = CONNECTION.fullmatch(message)
                if connection is None:
                    fail("unrecognized_connection", log_index=log_index, line=line_number)
                    continue
                raw_transport = connection["transport"]
                record = {"log_index": log_index, "line": line_number, "source": int(connection["source"]),
                          "destination": int(connection["destination"]), "channel": int(connection["channel"]),
                          "connection_index": int(connection["index"]) if connection["index"] else None,
                          "direction": connection["direction"] or "send", "transport": transport_name(raw_transport)}
                local_connections.append(record)
                if not SHM_TRANSPORT.fullmatch(raw_transport):
                    fail("non_shm_connection", log_index=log_index, line=line_number, transport=record["transport"])
                if not (0 <= record["source"] < world and 0 <= record["destination"] < world):
                    fail("connection_rank_out_of_range", log_index=log_index, line=line_number)
                if record["source"] == record["destination"]:
                    fail("self_connection", log_index=log_index, line=line_number)

        if len(identities) != 1:
            fail("one_process_per_log_required", log_index=log_index)
        if len(local_inits) != 1:
            fail("one_initialized_communicator_per_log_required", log_index=log_index, observed=len(local_inits))
        if not versions or any(not version.startswith("2.29.") for version in versions):
            fail("missing_or_unsupported_nccl_version", log_index=log_index)
        if not local_connections:
            fail("missing_connections", log_index=log_index)
        logs.append({"log_index": log_index, "sha256": hashlib.sha256(blob).hexdigest(),
                     "nccl_versions": sorted(versions), "init_complete_count": len(local_inits),
                     "connection_candidate_count": candidate_count, "parsed_connection_count": len(local_connections)})
        if len(local_inits) == 1:
            initialized = local_inits[0]
            for record in local_connections:
                record["comm"] = initialized["comm"]
                record["rank"] = initialized["rank"]
                owner = record["destination"] if record["direction"] == "receive" else record["source"]
                if owner != record["rank"]:
                    fail("connection_owner_differs_from_initialized_rank", log_index=log_index, line=record["line"])
                pending.append(record)

    if not blobs:
        fail("no_logs")
    communicators, connections = [], []
    expected = set(range(world))
    for comm_index, comm in enumerate(communicator_ids):
        initialized = [row for row in init_records if row["comm"] == comm]
        counts = Counter(row["rank"] for row in initialized)
        ranks = set(counts)
        selected = [row for row in pending if row["comm"] == comm]
        sender_ranks = {row["source"] for row in selected if row["direction"] == "send"}
        destination_ranks = {row["destination"] for row in selected}
        if ranks != expected or any(count != 1 for count in counts.values()):
            fail("communicator_requires_one_init_per_rank", communicator=comm_index,
                 missing_ranks=sorted(expected - ranks), duplicate_ranks=sorted(rank for rank, count in counts.items() if count > 1))
        if any(row["nranks"] != world for row in initialized):
            fail("communicator_size_mismatch", communicator=comm_index)
        if sender_ranks != expected:
            fail("communicator_missing_sender_ranks", communicator=comm_index, missing_ranks=sorted(expected - sender_ranks))
        if destination_ranks != expected:
            fail("communicator_missing_destination_ranks", communicator=comm_index,
                 missing_ranks=sorted(expected - destination_ranks))
        for row in selected:
            connections.append({"communicator": comm_index, **{key: value for key, value in row.items() if key != "comm"}})
        communicators.append({"communicator": comm_index, "initialized_ranks": sorted(ranks),
                              "init_complete_count": len(initialized), "sender_ranks": sorted(sender_ranks),
                              "destination_ranks": sorted(destination_ranks), "connection_count": len(selected),
                              "channels": sorted({row["channel"] for row in selected}),
                              "transport_counts": dict(sorted(Counter(row["transport"] for row in selected).items()))})

    candidates = sum(row["connection_candidate_count"] for row in logs)
    parsed = sum(row["parsed_connection_count"] for row in logs)
    all_shm = bool(connections) and all(SHM_TRANSPORT.fullmatch(row["transport"]) for row in connections)
    complete = candidates == parsed == len(connections) and bool(communicators)
    return {"schema_version": 1, "status": "passed" if not errors and complete and all_shm else "failed",
            "expected_world_size": world, "format_source_tag": SOURCE_TAG, "format_sources": SOURCE_URLS,
            "logs": logs, "communicators": communicators, "connections": connections,
            "all_logged_data_connections_are_shm": all_shm,
            "all_connection_candidates_parsed_and_bound": complete, "errors": errors,
            "limitations": ["Covers every Channel/via INFO record in the supplied complete per-process logs.",
                            "Connection records lack communicator IDs; one initialized communicator per process log is required.",
                            "Bootstrap network discovery is allowed and is distinct from a Channel/via NET data connection.",
                            "Logs do not replace payload correctness checks or before/after NVLink counters."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, default=8)
    args = parser.parse_args()
    blobs, read_errors = [], []
    for log_index, path in enumerate(args.logs):
        try:
            blobs.append(path.read_bytes())
        except OSError:
            read_errors.append({"code": "unreadable_log", "log_index": log_index})
            blobs.append(b"")
    report = analyze_logs(blobs, args.world)
    if read_errors:
        report["errors"].extend(read_errors)
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "logs": len(report["logs"]),
                      "communicators": len(report["communicators"]), "connections": len(report["connections"]),
                      "error_count": len(report["errors"])}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
