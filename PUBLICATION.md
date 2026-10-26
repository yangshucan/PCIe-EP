# Public repository scope

This repository contains the PCIe host-staged dispatch implementation, Python API,
tests, generic execution scripts, design documents, and the latest pipelined F2
performance table. The bandwidth formula is `T*K*H*b/t`, with input tokens per
GPU, BF16 `b=2` bytes, and decimal GB/s.

Before publication, machine names, personal checkout paths, private workspaces,
container image identifiers, and hardcoded execution parameters were removed or
replaced with caller-provided configuration. No known embedded credentials or
non-loopback IP address literals were found in the published files.

The ignored `results/` directory is private local output. It can contain machine
identity and topology diagnostics, process information, raw logs, profiler traces,
source archives and native binaries. These outputs are not included in Git.
Validation and audit tools intentionally collect such diagnostics for local
debugging; their output must be reviewed and exported selectively before sharing.
Environment files, key files and generated archives are also ignored.

`benchmarks/data/` contains an allowlisted anonymous export of the latest
pipelined experiment: workload configuration, two rounds of full-dispatch timing
samples, and the recalculated table. Rank numbers are anonymous logical indices.
All timing values are preserved. No GPU experiment was rerun for publication.

Runtime source hashes in `benchmarks/data/f2_measured_runtime_source_sha256.json` identify
the original sources used by the completed experiment before the later cleanup.
They do not assert that the cleaned current source is byte-identical. Publication changes to execution
scripts, report tools and documentation are outside that fingerprint scope.
Machine-specific executed source archives and binaries remain local.

Recalculate the public table on a CPU with:

```bash
python3 benchmarks/report_latest.py
```

To use your own GPU host, pass its SSH alias explicitly to
`scripts/sync_remote.sh` and set `PCIE_EP_CONTAINER` for container execution.
The generic mounts are `/workspace` for this checkout and `/build` for the cache.
