#!/usr/bin/env bash
# Run on the remote host. Wait only for this task's existing stress launcher.
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
container=${PCIE_EP_CONTAINER:-pcie-ep}
stress_pid=${PCIE_EP_STRESS_PID:?Set PCIE_EP_STRESS_PID to the stress launcher PID for this experiment}
if [[ ! $stress_pid =~ ^[1-9][0-9]*$ ]]; then
    printf 'PCIE_EP_STRESS_PID must be a positive process ID.\n' >&2
    exit 2
fi
mkdir -p "$task_dir/results"
trap 'printf "%s\n" "$?" > "$task_dir/results/pipeline.exit"' EXIT
while docker exec "$container" test -d "/proc/$stress_pid"; do sleep 2; done
docker exec -w /workspace "$container" python -c 'import json; d=json.load(open("results/s4_stress.json")); assert len(d["records"])==3 and all(x["status"]=="passed" and x["dispatches_per_rank"]==100000 for x in d["records"])'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. python -m unittest discover -s tests -p test_api.py -v > results/api_tests.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_timeout.py > results/timeout.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_profiler.py > results/profiler.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=8 tests/test_large.py --output results/large.json > results/large.log 2>&1'
bash "$task_dir/tests/capture_path_evidence.sh" s4_before
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=8 benchmarks/benchmark.py --tokens 16 32 64 128 256 512 1024 2048 4096 8192 --experts 256 --topk 8 --hidden 7168 --expand --stage 4 --payload-slots 256 --descriptor-slots 256 --warmup 5 --iterations 20 --output results/dispatch > results/benchmark.log 2>&1'
bash "$task_dir/tests/capture_path_evidence.sh" s4_after
