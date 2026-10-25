#!/usr/bin/env bash
# Final sequential verification and benchmark after S4 lifecycle correction.
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
container=${PCIE_EP_CONTAINER:-pcie-ep}
mkdir -p "$task_dir/results"
trap 'printf "%s\n" "$?" > "$task_dir/results/pipeline_final.exit"' EXIT
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_consumer_lifetime.py > results/consumer_lifetime.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_integration.py --nccl-reuse > results/s4_integration.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_timeout.py > results/timeout.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_profiler.py --expand --output-dir results/profiler_expanded > results/profiler_expanded.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=8 tests/test_correctness.py --stage 4 --output results/s4_correctness.json > results/s4.log 2>&1'
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=8 tests/test_large.py --output results/large.json > results/large.log 2>&1'
bash "$task_dir/tests/capture_path_evidence.sh" s4_before
docker exec -w /workspace "$container" bash -lc 'PYTHONPATH=. torchrun --standalone --nproc_per_node=8 benchmarks/benchmark.py --tokens 16 32 64 128 256 512 1024 2048 4096 8192 --experts 256 --topk 8 --hidden 7168 --expand --stage 4 --payload-slots 256 --descriptor-slots 256 --warmup 5 --iterations 20 --output results/dispatch > results/benchmark.log 2>&1'
bash "$task_dir/tests/capture_path_evidence.sh" s4_after
