#!/usr/bin/env bash
# Run the refactor validation sequentially in the existing experiment container.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=${PCIE_EP_CONTAINER:-pcie-ep}
results_dir="$project_dir/results/refactor"
mkdir -p "$results_dir"
host_manifest=$(mktemp)
container_manifest=$(mktemp)
finish() {
    local status=$?
    rm -f -- "$host_manifest" "$container_manifest"
    printf '%s\n' "$status" > "$results_dir/pipeline.exit"
}
trap finish EXIT

run() {
    local log_name=$1
    shift
    printf 'Running %s\n' "$log_name"
    docker exec -e PYTHONPATH=/workspace -w /workspace "$container" "$@" \
        > "$results_dir/$log_name" 2>&1
}

# The host's directory must be the code actually mounted at /workspace. This
# prevents validating another checkout when using a different container/path.
python3 "$project_dir/scripts/source_manifest.py" > "$host_manifest"
docker exec -w /workspace "$container" python scripts/source_manifest.py > "$container_manifest"
if ! cmp -s -- "$host_manifest" "$container_manifest"; then
    printf 'Host and container source differ; verify the /workspace mount.\n' >&2
    exit 1
fi
cp -- "$host_manifest" "$results_dir/executed_source_manifest.json"
run build.log python -c 'import pcie_ep._C; print("CUDA extension loaded")'
run unit.log python -m unittest discover -s tests -p test_api.py -v

run s0_build.log /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_90 \
    -Xcompiler -pthread tests/mapped_litmus.cu -o /build/mapped_litmus
run s0_same_numa.json /build/mapped_litmus --iterations 10000
run s0_cross_numa.json /build/mapped_litmus --iterations 10000 --consumer 4

run s1.log torchrun --standalone --nproc_per_node=2 tests/test_correctness.py \
    --stage 1 --quick --output results/refactor/s1_correctness.json
run s2.log torchrun --standalone --nproc_per_node=8 tests/test_correctness.py \
    --stage 2 --output results/refactor/s2_correctness.json
run s3.log torchrun --standalone --nproc_per_node=8 tests/test_correctness.py \
    --stage 3 --quick --output results/refactor/s3_correctness.json
run s4.log torchrun --standalone --nproc_per_node=8 tests/test_correctness.py \
    --stage 4 --output results/refactor/s4_correctness.json

run integration.log torchrun --standalone --nproc_per_node=2 tests/test_integration.py \
    --nccl-reuse --output results/refactor/integration.json
run consumer_lifetime.log torchrun --standalone --nproc_per_node=2 tests/test_consumer_lifetime.py \
    --output results/refactor/consumer_lifetime.json
run timeout.log torchrun --standalone --nproc_per_node=2 tests/test_timeout.py \
    --output results/refactor/timeout.json
run profiler_unique.log torchrun --standalone --nproc_per_node=2 tests/test_profiler.py \
    --output-dir results/refactor/profiler_unique
run profiler_expanded.log torchrun --standalone --nproc_per_node=2 tests/test_profiler.py \
    --expand --output-dir results/refactor/profiler_expanded

run stress.log torchrun --standalone --nproc_per_node=8 tests/test_stress.py \
    --iterations 10000 --output results/refactor/stress.json
run large.log torchrun --standalone --nproc_per_node=8 tests/test_large.py \
    --output results/refactor/large.json

bash "$project_dir/tests/capture_path_evidence.sh" refactor_before
run benchmark.log torchrun --standalone --nproc_per_node=8 benchmarks/benchmark.py \
    --tokens 16 32 64 128 256 512 1024 2048 4096 8192 \
    --experts 256 --topk 8 --hidden 7168 --expand \
    --stage 4 --payload-slots 256 --descriptor-slots 256 \
    --warmup 5 --iterations 20 --output results/refactor/dispatch
bash "$project_dir/tests/capture_path_evidence.sh" refactor_after
python3 "$project_dir/scripts/check_path_counters.py" \
    "$project_dir/results/nvlink_refactor_before.json" \
    "$project_dir/results/nvlink_refactor_after.json" \
    --output "$results_dir/nvlink_counter_delta.json"
printf 'Validation completed\n'
