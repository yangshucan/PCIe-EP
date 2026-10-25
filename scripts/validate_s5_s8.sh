#!/usr/bin/env bash
# Sequential advanced-path acceptance in the existing eight-GPU container.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=${PCIE_EP_CONTAINER:-pcie-ep}
results_prefix=${PCIE_EP_VALIDATION_RESULTS:-results/s5_s8}
results_dir="$project_dir/$results_prefix"
mkdir -p "$results_dir"
finish() {
    local status=$?
    printf '%s\n' "$status" > "$results_dir/validation.exit"
}
trap finish EXIT

run() {
    local name=$1
    shift
    printf 'Running %s\n' "$name"
    docker exec -e PYTHONPATH=/workspace -w /workspace "$container" "$@" \
        > "$results_dir/$name.log" 2>&1
}

run build python -c 'import pcie_ep._C; print("CUDA extension loaded")'
run unit python -m unittest discover -s tests -p test_api.py -v
run unit_advanced python -m unittest discover -s tests -p 'test_s[67].py' -v
run s4_regression torchrun --standalone --nproc-per-node=8 tests/test_correctness.py \
    --stage 4 --quick --output "$results_prefix/s4_regression.json"
run s5_homogeneous torchrun --standalone --nproc-per-node=8 tests/test_advanced.py \
    --stage 5 --channels 8 --quick --output "$results_prefix/s5_homogeneous.json"
run s5_channels32 torchrun --standalone --nproc-per-node=8 tests/test_advanced.py \
    --stage 5 --channels 32 --warp-specialization --fuse-receive-expand --quick \
    --output "$results_prefix/s5_channels32.json"

for stage in 5 6 7 8; do
    options=(--stage "$stage" --channels 4 --warp-specialization --fuse-receive-expand)
    if (( stage >= 6 )); then options+=(--source-fusion); fi
    if (( stage == 8 )); then options+=(--numa-mode relay --replica-slots 2); fi
    run "s${stage}_correctness" torchrun --standalone --nproc-per-node=8 \
        tests/test_advanced.py "${options[@]}" \
        --output "$results_prefix/s${stage}_correctness.json"
done

run s6_cache_graph torchrun --standalone --nproc-per-node=8 tests/test_s6.py \
    --distributed --output "$results_prefix/s6_cache_graph.json"
run s7_fp8 torchrun --standalone --nproc-per-node=8 tests/test_s7.py \
    --distributed --output "$results_prefix/s7_fp8.json"
run s8_cache_graph torchrun --standalone --nproc-per-node=8 tests/test_s6.py \
    --distributed --stage 8 --numa-mode relay --output "$results_prefix/s8_cache_graph.json"
run s8_fp8 torchrun --standalone --nproc-per-node=8 tests/test_s7.py \
    --distributed --stage 8 --numa-mode relay --output "$results_prefix/s8_fp8.json"
run relay torchrun --standalone --nproc-per-node=8 tests/test_relay.py \
    --require-placement --output "$results_prefix/relay.json"
run relay_workers2 torchrun --standalone --nproc-per-node=8 tests/test_relay.py \
    --workers 2 --require-placement --output "$results_prefix/relay_workers2.json"
for stage in 5 8; do
    relay_args=()
    if (( stage == 8 )); then relay_args+=(--numa-mode relay); fi
    run "s${stage}_protocol_stages" torchrun --standalone --nproc-per-node=8 \
        tests/test_protocol_stages.py --stage "$stage" "${relay_args[@]}" \
        --output "$results_prefix/s${stage}_protocol_stages.json"
done

run lifetime torchrun --standalone --nproc-per-node=2 tests/test_consumer_lifetime.py \
    --stage 5 --channels 4 --warp-specialization --fuse-receive-expand \
    --output "$results_prefix/consumer_lifetime.json"
run integration torchrun --standalone --nproc-per-node=2 tests/test_integration.py \
    --stage 6 --channels 4 --warp-specialization --source-fusion --nccl-reuse \
    --output "$results_prefix/integration.json"
run timeout torchrun --standalone --nproc-per-node=2 tests/test_timeout.py \
    --stage 5 --channels 4 --warp-specialization \
    --output "$results_prefix/timeout.json"

for stage in 5 8; do
    options=(--stage "$stage" --channels 4 --warp-specialization --fuse-receive-expand)
    if (( stage == 8 )); then options+=(--numa-mode relay --replica-slots 2); fi
    run "s${stage}_stress" torchrun --standalone --nproc-per-node=8 \
        tests/test_stress.py "${options[@]}" --iterations "${PCIE_EP_STRESS_ITERATIONS:-100000}" \
        --output "$results_prefix/s${stage}_stress.json"
    run "s${stage}_large" torchrun --standalone --nproc-per-node=8 \
        tests/test_large.py "${options[@]}" --output "$results_prefix/s${stage}_large.json"
    for mode in unique expanded; do
        mode_args=()
        if [[ $mode == expanded ]]; then mode_args+=(--expand); fi
        run "s${stage}_profiler_${mode}" torchrun --standalone --nproc-per-node=8 \
            tests/test_profiler.py "${options[@]}" "${mode_args[@]}" \
            --output-dir "$results_prefix/s${stage}_profiler_${mode}"
    done
done

for stage in 6 8; do
    relay_args=()
    if (( stage == 8 )); then relay_args+=(--numa-mode relay); fi
    run "s${stage}_profiler_graph" torchrun --standalone --nproc-per-node=8 \
        tests/test_profiler.py --stage "$stage" --channels 4 --warp-specialization \
        --source-fusion --fuse-receive-expand --execution-mode graph --expand "${relay_args[@]}" \
        --output-dir "$results_prefix/s${stage}_profiler_graph"
done

python3 "$project_dir/scripts/source_manifest.py" > "$results_dir/validated_source_manifest.json"
printf 'Advanced acceptance completed\n'
