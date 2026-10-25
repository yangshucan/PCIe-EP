#!/usr/bin/env bash
# Sequential acceptance groups. This runner uses the existing extension; no build.
# bash scripts/validate_f2.sh [smoke|buffers|correctness|delays|overlap|combinations|stress|large|profiler|all] [--dry-run]
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
group=${1:-smoke}
if (( $# )); then shift; fi
dry_run=0
if (( $# )) && [[ $1 == --dry-run ]]; then dry_run=1; shift; fi
if (( $# )) || [[ ! $group =~ ^(smoke|buffers|correctness|delays|overlap|combinations|stress|large|profiler|all)$ ]]; then
    printf 'Unknown group or arguments. See the usage comment in %s.\n' "$0" >&2
    exit 2
fi
container=${PCIE_EP_CONTAINER:-pcie-ep}
run_id=${PCIE_EP_F2_VALIDATION_ID:-validation_$(date -u +%Y%m%dT%H%M%SZ)_$$}
stress_iterations=${PCIE_EP_F2_STRESS_ITERATIONS:-100000}
if [[ ! $run_id =~ ^[A-Za-z0-9_-]+$ || ! $stress_iterations =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid validation ID or stress iteration count.\n' >&2
    exit 2
fi
prefix="results/f2/validation/$run_id/$group"
output_dir="$project_dir/$prefix"
audit_started=0
jobs=0
finish() {
    local status=$?
    trap - EXIT
    set +e
    if [[ $audit_started == 1 ]]; then
        if ! bash "$project_dir/tests/capture_path_evidence.sh" validation_after "$output_dir" > "$output_dir/nvlink_after.log" 2>&1; then
            status=1
        elif ! python3 "$project_dir/scripts/check_path_counters.py" \
            "$output_dir/nvlink_validation_before.json" "$output_dir/nvlink_validation_after.json" \
            --output "$output_dir/nvlink_counter_delta.json" > "$output_dir/nvlink_delta.log" 2>&1; then
            status=1
        fi
    fi
    python3 "$project_dir/scripts/source_manifest.py" > "$output_dir/final_source_manifest.json"
    if ! cmp -s "$output_dir/executed_source_manifest.json" "$output_dir/final_source_manifest.json"; then status=1; fi
    printf '%s\n' "$status" > "$output_dir/validation.exit"
    printf 'F2 validation %s: status=%s; %s\n' "$group" "$status" "$output_dir"
    exit "$status"
}
if [[ $dry_run == 0 ]]; then
    if [[ -e $output_dir ]]; then
        printf 'Refusing to overwrite validation evidence: %s\n' "$output_dir" >&2
        exit 1
    fi
    mkdir -p "$output_dir"
    trap finish EXIT
    python3 "$project_dir/scripts/source_manifest.py" > "$output_dir/executed_source_manifest.json"
    docker exec -w /workspace "$container" python scripts/source_manifest.py > "$output_dir/container_source_manifest.json"
    cmp -s "$output_dir/executed_source_manifest.json" "$output_dir/container_source_manifest.json"
    bash "$project_dir/tests/capture_path_evidence.sh" validation_before "$output_dir" > "$output_dir/nvlink_before.log" 2>&1
    audit_started=1
fi
run() {
    local label=$1
    shift
    local command=(docker exec -e PYTHONPATH=/workspace -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1
        -w /workspace "$container" torchrun --standalone --nproc-per-node=8 "$@")
    jobs=$((jobs + 1))
    printf '[%d] %s\n  ' "$jobs" "$label"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ $dry_run == 1 ]]; then return; fi
    printf '%q ' "${command[@]}" > "$output_dir/$label.command.sh"
    printf '\n' >> "$output_dir/$label.command.sh"
    local status=0
    "${command[@]}" > "$output_dir/$label.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$output_dir/$label.exit"
    return "$status"
}
selected() { [[ $group == all || $group == "$1" ]]; }
if [[ $group == smoke ]]; then
    run smoke tests/test_f2.py --suite correctness --quick --output "$prefix/smoke.json"
fi
if selected buffers; then
    run buffers tests/test_f2_buffers.py --output "$prefix/buffers.json"
fi
for suite in correctness delays overlap combinations; do
    if selected "$suite"; then
        run "$suite" tests/test_f2.py --suite "$suite" --output "$prefix/$suite.json"
    fi
done
for mode in vectorized pipelined; do
    common=(--stage 6 --pipeline-stages 2 --warp-specialization
        --fuse-receive-expand --receive-expand-mode "$mode" --source-fusion --numa-mode direct)
    if selected stress; then
        run "stress_$mode" tests/test_stress.py "${common[@]}" --channels 2 --iterations "$stress_iterations" \
            --payload-slots 1 2 3 --descriptor-slots 2 1 2 --hidden 7168 \
            --output "$prefix/stress_$mode.json"
    fi
    if selected large; then
        run "large_$mode" tests/test_large.py "${common[@]}" --channels 32 \
            --payload-slots 256 --descriptor-slots 256 --tokens 8192 --hidden 7168 --alignment 16 \
            --mode both --output "$prefix/large_$mode.json"
    fi
    if selected profiler; then
        for execution in eager graph graph_cached; do
            run "profiler_${mode}_$execution" tests/test_profiler.py "${common[@]}" --channels 2 \
                --expand --hidden 7168 --execution-mode "$execution" \
                --output-dir "$prefix/profiler_${mode}_$execution"
        done
    fi
done
printf 'Scheduled %d sequential validation jobs; no build step.\n' "$jobs"
