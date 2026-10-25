#!/usr/bin/env bash
# Counterbalanced, sequential eight-GPU F2 benchmark; run on the remote host.
# bash scripts/benchmark_f2.sh [all|round1|round2] [--dry-run]
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
selection=${1:-all}
if (( $# )); then shift; fi
dry_run=0
if (( $# )) && [[ $1 == --dry-run ]]; then dry_run=1; shift; fi
if (( $# )) || [[ ! $selection =~ ^(all|round1|round2)$ ]]; then
    printf 'Usage: %s [all|round1|round2] [--dry-run]\n' "$0" >&2
    exit 2
fi
container=${PCIE_EP_CONTAINER:-pcie-ep}
run_id=${PCIE_EP_F2_RUN_ID:-f2_$(date -u +%Y%m%dT%H%M%SZ)_$$}
if [[ ! $run_id =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'PCIE_EP_F2_RUN_ID may contain only letters, digits, underscore and hyphen.\n' >&2
    exit 2
fi
prefix="results/f2/runs/$run_id"
run_dir="$project_dir/$prefix"
active_round=""
audit_started=0
jobs=0

complete_round() {
    local status=$1
    local directory="$run_dir/$active_round"
    if [[ $audit_started == 1 ]]; then
        if ! bash "$project_dir/tests/capture_path_evidence.sh" "${active_round}_after" "$directory" > "$directory/nvlink_after.log" 2>&1; then
            status=1
        elif ! python3 "$project_dir/scripts/check_path_counters.py" \
            "$directory/nvlink_${active_round}_before.json" "$directory/nvlink_${active_round}_after.json" \
            --output "$directory/nvlink_counter_delta.json" > "$directory/nvlink_delta.log" 2>&1; then
            status=1
        fi
    fi
    printf '%s\n' "$status" > "$directory/round.exit"
    active_round=""
    audit_started=0
    return "$status"
}
finish() {
    local status=$?
    trap - EXIT
    set +e
    if [[ -n $active_round ]]; then
        complete_round "$status"
        local audit_status=$?
        if (( status == 0 )); then status=$audit_status; fi
    fi
    printf '%s\n' "$status" > "$run_dir/benchmark.exit"
    printf 'F2 benchmark status=%s; preserved results: %s\n' "$status" "$run_dir"
    exit "$status"
}

if [[ $dry_run == 0 ]]; then
    mkdir -p "$run_dir"
    trap finish EXIT
    python3 "$project_dir/scripts/source_manifest.py" > "$run_dir/host_source_manifest.json"
    docker exec -w /workspace "$container" python scripts/source_manifest.py > "$run_dir/container_source_manifest.json"
    if ! cmp -s "$run_dir/host_source_manifest.json" "$run_dir/container_source_manifest.json"; then
        printf 'Source mismatch: /workspace must mount this synchronized checkout.\n' >&2
        exit 1
    fi
fi

run_case() {
    local variant=$1
    local round=$2
    local mode=scalar
    local fusion=--fuse-receive-expand
    case "$variant" in
        scalar_f2) ;;
        nof2) fusion=--no-fuse-receive-expand ;;
        vectorized) mode=vectorized ;;
        pipelined) mode=pipelined ;;
    esac
    local output="$prefix/$round/dispatch_$variant"
    local command=(docker exec -e PYTHONPATH=/workspace -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1
        -w /workspace "$container" torchrun --standalone --nproc-per-node=8 benchmarks/benchmark.py
        --stage 6 --channels 32 --payload-slots 256 --descriptor-slots 256 --replica-slots 256
        --pipeline-stages 2 --warp-specialization --source-fusion "$fusion" --receive-expand-mode "$mode"
        --numa-mode direct --wire-dtype bf16 --execution-mode eager
        --experts 256 --topk 8 --hidden 7168 --expand --alignment 1 --seed 20261024
        --tokens 16 32 64 128 256 512 1024 2048 4096 8192 --warmup 5 --iterations 20
        --output "$output")
    jobs=$((jobs + 1))
    printf '[%d] %s / %s\n  ' "$jobs" "$round" "$variant"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ $dry_run == 1 ]]; then return; fi
    if [[ -e "$project_dir/$output.json" || -e "$project_dir/$output.csv" ]]; then
        printf 'Refusing to overwrite preserved measurements: %s. Use a new PCIE_EP_F2_RUN_ID.\n' "$output" >&2
        return 1
    fi
    printf '%q ' "${command[@]}" > "$run_dir/$round/$variant.command.sh"
    printf '\n' >> "$run_dir/$round/$variant.command.sh"
    local status=0
    "${command[@]}" > "$run_dir/$round/$variant.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$run_dir/$round/$variant.exit"
    return "$status"
}

for round in round1 round2; do
    if [[ $selection != all && $selection != "$round" ]]; then continue; fi
    order=(scalar_f2 nof2 vectorized pipelined)
    if [[ $round == round2 ]]; then order=(pipelined vectorized nof2 scalar_f2); fi
    printf 'Suite %s: %s\n' "$round" "${order[*]}"
    if [[ $dry_run == 0 ]]; then
        if [[ -e "$run_dir/$round" ]]; then
            printf 'Round directory already exists; preserved results must not be overwritten: %s\n' "$run_dir/$round" >&2
            exit 1
        fi
        mkdir -p "$run_dir/$round"
        active_round=$round
        cp "$run_dir/host_source_manifest.json" "$run_dir/$round/executed_source_manifest.json"
        printf '%s\n' "${order[*]}" > "$run_dir/$round/execution_order.txt"
        bash "$project_dir/tests/capture_path_evidence.sh" "${round}_before" "$run_dir/$round" \
            > "$run_dir/$round/nvlink_before.log" 2>&1
        audit_started=1
    fi
    for variant in "${order[@]}"; do run_case "$variant" "$round"; done
    if [[ $dry_run == 0 ]]; then complete_round 0; fi
done
if [[ $dry_run == 0 ]]; then
    python3 "$project_dir/scripts/source_manifest.py" > "$run_dir/final_source_manifest.json"
    cmp -s "$run_dir/host_source_manifest.json" "$run_dir/final_source_manifest.json"
fi
printf 'Scheduled %d sequential jobs; all raw JSON/CSV and per-round audits are retained under %s.\n' "$jobs" "$prefix"
