#!/usr/bin/env bash
# Run on the remote host whose checkout is mounted at /workspace in the container.
# Examples:
#   bash scripts/benchmark_s5_s8.sh sweep
#   PCIE_EP_CHANNELS=8 bash scripts/benchmark_s5_s8.sh main
#   PCIE_EP_RELAY_TUNING=1 bash scripts/benchmark_s5_s8.sh optional
#   bash scripts/benchmark_s5_s8.sh sweep --dry-run
# Existing JSON/CSV outputs are archived before a case is rerun; logs are retained.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-main}
dry_run=${PCIE_EP_DRY_RUN:-0}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )) && [[ $1 == --dry-run ]]; then dry_run=1; shift; fi
if (( $# > 0 )) || [[ ! $mode =~ ^(sweep|main|optional)$ ]] || [[ ! $dry_run =~ ^[01]$ ]]; then
    printf 'Usage: %s {sweep|main|optional} [--dry-run]\n' "$0" >&2
    exit 2
fi

container=${PCIE_EP_CONTAINER:-pcie-ep}
channels=${PCIE_EP_CHANNELS:-8}
pipeline_stages=${PCIE_EP_PIPELINE_STAGES:-2}
relay_tuning=${PCIE_EP_RELAY_TUNING:-0}
audit_nvlink=${PCIE_EP_AUDIT_NVLINK:-1}
if [[ ! $channels =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
    printf 'PCIE_EP_CHANNELS must be in [1, 32].\n' >&2
    exit 2
fi
if [[ ! $pipeline_stages =~ ^[23]$ ]] || [[ ! $relay_tuning =~ ^[01]$ ]] || [[ ! $audit_nvlink =~ ^[01]$ ]]; then
    printf 'Use pipeline stages 2/3 and boolean relay-tuning/NVLink-audit values 0/1.\n' >&2
    exit 2
fi
if [[ $mode == optional && $relay_tuning == 1 ]] && (( channels < 2 )); then
    printf 'The two-worker relay comparison requires PCIE_EP_CHANNELS >= 2.\n' >&2
    exit 2
fi

results_dir="$project_dir/results/s5_s8"
run_label="benchmark_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
log_dir="$results_dir/logs/$run_label"
all_tokens=(16 32 64 128 256 512 1024 2048 4096 8192)
sweep_tokens=(16 1024 8192)
common=(--experts 256 --topk 8 --hidden 7168 --expand --alignment 1
        --pipeline-stages "$pipeline_stages" --seed 20261024)
case_count=0
audit_started=0

finish() {
    local status=$?
    trap - EXIT
    if [[ $audit_started == 1 ]]; then
        if ! bash "$project_dir/tests/capture_path_evidence.sh" "${run_label}_after" "$log_dir" \
            > "$log_dir/nvlink_after.log" 2>&1; then
            status=1
        elif ! python3 "$project_dir/scripts/check_path_counters.py" \
            "$log_dir/nvlink_${run_label}_before.json" "$log_dir/nvlink_${run_label}_after.json" \
            --output "$log_dir/nvlink_counter_delta.json" > "$log_dir/nvlink_delta.log" 2>&1; then
            status=1
        fi
    fi
    printf '%s\n' "$status" > "$log_dir/pipeline.exit"
    printf 'Benchmark suite %s finished with status %s; logs: %s\n' "$mode" "$status" "$log_dir"
    exit "$status"
}

if [[ $dry_run == 0 ]]; then
    mkdir -p "$log_dir"
    trap finish EXIT
    host_manifest="$log_dir/host_source_manifest.json"
    container_manifest="$log_dir/container_source_manifest.json"
    python3 "$project_dir/scripts/source_manifest.py" > "$host_manifest"
    docker exec -w /workspace "$container" python scripts/source_manifest.py > "$container_manifest"
    if ! cmp -s -- "$host_manifest" "$container_manifest"; then
        printf 'Host/container source hashes differ; verify the /workspace mount and synchronize first.\n' >&2
        exit 1
    fi
    if [[ $audit_nvlink == 1 ]]; then
        bash "$project_dir/tests/capture_path_evidence.sh" "${run_label}_before" "$log_dir" \
            > "$log_dir/nvlink_before.log" 2>&1
        audit_started=1
    fi
fi

run_case() {
    local name=$1
    local output=$2
    shift 2
    case_count=$((case_count + 1))
    local command=(docker exec -e PYTHONPATH=/workspace -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1
                   -w /workspace "$container" torchrun --standalone --nproc-per-node=8
                   benchmarks/benchmark.py "${common[@]}" "$@" --output "$output")
    printf '[%02d] %s\n' "$case_count" "$name"
    printf '  '
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ $dry_run == 1 ]]; then return; fi

    # Preserve any complete or interrupted earlier result. benchmark.py itself
    # writes JSON/CSV after every completed T, including partial suite runs.
    local extension
    for extension in json csv; do
        if [[ -f "$project_dir/$output.$extension" ]]; then
            mkdir -p "$log_dir/previous_outputs/$(dirname -- "$output")"
            mv -- "$project_dir/$output.$extension" "$log_dir/previous_outputs/$output.$extension"
        fi
    done
    printf '%q ' "${command[@]}" > "$log_dir/$name.command.sh"
    printf '\n' >> "$log_dir/$name.command.sh"
    local status=0
    if "${command[@]}" > "$log_dir/$name.log" 2>&1; then
        printf '  completed: %s.json\n' "$output"
    else
        status=$?
        printf '  failed (%s): %s\n' "$status" "$log_dir/$name.log" >&2
    fi
    printf '%s\n' "$status" > "$log_dir/$name.exit"
    # Avoid proceeding after a failing protocol or a partial benchmark case.
    return "$status"
}

case "$mode" in
    sweep)
        for count in 1 2 4 8 16 32; do
            for variant in homogeneous warp warp_f2; do
                flags=(--no-warp-specialization --no-fuse-receive-expand)
                if [[ $variant == warp ]]; then flags=(--warp-specialization --no-fuse-receive-expand); fi
                if [[ $variant == warp_f2 ]]; then flags=(--warp-specialization --fuse-receive-expand); fi
                name="s5_c${count}_${variant}_q256"
                run_case "$name" "results/s5_s8/sweep/per_channel/dispatch_$name" \
                    --stage 5 --channels "$count" --no-source-fusion --numa-mode direct --wire-dtype bf16 \
                    --payload-slots 256 --descriptor-slots 256 --replica-slots 256 \
                    "${flags[@]}" --tokens "${sweep_tokens[@]}" --warmup 2 --iterations 8
            done
        done
        # Keep total source payload AND per-endpoint descriptor windows fixed.
        # All counts divide 256, so channel_count * per_channel_slots == 256.
        for count in 1 2 4 8 16 32; do
            slots=$((256 / count))
            name="s5_c${count}_warp_f2_total256"
            run_case "$name" "results/s5_s8/sweep/fixed_total/dispatch_$name" \
                --stage 5 --channels "$count" --warp-specialization --fuse-receive-expand \
                --no-source-fusion --numa-mode direct --wire-dtype bf16 \
                --payload-slots "$slots" --descriptor-slots "$slots" --replica-slots "$slots" \
                --tokens "${sweep_tokens[@]}" --warmup 2 --iterations 8
        done
        ;;
    main)
        for variant in s5 s6 s8_direct s8_relay; do
            stage=5
            mode_flags=(--no-source-fusion --numa-mode direct)
            if [[ $variant == s6 ]]; then stage=6; mode_flags=(--source-fusion --numa-mode direct); fi
            if [[ $variant == s8_direct ]]; then stage=8; mode_flags=(--source-fusion --numa-mode direct); fi
            if [[ $variant == s8_relay ]]; then stage=8; mode_flags=(--source-fusion --numa-mode relay); fi
            run_case "$variant" "results/s5_s8/dispatch_$variant" \
                --stage "$stage" --channels "$channels" --warp-specialization --fuse-receive-expand \
                "${mode_flags[@]}" --wire-dtype bf16 --execution-mode eager --relay-workers 1 \
                --payload-slots 256 --descriptor-slots 256 --replica-slots 256 \
                --tokens "${all_tokens[@]}" --warmup 5 --iterations 20
        done
        ;;
    optional)
        for execution in cached graph graph_cached; do
            run_case "$execution" "results/s5_s8/dispatch_$execution" \
                --stage 6 --channels "$channels" --warp-specialization --fuse-receive-expand \
                --source-fusion --numa-mode direct --wire-dtype bf16 --execution-mode "$execution" \
                --payload-slots 256 --descriptor-slots 256 --replica-slots 256 \
                --tokens "${all_tokens[@]}" --warmup 5 --iterations 20
        done
        run_case s7_fp8 "results/s5_s8/dispatch_s7_fp8" \
            --stage 7 --channels "$channels" --warp-specialization --fuse-receive-expand \
            --source-fusion --numa-mode direct --wire-dtype fp8_e4m3 --execution-mode eager \
            --payload-slots 256 --descriptor-slots 256 --replica-slots 256 \
            --tokens "${all_tokens[@]}" --warmup 5 --iterations 20
        if [[ $relay_tuning == 1 ]]; then
            for workers in 1 2; do
                run_case "s8_relay_workers$workers" "results/s5_s8/relay_tuning/dispatch_s8_relay_workers$workers" \
                    --stage 8 --channels "$channels" --warp-specialization --fuse-receive-expand \
                    --source-fusion --numa-mode relay --wire-dtype bf16 --execution-mode eager \
                    --relay-workers "$workers" --payload-slots 256 --descriptor-slots 256 --replica-slots 256 \
                    --tokens "${sweep_tokens[@]}" --warmup 5 --iterations 20
            done
        fi
        ;;
esac
printf 'Scheduled %s sequential eight-GPU jobs for mode=%s, selected channels=%s.\n' "$case_count" "$mode" "$channels"
