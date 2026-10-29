#!/usr/bin/env bash
# Run inside an eight-GPU environment with dependencies already installed.
# Refuse occupied devices; never stop or change another GPU workload.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
result_dir=results/nccl_shm
topk=8
routing_fixture=
if [[ $# -gt 0 && $1 != --* ]]; then
    result_dir=$1
    shift
fi
while [[ $# -gt 0 ]]; do
    case $1 in
        --topk)
            [[ $# -ge 2 ]] || { echo '--topk requires a value.' >&2; exit 2; }
            topk=$2
            shift 2
            ;;
        --topk4)
            topk=4
            shift
            ;;
        --routing-fixture)
            [[ $# -ge 2 ]] || { echo '--routing-fixture requires a JSON manifest.' >&2; exit 2; }
            routing_fixture=$2
            shift 2
            ;;
        --help)
            echo 'Usage: benchmark_nccl_shm.sh [RESULT_DIR] [--topk 4|8] [--routing-fixture MANIFEST.json]'
            echo 'Top-k 4 requires an offline routing fixture; the default top-k 8 remains compatible.'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
[[ $topk == 4 || $topk == 8 ]] || { echo '--topk must be 4 or 8.' >&2; exit 2; }
if [[ $topk == 4 && -z $routing_fixture ]]; then
    echo 'Top-k 4 requires --routing-fixture; create the fixed routes before benchmarking.' >&2
    exit 2
fi
mkdir -p "$result_dir"
result_dir=$(CDPATH= cd -- "$result_dir" && pwd)
benchmark_args=(--topk "$topk")
report_fixture="$root/benchmarks/data/nccl_shm_routes.json"
if [[ -n $routing_fixture ]]; then
    routing_fixture=$(python3 - "$routing_fixture" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=True))
PY
)
    benchmark_args+=(--routing-fixture "$routing_fixture")
    report_fixture=$routing_fixture
fi
python3 - "$result_dir" "$topk" "$routing_fixture" "$report_fixture" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result_dir, topk, routing_fixture, report_fixture = sys.argv[1:]
record = dict(schema_version=1, topk=int(topk),
              routing_selection='offline_fixture' if routing_fixture else 'legacy_seeded',
              routing_fixture_path=routing_fixture or None,
              report_route_fixture_path=report_fixture,
              route_fixture_sha256=hashlib.sha256(Path(report_fixture).read_bytes()).hexdigest())
if routing_fixture:
    fixture = json.loads(Path(routing_fixture).read_text())
    data_file = fixture.get('data_file')
    if not isinstance(data_file, str) or Path(data_file).name != data_file:
        raise SystemExit('Routing fixture data_file must be a filename in the manifest directory.')
    tensors = Path(routing_fixture).parent / data_file
    tensors_sha256 = hashlib.sha256(tensors.read_bytes()).hexdigest()
    if tensors_sha256 != fixture.get('tensors_sha256'):
        raise SystemExit('Routing fixture tensor hash does not match its manifest.')
    record.update(route_tensors_path=str(tensors), route_tensors_sha256=tensors_sha256)
Path(result_dir, 'run_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
PY
python3 scripts/source_manifest.py --root "$root" --output "$result_dir/executed_source_manifest.json"

# Keep the original manifest path: its tensor filenames can be relative to it.
# Both rounds and the report use that same file, whose identity is checked here.
check_routing_fixture() {
    python3 - "$result_dir/run_manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text())
actual = hashlib.sha256(Path(record['report_route_fixture_path']).read_bytes()).hexdigest()
if actual != record['route_fixture_sha256']:
    raise SystemExit('Routing fixture changed during the comparison; discard this run.')
if 'route_tensors_path' in record:
    actual = hashlib.sha256(Path(record['route_tensors_path']).read_bytes()).hexdigest()
    if actual != record['route_tensors_sha256']:
        raise SystemExit('Routing fixture tensors changed during the comparison; discard this run.')
PY
}

python3 - "$result_dir" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

def output(args):
    return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=30)

processes = output(['--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'])
gpus = output(['--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'])
record = {'compute_processes':processes, 'gpu_utilization_and_memory':gpus}
Path(sys.argv[1], 'idle_preflight.json').write_text(json.dumps(record, indent=2)+'\n')
if processes.strip():
    raise SystemExit('GPU compute processes are present; obtain an idle window before benchmarking.')
rows = [list(map(int, row.split(','))) for row in gpus.strip().splitlines()]
if len(rows) != 8 or any(util > 5 or memory > 1024 for _, util, memory in rows):
    raise SystemExit('All eight GPUs must be idle, with no occupied device memory.')
PY

export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PCIE_EP_PROFILE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET,NVLS,ENV,TUNING
bash tests/capture_path_evidence.sh nccl_shm_before "$result_dir"

for round in 1 2; do
    check_routing_fixture
    mkdir -p "$result_dir/round$round-logs"
    export NCCL_DEBUG_FILE="$result_dir/round$round-logs/nccl.%p.log"
    variants=(pcie_ep nccl_payload nccl_dispatch)
    if [[ $round == 2 ]]; then
        variants=(nccl_dispatch nccl_payload pcie_ep)
    fi
    torchrun --standalone --nproc-per-node=8 benchmarks/benchmark_nccl_shm.py \
        "${benchmark_args[@]}" \
        --warmup 5 --iterations 20 --variants "${variants[@]}" \
        --output "$result_dir/round$round" > "$result_dir/round$round.log" 2>&1
    check_routing_fixture
done

bash tests/capture_path_evidence.sh nccl_shm_after "$result_dir"
python3 scripts/check_path_counters.py "$result_dir/nvlink_nccl_shm_before.json" \
    "$result_dir/nvlink_nccl_shm_after.json" --output "$result_dir/nvlink_delta.json"
python3 scripts/check_nccl_shm_logs.py "$result_dir"/round*-logs/nccl.*.log \
    --output "$result_dir/shm_path.json"
check_routing_fixture
python3 benchmarks/report_nccl_shm.py "$result_dir/round1.json" "$result_dir/round2.json" \
    --route-fixture "$report_fixture" \
    --output-dir "$result_dir/summary"
