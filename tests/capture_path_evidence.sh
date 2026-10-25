#!/usr/bin/env bash
# Read-only GPU telemetry; never resets counters or changes physical links.
# Run immediately before and after a benchmark, using distinct labels.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
label=${1:?usage: capture_path_evidence.sh LABEL [RESULT_DIRECTORY]}
result_dir=${2:-"$script_dir/../results"}
python3 - "$label" "$result_dir" <<'PY'
import datetime
import json
import pathlib
import re
import socket
import subprocess
import sys
import time

label, output = sys.argv[1:]
if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
    raise SystemExit('LABEL may contain only letters, numbers, underscore, and hyphen')
output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
record = {
    'label': label,
    'host': socket.gethostname(),
    'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'gpu_workload_launched': False,
    'counter_reset_or_link_change': False,
    'commands': {},
}
commands = {
    'data': ['nvidia-smi', 'nvlink', '-gt', 'd'],
    'pcie': ['nvidia-smi', '--query-gpu=index,pci.bus_id,name,pcie.link.gen.current,pcie.link.width.current', '--format=csv'],
    'processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'],
}
for name, args in commands.items():
    start = time.time_ns()
    try:
        completed = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=30)
        result = {'argv': args, 'returncode': completed.returncode,
                  'output': completed.stdout}
    except (OSError, subprocess.SubprocessError) as error:
        result = {'argv': args, 'returncode': None, 'output': str(error)}
    result['started_unix_ns'] = start
    result['finished_unix_ns'] = time.time_ns()
    record['commands'][name] = result
    (output / f'nvlink_{label}_{name}.txt').write_text(result['output'])

gpu = None
counters = []
for line in record['commands']['data']['output'].splitlines():
    match = re.match(r'GPU (\d+):', line)
    if match:
        gpu = int(match.group(1))
    match = re.search(r'Link (\d+): Data (Tx|Rx): (\d+) KiB', line)
    if match:
        counters.append({'gpu': gpu, 'link': int(match.group(1)),
                         'direction': match.group(2), 'KiB': int(match.group(3))})
record['data_counters'] = counters
record['all_8x18x2_counters_present'] = len(counters) == 8 * 18 * 2
target = output / f'nvlink_{label}.json'
target.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'path': str(target),
                  'counter_values': len(counters),
                  'all_counters_present': record['all_8x18x2_counters_present']}))
if record['commands']['data']['returncode'] != 0 or not counters:
    raise SystemExit(1)
PY
