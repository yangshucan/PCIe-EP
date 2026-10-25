#!/usr/bin/env bash
# Push reviewed source files; measured results remain on each machine.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
remote_host=${1:-${PCIE_EP_REMOTE_HOST:-}}
remote_dir=${2:-${PCIE_EP_REMOTE_DIR:-/opt/pcie-ep}}
if [[ -z $remote_host ]]; then
    printf 'Usage: %s REMOTE_HOST [REMOTE_DIRECTORY]\n' "$0" >&2
    exit 2
fi
if [[ ! $remote_host =~ ^[a-zA-Z0-9_.@-]+$ || $remote_host == -* ]]; then
    printf 'Unsupported remote host syntax.\n' >&2
    exit 2
fi

# Keep the destination a simple absolute path for SSH/rsync shell quoting.
if [[ ! "$remote_dir" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
    printf 'Unsupported remote directory: %s\n' "$remote_dir" >&2
    exit 2
fi

ssh "$remote_host" "mkdir -p -- '$remote_dir'"
rsync -av --checksum \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.so' \
    --exclude='*.o' \
    --exclude='build/' \
    --exclude='results/' \
    --exclude='.DS_Store' \
    --exclude='._*' \
    "$project_dir/" "$remote_host:$remote_dir/"

local_manifest=$(mktemp)
remote_manifest=$(mktemp)
trap 'rm -f -- "$local_manifest" "$remote_manifest"' EXIT
python3 "$project_dir/scripts/source_manifest.py" > "$local_manifest"
ssh "$remote_host" "python3 '$remote_dir/scripts/source_manifest.py'" > "$remote_manifest"
if ! cmp -s -- "$local_manifest" "$remote_manifest"; then
    diff -u -- "$local_manifest" "$remote_manifest" || true
    printf 'Source verification failed. Inspect changed or remote-only files.\n' >&2
    exit 1
fi
printf 'Source hashes match: %s <-> %s:%s\n' "$project_dir" "$remote_host" "$remote_dir"
