#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 3 )); then
  echo "usage: $0 <stonkbot-run-root> <local-run-root> <expected-columns>" >&2
  exit 64
fi

stonkbot_run_root=${1%/}
local_run_root=${2%/}
expected_columns=$3

mkdir -p "$local_run_root"
resume_args=(--partial)
if rsync --help 2>&1 | grep -q -- '--append-verify'; then
  resume_args+=(--append-verify)
fi
while true; do
  set +e
  rsync -a \
    "${resume_args[@]}" \
    --exclude 'verified-union/complete/' \
    -e "ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "brandon@192.168.68.71:$stonkbot_run_root/" "$local_run_root/"
  rsync_rc=$?
  set -e
  if (( rsync_rc != 0 && rsync_rc != 24 )); then
    echo "rsync failed with status $rsync_rc" >&2
    exit "$rsync_rc"
  fi
  if (( rsync_rc == 24 )); then
    echo "source files changed during transfer; retrying after mirror count" >&2
  fi

  ready_count=$(find "$local_run_root/acknowledged" -type f -name 'column-*.ready' 2>/dev/null | wc -l | tr -d ' ')
  printf '%s locally mirrored=%d expected=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ready_count" "$expected_columns"
  if (( ready_count == expected_columns )); then
    break
  fi
  sleep 15
done

column_map="$local_run_root/verified-union/provenance/column-source-map.tsv"
union_receipt="$local_run_root/verified-union/provenance/union-files-relative.sha256"
if [[ ! -f "$column_map" || ! -f "$union_receipt" ]]; then
  echo "missing verified-union reconstruction metadata" >&2
  exit 66
fi
mkdir -p "$local_run_root/verified-union/complete"
while IFS=$'\t' read -r column host; do
  [[ -n "$column" && -n "$host" ]] || continue
  source_dir="$local_run_root/hosts/$host/complete/$column"
  destination_dir="$local_run_root/verified-union/complete/$column"
  [[ -d "$source_dir" ]] || {
    echo "missing mirrored source column: $source_dir" >&2
    exit 67
  }
  if [[ ! -e "$destination_dir" ]]; then
    cp -al "$source_dir" "$destination_dir"
  fi
done <"$column_map"
(
  cd "$local_run_root/verified-union"
  shasum -a 256 -c provenance/union-files-relative.sha256
)

touch "$local_run_root/LOCAL_COPY_COMPLETE"
sync "$local_run_root/LOCAL_COPY_COMPLETE"
