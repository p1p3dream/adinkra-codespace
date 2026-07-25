#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 5 )); then
  echo "usage: $0 <source-name> <source-root> <run-root> <expected-columns> <poll-seconds>" >&2
  exit 64
fi

source_name=$1
source_root=${2%/}
run_root=${3%/}
expected_columns=$4
poll_seconds=$5
ack_root="$run_root/acknowledged/$source_name"
mkdir -p "$ack_root"

while true; do
  ready_count=0
  acknowledged_count=0
  if [[ -d "$source_root/ready" ]]; then
    while IFS= read -r ready; do
      [[ -n "$ready" ]] || continue
      ready_count=$((ready_count + 1))
      column=$(basename "$ready" .ready)
      receipt="$source_root/receipts/${column}.sha256"
      [[ -f "$receipt" ]] || continue
      (
        cd "$source_root"
        sha256sum -c "receipts/${column}.sha256"
      ) >"$ack_root/${column}.verification.log"
      cp "$ready" "$ack_root/${column}.ready"
      sync "$ack_root/${column}.verification.log" "$ack_root/${column}.ready"
      acknowledged_count=$((acknowledged_count + 1))
    done < <(find "$source_root/ready" -maxdepth 1 -type f -name 'column-*.ready' | sort)
  fi
  printf '%s ready=%d acknowledged=%d expected=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$ready_count" "$acknowledged_count" "$expected_columns"
  if (( acknowledged_count == expected_columns )); then
    break
  fi
  sleep "$poll_seconds"
done

touch "$ack_root/COMPLETE"
sync "$ack_root/COMPLETE"
