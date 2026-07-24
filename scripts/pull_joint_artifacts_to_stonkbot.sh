#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 7 )); then
  echo "usage: $0 <source-name> <ssh-host> <ssh-port> <source-root> <stonkbot-root> <expected-columns> <poll-seconds>" >&2
  exit 64
fi

source_name=$1
ssh_host=$2
ssh_port=$3
source_root=${4%/}
stonkbot_root=${5%/}
expected_columns=$6
poll_seconds=$7
destination="$stonkbot_root/hosts/$source_name"

mkdir -p "$destination" "$stonkbot_root/acknowledged/$source_name"
resume_args=(--partial)
if rsync --help 2>&1 | grep -q -- '--append-verify'; then
  resume_args+=(--append-verify)
fi

while true; do
  set +e
  rsync -a \
    "${resume_args[@]}" \
    --exclude target \
    -e "ssh -p $ssh_port -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4" \
    "$ssh_host:$source_root/" "$destination/"
  rsync_rc=$?
  set -e
  if (( rsync_rc != 0 && rsync_rc != 24 )); then
    echo "rsync failed with status $rsync_rc" >&2
    exit "$rsync_rc"
  fi
  if (( rsync_rc == 24 )); then
    echo "source files changed during transfer; retrying after verification pass" >&2
  fi

  ready_count=0
  acknowledged_count=0
  if [[ -d "$destination/ready" ]]; then
    while IFS= read -r ready; do
      [[ -n "$ready" ]] || continue
      ready_count=$((ready_count + 1))
      column=$(basename "$ready" .ready)
      receipt="$destination/receipts/${column}.sha256"
      [[ -f "$receipt" ]] || continue
      (
        cd "$destination"
        sha256sum -c "receipts/${column}.sha256"
      ) >"$stonkbot_root/acknowledged/$source_name/${column}.verification.log"
      cp "$ready" "$stonkbot_root/acknowledged/$source_name/${column}.ready"
      sync "$stonkbot_root/acknowledged/$source_name/${column}.verification.log" \
        "$stonkbot_root/acknowledged/$source_name/${column}.ready"
      acknowledged_count=$((acknowledged_count + 1))
    done < <(find "$destination/ready" -maxdepth 1 -type f -name 'column-*.ready' | sort)
  fi

  printf '%s ready=%d acknowledged=%d expected=%d\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$ready_count" "$acknowledged_count" "$expected_columns"

  if (( acknowledged_count == expected_columns )); then
    break
  fi
  sleep "$poll_seconds"
done

touch "$stonkbot_root/acknowledged/$source_name/COMPLETE"
sync "$stonkbot_root/acknowledged/$source_name/COMPLETE"
