#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 4 )); then
  echo "usage: $0 <host-prefix> <binary> <artifact-root> <shard-plan>" >&2
  exit 64
fi

host_prefix=$1
binary=$2
artifact_root=$3
shard_plan=$4
script_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$artifact_root/supervisors"
launched=0
while IFS=$'\t' read -r worker_id _host_role ordinals; do
  [[ -n "$worker_id" ]] || continue
  [[ "$worker_id" == \#* ]] && continue
  [[ "$worker_id" == "$host_prefix"-* ]] || continue

  pid_file="$artifact_root/supervisors/${worker_id}.pid"
  supervisor_log="$artifact_root/supervisors/${worker_id}.log"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$worker_id already running with PID $(cat "$pid_file")"
    continue
  fi

  IFS=',' read -r -a ordinal_array <<<"$ordinals"
  nohup "$script_dir/run_joint_column_worker.sh" \
    "$binary" "$artifact_root" "$worker_id" "${ordinal_array[@]}" \
    >"$supervisor_log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  echo "launched $worker_id pid=$pid ordinals=$ordinals"
  launched=$((launched + 1))
done <"$shard_plan"

if (( launched == 0 )); then
  echo "no workers launched for host prefix $host_prefix" >&2
  exit 1
fi
