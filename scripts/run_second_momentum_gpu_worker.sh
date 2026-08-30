#!/usr/bin/env bash
# Supervise a portable second-momentum GPU job list and reconcile abrupt exits.
set -uo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_second_momentum_gpu_worker.sh <job-list> [output-dir] [device] [cpu-parity-terms] [binary]

Examples:
  run_second_momentum_gpu_worker.sh all@0
  run_second_momentum_gpu_worker.sh 20001@0 /data/second-momentum 0
  run_second_momentum_gpu_worker.sh 30001-g1-p0,30001-g3-p0 /data/second-momentum 1

The worker runs in the foreground. Finished jobs are validated and adopted,
live duplicate ownership is rejected by flock, and grouped jobs resume from
their last durable word checkpoint.
EOF
  exit 2
}

[[ $# -ge 1 && $# -le 5 ]] || usage

job_list=$1
output_dir=${2:-results/second_momentum_gpu_fx}
device=${3:-0}
cpu_parity_terms=${4:-128}
binary=${5:-target/release/adinkra-codespace}

[[ -x "$binary" ]] || {
  echo "worker binary is not executable: $binary" >&2
  echo "build it with: cargo build --release --features cuda --bin adinkra-codespace" >&2
  exit 2
}

mkdir -p "$output_dir/supervisor-logs"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
safe_jobs=$(printf '%s' "$job_list" | tr -c 'A-Za-z0-9._@,-' '_')
log_path="$output_dir/supervisor-logs/${stamp}-${HOSTNAME:-unknown}-${safe_jobs}.log"

child_pid=''
forward_signal() {
  local signal=$1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -s "$signal" "$child_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal HUP' HUP
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

echo "job_list=$job_list output_dir=$output_dir device=$device parity=$cpu_parity_terms" | tee -a "$log_path"
"$binary" adynkra-11d-second-momentum-gpu-fx-worker \
  "$job_list" "$output_dir" "$device" "$cpu_parity_terms" \
  > >(tee -a "$log_path") 2>&1 &
child_pid=$!

wait "$child_pid"
status=$?

if (( status > 128 )); then
  observation="signal:$((status - 128))"
else
  observation="exit:$status"
fi

# The child normally writes its own terminal snapshot. This only repairs a
# still-running snapshot after SIGKILL, kernel OOM, or another abrupt exit.
while IFS= read -r -d '' status_file; do
  recorded_pid=$(sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$status_file" | head -1)
  recorded_state=$(sed -n 's/.*"state"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$status_file" | head -1)
  if [[ "$recorded_pid" == "$child_pid" && "$recorded_state" == "running" ]]; then
    "$binary" adynkra-11d-second-momentum-gpu-status-reconcile \
      "$status_file" "$child_pid" "$observation" >>"$log_path" 2>&1 || true
  fi
done < <(find "$output_dir/jobs" -type f -name status.json -print0 2>/dev/null)

"$binary" adynkra-11d-second-momentum-gpu-fx-status \
  "$job_list" "$output_dir" | tee -a "$log_path" || true

echo "worker_exit=$status log=$log_path"
exit "$status"
