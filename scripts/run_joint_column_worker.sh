#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 4 )); then
  echo "usage: $0 <binary> <artifact-root> <worker-id> <ordinal> [ordinal ...]" >&2
  exit 64
fi

binary=$1
artifact_root=$2
worker_id=$3
shift 3
ordinals=("$@")

if [[ ! -x "$binary" ]]; then
  echo "binary is not executable: $binary" >&2
  exit 66
fi

sha_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

source_revision=${ADINKRA_SOURCE_REVISION:-}
if [[ -z "$source_revision" ]] && command -v git >/dev/null 2>&1; then
  source_revision=$(git rev-parse HEAD 2>/dev/null || true)
fi
source_revision=${source_revision:-unrecorded}
executable_sha256=$(sha_file "$binary")
host=$(hostname)
if /usr/bin/time -v true >/dev/null 2>&1; then
  time_args=(-v)
else
  time_args=(-l)
fi

mkdir -p \
  "$artifact_root/logs/complete" \
  "$artifact_root/logs/incomplete" \
  "$artifact_root/receipts" \
  "$artifact_root/ready"

worker_failed=0
for ordinal in "${ordinals[@]}"; do
  if ! [[ "$ordinal" =~ ^[0-9]+$ ]] || (( ordinal < 0 || ordinal >= 56 )); then
    echo "invalid joint column ordinal: $ordinal" >&2
    worker_failed=1
    continue
  fi

  column=$(printf 'column-%03d' "$ordinal")
  attempt="$artifact_root/logs/incomplete/${column}-${worker_id}-$$"
  completed_log="$artifact_root/logs/complete/$column"
  mkdir -p "$attempt"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  cat >"$attempt/execution.json" <<EOF
{
  "schema_version": "adynkra-11d-joint-column-execution-v1",
  "column_ordinal": $ordinal,
  "worker_id": "$worker_id",
  "host": "$host",
  "process_id": $$,
  "source_revision": "$source_revision",
  "executable_sha256": "$executable_sha256",
  "started_utc": "$started",
  "status": "running"
}
EOF
  sync "$attempt/execution.json"

  set +e
  /usr/bin/time "${time_args[@]}" env \
    ADINKRA_SOURCE_REVISION="$source_revision" \
    ADINKRA_EXECUTABLE_SHA256="$executable_sha256" \
    HOSTNAME="$host" \
    "$binary" adynkra-11d-joint-column "$ordinal" "$artifact_root" \
    >"$attempt/stdout.log" 2>"$attempt/stderr-and-resource.log"
  rc=$?
  set -e

  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  status=failed
  if (( rc == 0 )); then
    status=complete
  fi
  cat >"$attempt/execution.json" <<EOF
{
  "schema_version": "adynkra-11d-joint-column-execution-v1",
  "column_ordinal": $ordinal,
  "worker_id": "$worker_id",
  "host": "$host",
  "process_id": $$,
  "source_revision": "$source_revision",
  "executable_sha256": "$executable_sha256",
  "started_utc": "$started",
  "finished_utc": "$finished",
  "exit_status": $rc,
  "status": "$status"
}
EOF
  sync "$attempt"

  if (( rc != 0 )); then
    echo "$column failed on $host; preserved at $attempt" >&2
    worker_failed=1
    continue
  fi

  if [[ -e "$completed_log" ]]; then
    echo "refusing to overwrite completed execution log: $completed_log" >&2
    worker_failed=1
    continue
  fi
  mv "$attempt" "$completed_log"

  column_dir="$artifact_root/complete/$column"
  if [[ ! -f "$column_dir/manifest.json" ]]; then
    echo "missing completed Rust manifest for $column" >&2
    worker_failed=1
    continue
  fi

  receipt_tmp="$artifact_root/receipts/.${column}.$$"
  (
    cd "$artifact_root"
    find "complete/$column" "logs/complete/$column" -type f -print0 |
      sort -z |
      while IFS= read -r -d '' path; do
        printf '%s  %s\n' "$(sha_file "$path")" "$path"
      done
  ) >"$receipt_tmp"
  sync "$receipt_tmp"
  mv "$receipt_tmp" "$artifact_root/receipts/${column}.sha256"
  receipt_sha=$(sha_file "$artifact_root/receipts/${column}.sha256")
  printf '%s  %s\n' "$receipt_sha" "receipts/${column}.sha256" \
    >"$artifact_root/ready/${column}.ready"
  sync "$artifact_root/receipts/${column}.sha256" "$artifact_root/ready/${column}.ready"
  echo "$column complete and ready for verified transfer"
done

exit "$worker_failed"
