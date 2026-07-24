#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 <run-root> [workers]" >&2
  exit 64
fi

run_root=${1%/}
workers=${2:-4}
if (( workers < 1 )); then
  echo "workers must be positive" >&2
  exit 64
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary="$repo_root/target/release/adinkra-codespace"
[[ -x "$binary" ]] || {
  echo "missing release binary: $binary" >&2
  exit 66
}

mkdir -p "$run_root/logs"
executable_sha256=$(shasum -a 256 "$binary" | awk '{print $1}')
source_revision="$(
  cd "$repo_root"
  printf '%s+worktree-%s' \
    "$(git rev-parse HEAD)" \
    "$(git diff --binary | shasum -a 256 | awk '{print $1}')"
)"

run_one() {
  local form_degree=$1
  local leading_ordinal=$2
  local padded
  padded=$(printf '%03d' "$leading_ordinal")
  mkdir -p "$run_root/logs/form-$form_degree"
  ADINKRA_SOURCE_REVISION="$source_revision" \
  ADINKRA_EXECUTABLE_SHA256="$executable_sha256" \
    /usr/bin/time -l "$binary" \
      adynkra-11d-gauge-zero-column \
      "$form_degree" \
      "$leading_ordinal" \
      "$run_root" \
      >"$run_root/logs/form-$form_degree/column-$padded.stdout.json" \
      2>"$run_root/logs/form-$form_degree/column-$padded.resource.log"
}

pids=()
labels=()
wait_batch() {
  local index
  local status=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "failed: ${labels[$index]}" >&2
      status=1
    fi
  done
  pids=()
  labels=()
  return "$status"
}

for form_degree in 0 1 2 3 4 5; do
  for leading_ordinal in $(seq 0 11); do
    run_one "$form_degree" "$leading_ordinal" &
    pids+=("$!")
    labels+=("form=$form_degree column=$leading_ordinal")
    if (( ${#pids[@]} == workers )); then
      wait_batch
    fi
  done
done
if (( ${#pids[@]} > 0 )); then
  wait_batch
fi

completed=$(
  find "$run_root/complete" -mindepth 3 -maxdepth 3 \
    -type f -name manifest.json | wc -l | tr -d ' '
)
printf 'completed=%s expected=72\n' "$completed"
(( completed == 72 ))
