#!/usr/bin/env bash
# Prove that the rendered-orientation gate rejects representative defects.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
page="$here/permutahedron_s4_disassembly.html"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

if [ ! -f "$page" ]; then
  echo "FAIL: $page does not exist. Run build_permutahedron_s4_disassembly.mjs first." >&2
  exit 1
fi
if [ ! -x "$chrome" ]; then
  echo "FAIL: Chrome not found at $chrome. Set CHROME to override, or ALLOW_MISSING_CHROME=1 to skip on purpose." >&2
  [ "${ALLOW_MISSING_CHROME:-}" = "1" ] && { echo "skipping by explicit request" >&2; exit 0; }
  exit 1
fi

dom="$(mktemp)"
trap 'rm -f "$dom"' EXIT
"$chrome" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=20000 \
  --dump-dom "file://$page?positions=1" >"$dom" 2>/dev/null

python3 "$here/verify_s4_rendered_positions.py" "$dom"

for mutation in swap-labels shift-node mirror rotate; do
  if python3 "$here/verify_s4_rendered_positions.py" "$dom" --mutate "$mutation" >/dev/null 2>&1; then
    echo "FAIL: orientation gate accepted the $mutation mutation." >&2
    exit 1
  fi
  echo "PASS: orientation gate rejected the $mutation mutation."
done
