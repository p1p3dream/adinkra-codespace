#!/usr/bin/env bash
# Browser gate for camera-preset restoration and state-dependent UI text.

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
stderr="$(mktemp)"
trap 'rm -f "$dom" "$stderr"' EXIT

"$chrome" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=10000 \
  --dump-dom "file://$page?controlscheck=1" >"$dom" 2>"$stderr" || {
    echo "FAIL: Chrome exited non-zero." >&2
    cat "$stderr" >&2
    exit 1
  }

python3 - "$dom" <<'PY'
import html
import json
import re
import sys

dom = open(sys.argv[1]).read()
match = re.search(r'<pre id="controlsCheckResult"[^>]*>(.*?)</pre>', dom, re.S)
if not match:
    print("FAIL: the camera and control harness never completed.")
    sys.exit(1)

report = json.loads(html.unescape(match.group(1)))
failures = list(report["failures"])

required = [
    "Show permutahedron edges",
    "arXiv:1701.00304 Appendix B",
    "Howard p.61 order, with members ascending inside each quartet",
    "One click moves one strand. Every other node remains fixed",
    "Separate strand 1 of 6",
]
for text in required:
    if text not in dom:
        failures.append(f"required source or ordering text is absent: {text}")

for stale in [
    "Permutahedron edges while assembled",
    "S4_six_quartets_ascending.png",
    "strands in ascending four-digit order",
    "Ringed nodes lie on the base hexagon",
    "Separate all",
    "progressSlider",
    "Assembly → six quartet strands",
    "Pop out strand",
]:
    if stale in dom:
        failures.append(f"stale or false wording survives: {stale}")

if failures:
    print("FAIL: camera and control-state gate")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("PASS: both camera presets restore their fitted view, simultaneous separation controls are absent, and source labels are explicit.")
PY
