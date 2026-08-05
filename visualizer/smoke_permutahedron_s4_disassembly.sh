#!/usr/bin/env bash
# Browser smoke test for the S4 disassembly page.
#
# The independent verifier checks the embedded data. This checks the consumer:
# it loads the built HTML in headless Chrome and confirms the page script ran to
# completion and populated the DOM. A schema mismatch between the builder and the
# template produces a valid data blob and a dead page, which only this catches.
#
# The browser self-check throws on bad geometry, which aborts initialisation
# before updateDetails() runs, leaving the panels empty. Empty panels are the
# failure signal.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
page="$here/permutahedron_s4_disassembly.html"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

if [ ! -f "$page" ]; then
  echo "FAIL: $page does not exist. Run build_permutahedron_s4_disassembly.mjs first." >&2
  exit 1
fi
if [ ! -x "$chrome" ]; then
  echo "SKIP: Chrome not found at $chrome. Set CHROME to override." >&2
  exit 0
fi

dom="$(mktemp)"
stderr="$(mktemp)"
trap 'rm -f "$dom" "$stderr"' EXIT

"$chrome" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=6000 \
  --dump-dom "file://$page" >"$dom" 2>"$stderr" || {
    echo "FAIL: Chrome exited non-zero." >&2
    cat "$stderr" >&2
    exit 1
  }

if grep -qiE "uncaught|self-check failed" "$stderr"; then
  echo "FAIL: page raised an error during initialisation." >&2
  grep -iE "uncaught|self-check failed" "$stderr" >&2
  exit 1
fi

python3 - "$dom" <<'PY'
import re, sys

dom = open(sys.argv[1]).read()
failures = []

# Panels the page fills only after selfCheck() passes and updateDetails() runs.
for probe, closing in [("sectorButtons", "div"), ("erratumNote", "div"),
                       ("pathMembers", "div"), ("quartetFacts", "dl")]:
    match = re.search(rf'id="{probe}"[^>]*>(.*?)</{closing}>', dom, re.S)
    inner = (match.group(1) if match else "").strip()
    if not inner:
        failures.append(f"#{probe} is empty, so page initialisation aborted")

# The six strands must render in ascending order with their distinct leg patterns.
buttons = re.search(r'id="sectorButtons"[^>]*>(.*?)</div>\s*<div class="stack"', dom, re.S)
labels = re.findall(r'<button[^>]*>(.*?)</button>', buttons.group(1), re.S) if buttons else []
if len(labels) != 6:
    failures.append(f"expected 6 strand buttons, rendered {len(labels)}")

expected = [("VM3", "2,6,2"), ("VM2", "2,4,2"), ("VM", "4,6,4"),
            ("TM", "6,4,6"), ("CM", "4,2,4"), ("VM1", "6,2,6")]
for index, (multiplet, legs) in enumerate(expected):
    if index >= len(labels):
        break
    text = re.sub(r"<[^>]+>", " ", labels[index])
    if multiplet not in text:
        failures.append(f"strand {index + 1} does not name {multiplet}")
    if legs not in text:
        failures.append(f"strand {index + 1} does not show legs {legs}")

# Language that must not survive the ascending-weight conversion.
for stale in ["H1-H4", "hopping operator", "root distance", "Hopper Separation"]:
    if stale in dom:
        failures.append(f"stale wording still rendered: {stale}")

if failures:
    print("FAIL: browser smoke test")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("PASS: page initialised, six strands rendered in ascending order with distinct legs.")
PY
