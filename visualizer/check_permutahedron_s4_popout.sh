#!/usr/bin/env bash
# Interaction gate for the six-strand pop-out.
#
# smoke_permutahedron_s4_disassembly.sh checks that the page initialises. This
# checks that clicking it six times does the right thing: strands leave in listed
# order, the glow walks each quartet's journey link by link, and every vacated
# permutahedron site gets a white ball.
#
# The page carries a harness behind ?phase3check=1. It clicks the real button and
# reports what the renderer drew, so the white-ball and glow counts come from the
# draw path rather than from state flags. The browser self-check independently
# validates each journey against the graph, so the harness is not merely
# comparing the animation against its own input.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
page="$here/permutahedron_s4_disassembly.html"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

if [ ! -f "$page" ]; then
  echo "FAIL: $page does not exist. Run build_permutahedron_s4_disassembly.mjs first." >&2
  exit 1
fi
if [ ! -x "$chrome" ]; then
  # A skip that exits 0 is a silent pass. If the browser is genuinely
  # unavailable, set ALLOW_MISSING_CHROME=1 deliberately.
  echo "FAIL: Chrome not found at $chrome. Set CHROME to override, or ALLOW_MISSING_CHROME=1 to skip on purpose." >&2
  [ "${ALLOW_MISSING_CHROME:-}" = "1" ] && { echo "skipping by explicit request" >&2; exit 0; }
  exit 1
fi

dom="$(mktemp)"
trap 'rm -f "$dom"' EXIT

"$chrome" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=240000 \
  --dump-dom "file://$page?phase3check=1" >"$dom" 2>/dev/null || {
    echo "FAIL: Chrome exited non-zero." >&2
    exit 1
  }

# Orientation gate. The page emits its rendered node positions; align them to the
# positions measured off the printed HowardTLK.v2.pdf p.64 figure and require the
# rotation to be near zero. Data-level checks all stayed green while a leftover
# camera pitch rotated the actual view 19.5 degrees off the fitted matrix, so this
# measures the pixels the browser drew rather than the matrix it drew them from.
orientdom="$(mktemp)"
trap 'rm -f "$dom" "$orientdom"' EXIT
"$chrome" --headless=new --disable-gpu --no-sandbox --virtual-time-budget=20000 \
  --dump-dom "file://$page?positions=1" >"$orientdom" 2>/dev/null || {
    echo "FAIL: Chrome exited non-zero during the orientation check." >&2; exit 1; }

python3 "$here/verify_s4_rendered_positions.py" "$orientdom"

python3 - "$dom" <<'PY'
import html, json, re, sys

dom = open(sys.argv[1]).read()

if 'id="phase3started"' not in dom:
    print("FAIL: the harness never ran. Page initialisation aborted before it.")
    sys.exit(1)

match = re.search(r'<pre id="phase3result">(.*?)</pre>', dom, re.S)
if not match:
    print("FAIL: the harness started but never finished, so a pop-out never completed.")
    sys.exit(1)

report = json.loads(html.unescape(match.group(1)))
failures = list(report["failures"])
observations = report["observations"]

if len(observations) != 6:
    failures.append(f"expected 6 pop-outs, observed {len(observations)}")

# Every quartet vacates four sites, so the white balls accumulate 4 at a time.
for index, observation in enumerate(observations, start=1):
    if observation["whiteBalls"] != index * 4:
        failures.append(
            f"after pop {index} the renderer drew {observation['whiteBalls']} white balls, expected {index * 4}"
        )
    if observation["strand"] != index:
        failures.append(f"click {index} popped strand {observation['strand']}")
    if observation["glowLinks"] < 1:
        failures.append(f"click {index} rendered no glow links")

if failures:
    print("FAIL: pop-out interaction gate")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

order = " ".join(f"{o['strand']}:{o['multiplet']}({o['glowLinks']})" for o in observations)
print("PASS: six clicks, six strands in order, 24 white balls, glow order matches each journey.")
print(f"      {order}")
PY
