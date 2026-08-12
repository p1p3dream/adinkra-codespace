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

python3 - "$orientdom" <<'ORIENT'
import sys, re, html, json, math

dom = open(sys.argv[1]).read()
match = re.search(r'<pre id="renderedPositions"[^>]*>(.*?)</pre>', dom, re.S)
if not match:
    print("FAIL: the page did not emit rendered positions.")
    sys.exit(1)
ours = json.loads(html.unescape(match.group(1)))

# Measured by reading all 24 labels off the 1881x1708 render of the printed
# figure. Independent of this codebase.
printed = {
 "4321":(821,273),"4312":(1231,344),"3421":(529,363),"3412":(952,438),
 "4231":(762,480),"4132":(1551,619),"3241":(187,670),"4213":(1071,734),
 "2431":(449,757),"4123":(1452,809),"3142":(1020,848),"2341":(151,861),
 "2413":(755,1004),"1432":(1628,1007),"3214":(227,1068),"1342":(1367,1135),
 "3124":(640,1170),"1423":(1520,1181),"2314":(199,1246),"2143":(804,1365),
 "1324":(1001,1449),"1243":(1188,1463),"2134":(528,1498),"1234":(929,1606)}
STRETCH = 1.1419   # the printed PNG is horizontally stretched; discs are ellipses

common = sorted(set(ours) & set(printed))
if len(common) != 24:
    print(f"FAIL: matched {len(common)} labels, expected 24.")
    sys.exit(1)

ax = [ours[l][0] for l in common]; ay = [ours[l][1] for l in common]
bx = [printed[l][0]/STRETCH for l in common]; by = [printed[l][1] for l in common]
amx, amy = sum(ax)/24, sum(ay)/24
bmx, bmy = sum(bx)/24, sum(by)/24
ax = [v-amx for v in ax]; ay = [v-amy for v in ay]
bx = [v-bmx for v in bx]; by = [v-bmy for v in by]

# Best-fit in-plane rotation, closed form for 2D Procrustes.
num = sum(ax[i]*by[i] - ay[i]*bx[i] for i in range(24))
den = sum(ax[i]*bx[i] + ay[i]*by[i] for i in range(24))
theta = math.degrees(math.atan2(num, den))

# Rotation alone is NOT sufficient. A camera pitch tilts the solid toward the
# viewer, which compresses the projection vertically without rotating it in
# plane, so the best-fit angle stays near zero while the view is plainly wrong.
# The residual after alignment is what catches that, so both are gated.
cos_t, sin_t = math.cos(math.radians(theta)), math.sin(math.radians(theta))
scale = den * cos_t + num * sin_t
scale /= sum(ax[i]*ax[i] + ay[i]*ay[i] for i in range(24))
resid = []
for i in range(24):
    px = scale * (ax[i]*cos_t - ay[i]*sin_t)
    py = scale * (ax[i]*sin_t + ay[i]*cos_t)
    resid.append(math.hypot(bx[i]-px, by[i]-py))
mean_resid = sum(resid)/24
span = 2 * max(math.hypot(bx[i], by[i]) for i in range(24))

ANGLE_TOLERANCE = 1.5     # correct build measures about 0.02
RESID_TOLERANCE = 30.0    # correct build measures about 17 px on a 1350 px span
failures = []
if abs(theta) > ANGLE_TOLERANCE:
    failures.append(f"rendered layout needs {theta:+.2f} deg of rotation to match the printed figure (tolerance {ANGLE_TOLERANCE})")
if mean_resid > RESID_TOLERANCE:
    failures.append(f"mean residual {mean_resid:.1f} px after best alignment, tolerance {RESID_TOLERANCE} px on a {span:.0f} px span; a camera pitch or yaw distorts the layout without rotating it")
if failures:
    print("FAIL: orientation gate")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASS: rendered layout matches the printed p.64 figure, {theta:+.2f} deg rotation, {mean_resid:.1f} px mean residual.")
ORIENT

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
