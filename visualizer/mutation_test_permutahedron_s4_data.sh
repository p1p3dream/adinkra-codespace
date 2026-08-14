#!/usr/bin/env bash
# Prove that the independent data verifier rejects representative defects.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
page="$here/permutahedron_s4_disassembly.html"
files=("$(mktemp)" "$(mktemp)" "$(mktemp)")
trap 'rm -f "${files[@]}"' EXIT

python3 - "$page" "${files[@]}" <<'PY'
import copy
import json
import re
import sys

source = open(sys.argv[1]).read()
match = re.search(r"const DISASSEMBLY=(\{.*?\});", source)
if not match:
    raise SystemExit("embedded DISASSEMBLY data not found")
base = json.loads(match.group(1))

mutations = []

swap = copy.deepcopy(base)
vm1 = next(chain for chain in swap["chains"] if chain["multiplet"] == "VM1")
vm1["labels"][2], vm1["labels"][3] = vm1["labels"][3], vm1["labels"][2]
mutations.append(swap)

leg = copy.deepcopy(base)
tm = next(chain for chain in leg["chains"] if chain["multiplet"] == "TM")
tm["legs"][0] += 1
mutations.append(leg)

face = copy.deepcopy(base)
face["base_face_cycle"][0] = "1243"
mutations.append(face)

for output, data in zip(sys.argv[2:], mutations):
    encoded = json.dumps(data, separators=(",", ":"))
    built = source[:match.start(1)] + encoded + source[match.end(1):]
    open(output, "w").write(built)
PY

names=(quartet-order leg-distance base-face)
for index in 0 1 2; do
  if S4_DISASSEMBLY_HTML="${files[$index]}" node "$here/verify_permutahedron_s4_disassembly.mjs" >/dev/null 2>&1; then
    echo "FAIL: data verifier accepted the ${names[$index]} mutation." >&2
    exit 1
  fi
  echo "PASS: data verifier rejected the ${names[$index]} mutation."
done
