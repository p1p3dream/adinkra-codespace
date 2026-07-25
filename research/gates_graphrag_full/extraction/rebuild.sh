#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python3 research/gates_graphrag_full/extraction/build_full_extraction.py --clean "$@"
python3 research/gates_graphrag_full/extraction/verify_determinism.py
python3 research/gates_graphrag_full/extraction/validate_extraction.py
