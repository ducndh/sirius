#!/usr/bin/env bash
# Gate-0 full run: setup -> probe (both scan routes) -> tpch sweep.
# Usage: ./run_all.sh [config.a100.json]
set -uo pipefail
cd "$(dirname "$0")"
CFG="${1:-config.a100.json}"
NATIVE_CFG="${CFG%.json}.native.json"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=== [1/4] setup ($CFG) ==="
python3 gate0_driver.py --config "$CFG" --phase setup || exit 1

echo "=== [2/4] probe: default route ==="
python3 gate0_driver.py --config "$CFG" --phase probe \
  --out "results/probe_default_$STAMP.jsonl"

if [ -f "$NATIVE_CFG" ]; then
  echo "=== [3/4] probe: native-scan route ==="
  python3 gate0_driver.py --config "$NATIVE_CFG" --phase probe \
    --out "results/probe_native_$STAMP.jsonl"
fi

echo "=== [4/4] sweep ==="
python3 gate0_driver.py --config "$CFG" --phase sweep \
  --out "results/sweep_$STAMP.jsonl"

python3 plot_gate0.py "results/sweep_$STAMP.jsonl" || true
echo "done: results/*_$STAMP.*"
