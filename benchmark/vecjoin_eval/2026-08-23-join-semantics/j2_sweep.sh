#!/usr/bin/env bash
# J2 — does J1's conclusion hold across selectivity? Sweeps the predicate on the CORPUS side.
#
# J1 measured one point (30%) and found we lose on corpus materialization, not on the join.
# The crossover matters: at low selectivity our materialize shrinks while cuVS still pays a full
# index build; at high selectivity the reverse. One point cannot be generalized.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG=${SIRIUS_CONFIG_FILE:-$HERE/../../vecjoin_bench.yaml}
DUCKDB=${DUCKDB:-/var/tmp/vjbuild/release/duckdb}
mkdir -p /var/tmp/ddb_spill
for N in 1 3 5 9; do          # id % 10 < N  ->  10%, 30%, 50%, 90%
  SEL=$(python3 -c "print($N/10)")
  echo "########## selectivity ${N}0%"
  sed "s/id % 10 < 3/id % 10 < $N/" "$HERE/j1_sirius.sql" > /var/tmp/vj/j2_$N.sql
  rm -f /var/tmp/vj/j2_$N.db*
  SIRIUS_CONFIG_FILE="$CFG" SIRIUS_LOG_BACKEND=noop \
    "$DUCKDB" -unsigned /var/tmp/vj/j2_$N.db < /var/tmp/vj/j2_$N.sql 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | grep -E '@@@|Run Time' \
    | awk -v s="${N}0%" 'BEGIN{ORS=""} /@@@/{next} /Run Time/{n++; split($0,a,"real "); split(a[2],b," "); t[n]=b[1]}
        END{printf "  sirius sel=%s  materialize %s  checkpoint %s  pin %s  join %s  TOTAL %.3f\n",
            s, t[1], t[2], t[3], t[4], t[1]+t[2]+t[3]+t[4]}'
  python3 "$HERE/j1_intermediate_corpus.py" --selectivity "$SEL" --n-probes 1024 2>&1 \
    | grep -E '^RESULT' \
    | awk -v s="${N}0%" '{printf "  cuvs   sel=%s  export %s  h2d %s  build %s  search %s  mapback %s  TOTAL %s  recall %s\n", s,$5,$6,$7,$8,$9,$10,$11}'
    # RESULT layout: $1..$3 = RESULT j1 cuvs, $4 = selectivity, then export h2d build search
    # mapback total recall. An earlier version started at $4 and shifted every column by one.
done
