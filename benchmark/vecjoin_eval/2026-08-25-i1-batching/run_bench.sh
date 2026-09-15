#!/usr/bin/env bash
# I1 regression + perf harness. One argument: a tag for the output directory.
#
# ONE session for every configuration, deliberately: the clustering is session state and
# cuvs::cluster::kmeans::fit is not bit-stable across processes (~1 session in 6 lands a
# different centroid set). The probe side is assigned from those centroids at join time, so
# two sessions with different centroids are not comparable at all. The session's centroid
# hash is recorded next to the results; only runs sharing it may be diffed.
#
# Per configuration: 3 timed reps, the phase breakdown, an aggregate fingerprint, and the
# full ordered (left_id, right_id) result. The fingerprint is over data-dependent columns --
# count(*) alone folds to a constant (QUEUE B5).
set -euo pipefail
TAG="${1:?usage: run_bench.sh <tag>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results/$TAG"
DB=/var/tmp/vj/i1/bench.db
DUCKDB=~/vecjoin/sirius/build/release/duckdb
mkdir -p "$OUT"

CONFIGS="corpus64:c64:1 corpus64:c64:2 corpus64:c64:4 corpus64:c64:8 corpus64:c64:16 corpus256:c256:32"

SQL="$OUT/all.sql"
cat "$HERE/prelude.sql" > "$SQL"
for cfg in $CONFIGS; do
  IFS=: read -r corpus clus probes <<<"$cfg"
  name="${clus}_p${probes}"
  probe="probe64"; [ "$clus" = "c256" ] && probe="probe256"
  J="sirius_knn_join('$probe','vec','$corpus','vec', probe_source => 'scan', metric => 'l2', k => 10, search_mode => 'approx', clustering => '$clus', cluster_column => 'cluster_id', n_probes => $probes)"
  {
    echo ".print @@@CONFIG $name"
    echo ".mode trash"
    echo ".timer on"
    for _ in 1 2 3; do echo "SELECT left_id, right_id, distance FROM $J;"; done
    echo ".timer off"
    echo ".mode csv"
    echo ".headers on"
    echo ".output $OUT/fingerprint_${name}.csv"
    echo "SELECT count(*) AS n, sum(left_id::HUGEINT) AS sum_left, sum(right_id::HUGEINT) AS sum_right, sum(distance::DOUBLE) AS sum_dist FROM $J;"
    echo ".output $OUT/rows_${name}.csv"
    echo "SELECT left_id, right_id FROM $J ORDER BY left_id, right_id;"
    echo ".output"
  } >> "$SQL"
done

SIRIUS_VECTOR_JOIN_PHASE_DEBUG=1 SIRIUS_VECTOR_JOIN_PRUNE_DEBUG=1 \
  "$DUCKDB" -unsigned "$DB" < "$SQL" > "$OUT/run.out" 2> "$OUT/run.err"

grep -o "centroids=[0-9a-f]*" "$OUT/run.err" | sort -u > "$OUT/centroid_hashes.txt"
echo "centroid hashes this session:"; sed 's/^/  /' "$OUT/centroid_hashes.txt"
awk '/@@@CONFIG/{c=$2} /^Run Time/{print c, $5}' "$OUT/run.out" | tee "$OUT/times.txt"
echo "results in $OUT"
