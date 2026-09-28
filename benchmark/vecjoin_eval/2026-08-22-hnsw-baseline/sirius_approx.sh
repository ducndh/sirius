#!/usr/bin/env bash
# Sirius clustered approximate join on the SAME workload as the HNSW baseline in this
# directory: SIFT1M, 10k probes x 1M corpus, k=10, L2, recall against the packaged ground
# truth. Produces the recall/time curve that pairs with hnsw_join.sql's curve.
#
# Derived from ../../approx_join_bench.sh with ONE correction: the timed runs there used
# `.output /dev/null`, which does NOT avoid duckbox's ~1.15 us/row formatting cost (see
# ../README.md). They use `.mode trash` here. At 100k output rows that is ~0.115 s of shell
# formatting folded into every timing -- irrelevant at 80 s, decisive at 0.3 s.
#
# Everything runs in ONE session on purpose: the clustering lives in a session-scoped cache and
# the corpus's cluster_id is expressed in that clustering's label space. A second process would
# re-fit different centroids and prune against labels that no longer mean the same thing.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

DATA=/var/tmp/vj/data/parquet
DB=${DB:-/var/tmp/vj/approx_hnswcmp.db}
SIRIUS=${SIRIUS:-$HOME/vecjoin/sirius}
DUCKDB=${DUCKDB:-$SIRIUS/build/release/duckdb}
EXT=${EXT:-$SIRIUS/build/release/extension/sirius/sirius.duckdb_extension}
N_CLUSTERS=${N_CLUSTERS:-1024}
K=${K:-10}
PROBES=${PROBES:-"1 4 16 64 256"}
RES=/var/tmp/vj/hcmp
SQL=/var/tmp/vj/approx_hnswcmp.sql

[ -f "$DATA/sift-128-euclidean_base.parquet" ] || { echo "MISSING SIFT1M -- see ../../README.md"; exit 1; }
[ -x "$DUCKDB" ] || { echo "MISSING $DUCKDB -- build Sirius for the CURRENT arch first"; exit 1; }
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
rm -f "$DB" "$RES"_*.csv
mkdir -p /var/tmp/vj

{
  # NO explicit LOAD: this duckdb binary is the Sirius build and auto-loads the extension at
  # startup. A second `LOAD '<path>'` runs sirius_duckdb_cpp_init AGAIN, which tries to allocate
  # a SECOND GPU pool. At the default 0.95 usage fraction that second pool cannot fit, init
  # throws, and the damage is silent: later statements still report Success and the first kmeans
  # call dies inside cuVS with cudaErrorIllegalAddress -- which reads like a kernel bug. Verified
  # 2026-08-23: without LOAD, 9 sirius functions and a clean init at 0.85.
  echo "CREATE TABLE base  AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_base.parquet');"
  echo "CREATE TABLE query AS SELECT id, vec::FLOAT[128] AS vec FROM read_parquet('$DATA/sift-128-euclidean_query.parquet');"
  echo "CREATE TABLE gt    AS SELECT * FROM read_parquet('$DATA/sift-128-euclidean_gt.parquet');"
  echo "CHECKPOINT;"
  echo "SELECT * FROM pin_table(name => 'base', tier => 'gpu', format => 'duckdb');"
  echo "SELECT * FROM pin_table(name => 'query', tier => 'gpu', format => 'duckdb');"
  echo ".print === TIMED kmeans fit (n_clusters=$N_CLUSTERS)"
  echo ".timer on"
  echo "SELECT * FROM sirius_kmeans_fit('base','vec', name => 'c', n_clusters => $N_CLUSTERS);"
  echo ".timer off"
  echo "CREATE TABLE asg AS SELECT * FROM sirius_kmeans_assign('base','vec','c', n_probes => 1);"
  echo "CREATE TABLE corpus AS SELECT b.id, b.vec, a.cluster_id FROM base b JOIN asg a ON b.rowid = a.row_id ORDER BY a.cluster_id;"
  echo "CHECKPOINT;"
  echo ".print === cluster size distribution (min/max/avg/count)"
  echo "SELECT min(n), max(n), round(avg(n),1), count(*) FROM (SELECT cluster_id, count(*) n FROM corpus GROUP BY cluster_id);"
  echo "SELECT * FROM unpin_table('base');"
  echo "SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');"
  echo "CREATE TABLE qasg AS SELECT * FROM sirius_kmeans_assign('query','vec','c', n_probes => 1);"
  echo "CREATE TABLE probe AS SELECT q.id, q.vec, a.cluster_id FROM query q JOIN qasg a ON q.rowid = a.row_id ORDER BY a.cluster_id;"
  echo "CHECKPOINT;"

  # Each case runs twice: once under the timer to `.mode trash`, once to CSV for recall.
  # Timing a run that also writes 100k CSV rows would fold the writer into the measurement.
  emit_case() { # emit_case <label> <csv-suffix> <sql>
    echo ".print === TIMED $1"
    echo ".timer on"; echo ".mode trash"
    echo "$3"
    echo ".timer off"; echo ".mode csv"; echo ".headers on"
    echo ".output ${RES}_$2.csv"
    echo "$3"
    echo ".output"
    echo ".mode duckbox"
  }
  emit_case "exact-gemm" exact \
    "SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => $K, search_mode => 'exact-gemm');"
  for p in $PROBES; do
    emit_case "approx n_probes=$p" "p$p" \
      "SELECT left_id, right_id FROM sirius_knn_join('probe','vec','corpus','vec', probe_source => 'scan', metric => 'l2', k => $K, search_mode => 'approx', clustering => 'c', cluster_column => 'cluster_id', n_probes => $p);"
  done

  echo ".print === recall vs packaged ground truth"
  echo "CREATE TABLE curve (mode VARCHAR, n_probes INTEGER, hits BIGINT, rows BIGINT);"
  echo "INSERT INTO curve SELECT 'exact', 0, count(*) FILTER (WHERE g.neighbor_id IS NOT NULL), count(*) FROM read_csv('${RES}_exact.csv', header=true) r LEFT JOIN gt g ON g.query_id = r.left_id AND g.rank < $K AND g.neighbor_id = r.right_id;"
  for p in $PROBES; do
    echo "INSERT INTO curve SELECT 'approx', $p, count(*) FILTER (WHERE g.neighbor_id IS NOT NULL), count(*) FROM read_csv('${RES}_p${p}.csv', header=true) r LEFT JOIN gt g ON g.query_id = r.left_id AND g.rank < $K AND g.neighbor_id = r.right_id;"
  done
  echo "SELECT mode, n_probes, rows, round(hits::DOUBLE / rows, 4) AS recall_at_k FROM curve ORDER BY n_probes;"
} > "$SQL"

echo "running $(grep -c sirius_knn_join "$SQL") joins in one session..."
SIRIUS_CONFIG_FILE=${SIRIUS_CONFIG_FILE:-$HERE/../../vecjoin_bench.yaml} \
  SIRIUS_LOG_BACKEND=noop "$DUCKDB" -unsigned "$DB" < "$SQL" 2>&1 | tee /var/tmp/vj/approx_hnswcmp.log

echo
echo "=== timings (file order: fit, exact, then each n_probes) ==="
grep -oE "Run Time \(s\): real [0-9.]+" /var/tmp/vj/approx_hnswcmp.log | grep -oE "[0-9.]+$" | \
  awk -v probes="$PROBES" 'BEGIN{n=split(probes,p," "); lab[1]="fit"; lab[2]="exact"; for(i=1;i<=n;i++) lab[i+2]="approx n_probes="p[i]}
       {t[NR]=$1; printf "%-22s %8.3fs", lab[NR], $1; if(NR>2 && t[2]>0) printf "   %6.2fx vs exact", t[2]/$1; print ""}'
