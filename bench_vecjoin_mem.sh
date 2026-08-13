#!/usr/bin/env bash
# Split vs fused vector join under a constrained GPU budget.
#
# Old path holds every (left,right) partial of a partition at once:
#   n_left_batch * k * 12 * n_right_batches  ==  12 * k * N_corpus
# independent of batch size, so corpus rows and k are the knobs. The fused path
# holds a fixed handful of [n_left_batch x k] blocks regardless of corpus size.
#
# nvidia-smi is useless here (Sirius preallocates its pool up front), so the
# observable is whether each path completes inside a small configured budget.
# Low dimension keeps the pinned vectors small so the partials dominate.
set -u
DUCKDB=build/release/duckdb
EXT=$(pwd)/build/release/extension/sirius/sirius.duckdb_extension
N_CORPUS=${N_CORPUS:-5000000}
N_PROBE=${N_PROBE:-20000}
DIM=${DIM:-4}
K=${K:-50}
GPU_FRACTION=${GPU_FRACTION:-0.05}
BATCH_BYTES=${BATCH_BYTES:-262144}

cfg=/tmp/vj_cfg_$$.yaml
sed -e "s/usage_limit_fraction: 0.5/usage_limit_fraction: $GPU_FRACTION/" \
    -e "s/scan_task_batch_size: 100000000/scan_task_batch_size: $BATCH_BYTES/" \
    test/cpp/integration/integration.yaml > "$cfg"

vec="[$(python3 -c "print(', '.join(f'(i*{j+1}%997)::float' for j in range($DIM)))")]"

run_one() {
  local label="$1" streaming="$2"
  local db=/tmp/vj_bench_$$_$streaming.db
  rm -f "$db"
  local sql=/tmp/vj_bench_$$.sql
  cat > "$sql" <<SQL

CREATE TABLE corpus (id INTEGER, vec FLOAT[$DIM]);
INSERT INTO corpus SELECT i, $vec FROM range($N_CORPUS) t(i);
CREATE TABLE probe (id INTEGER, vec FLOAT[$DIM]);
INSERT INTO probe SELECT i, $vec FROM range($N_PROBE) t(i);
CHECKPOINT;
SELECT * FROM pin_table(name => 'corpus', tier => 'gpu', format => 'duckdb');
SELECT * FROM pin_table(name => 'probe',  tier => 'gpu', format => 'duckdb');
.timer on
SELECT count(*) AS n FROM sirius_knn_join('probe','vec','corpus','vec',
       search_mode => 'exact', metric => 'l2', k => $K);
SQL

  local t0 t1 out rc elapsed
  t0=$(date +%s.%N)
  out=$(SIRIUS_CONFIG_FILE="$cfg" SIRIUS_VECTOR_JOIN_STREAMING=$streaming \
        timeout 1200 "$DUCKDB" -unsigned "$db" < "$sql" 2>&1)
  rc=$?
  t1=$(date +%s.%N)
  elapsed=$(python3 -c "print(f'{$t1-$t0:.1f}')")

  if [ $rc -ne 0 ] || echo "$out" | grep -qiE "^.*(error|out of memory|bad_alloc)"; then
    echo "  $label: FAILED (rc=$rc) after ${elapsed}s"
    echo "$out" | grep -iE "error|out of memory|bad_alloc" | head -2 | sed 's/^/      /'
  else
    echo "  $label: ok in ${elapsed}s  (rows: $(echo "$out" | grep -oE "^[0-9]+$" | tail -1))"
  fi
  rm -f "$sql" "$db"
}

echo "corpus=$N_CORPUS probe=$N_PROBE dim=$DIM k=$K gpu_fraction=$GPU_FRACTION"
echo "  predicted split-path partials = 12*k*N_corpus = $(python3 -c "print(f'{12*$K*$N_CORPUS/1e9:.2f} GB')")"
echo "  batch_bytes=$BATCH_BYTES -> right batches ~ $(python3 -c "print(int($N_CORPUS*$DIM*4/$BATCH_BYTES))")"
echo "  predicted split partials = n_probe*k*12*n_right_batches = $(python3 -c "print(f'{$N_PROBE*$K*12*int($N_CORPUS*$DIM*4/$BATCH_BYTES)/1e9:.2f} GB')")"
echo "  predicted fused state    = $(python3 -c "print(f'{6*12*$K*$N_PROBE/1e6:.0f} MB')")"
run_one "split (select+reduce)" 0
run_one "fused (streaming)    " 1
rm -f "$cfg"
