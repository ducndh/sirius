# Vector join over a corpus larger than memory

`sirius_knn_join` with `build_source => 'scan'` reads its corpus through the engine's own scan, and
the engine keeps each batch on whatever tier has room: device, host pool, or the disk tier on an
NVMe drive. Nothing is pinned whole, so the corpus can be larger than GPU and host memory together.
This page covers the approximate (clustered) top-k join at that size: build, config, data, the
statement, the switches, sizing, and what has and has not been tested.

## What the join does

1. **Build (one pass over the table).** With a clustering and no `cluster_column`, the corpus scan
   feeds a label operator that adds each row's nearest centroid as an INT32 column, and each
   labelled batch is cut into fixed cluster-id ranges. The ranges are parked on the tiers like any
   other batch. No pre-sorted copy of the table is needed.
2. **Probe blocks.** The probe rows are split into blocks sized to the device (the
   `probe blocks: N of up to R rows` line). For each block the corpus is streamed past it once, in
   cluster order, with chunk reads running ahead of the search. The search is a bounded FP16 filter
   plus an FP32 (or half, see below) re-score of the candidates, over the `n_probes` clusters
   nearest each probe row.
3. **Self-join.** When probe and corpus are the same table and column, one scan feeds both sides,
   so the table is materialized and spilled once.

## Requirements

- An NVIDIA GPU and a driver for CUDA 13 (580 or newer). With the 560 driver the engine fails with
  `cudaErrorCallRequiresNewerDriver` (`cudaMemcpyBatchAsync`). The default pixi environment brings
  the CUDA 13.5 toolkit, libcudf 26.10 and libcuvs 26.10.
- x86_64 or aarch64; both have been built (A100 sm_80 on x86, GH200 sm_90a on aarch64).
- One or more NVMe drives for the disk tier, formatted XFS (ext4 serialized DuckDB's parallel
  writes when loading the database). We used one drive for the database and another for the tier.
- `pixi` (https://pixi.sh).

## Build

```bash
git clone -b vecjoin-disk-stable-1118 https://github.com/ducndh/sirius.git sirius && cd sirius
git submodule sync --recursive && git submodule update --init --recursive
./third_party_patches/apply.sh           # cuCascade patch, see below; safe to run again
pixi install

# The GPU decides the architecture: a -real build for one arch does not run on another.
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
#   7.5 -> 75-real, 8.0 -> 80-real, 8.6 -> 86-real, 9.0 -> 90a-real (10.0 -> 100f-real, never built)
ARCH=80-real

# sccache opens many files: raise the descriptor limit and restart its server under it.
ulimit -n $(ulimit -Hn); pixi run sccache --stop-server 2>/dev/null
CMAKE_BUILD_PARALLEL_LEVEL=128 pixi run bash -c "export CUDAARCHS=$ARCH; make release && make sirius-duckdb" > build.log 2>&1
grep -E "CMake Error|error:" build.log | head; grep CMAKE_CUDA_ARCHITECTURES: build/release/CMakeCache.txt
```

- Set `CUDAARCHS` inside `pixi run`: the environment's activation sets it to all seven
  architectures, which builds far longer.
- Check the log rather than the exit status; a CMake error can still end with status 0.
- At 256 jobs sccache ran out of file handles (`os error 24`); the raised limit and 128 jobs worked.
- Outputs: the CLI `sirius-duckdb/build/release/duckdb`, the extension
  `sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension`, the unit tests
  `build/release/test/cpp/sirius_unittest`.

### cuCascade patch

The branch needs two changes to the `cucascade` submodule (base commit `f4c553c`) that are not in
upstream cuCascade yet: a `direct_io` option for the disk tier (`O_DIRECT` reads and writes; the
engine's `memory.disk.direct_io` key passes it through, and the build fails without it), and a
wake-up of reservation waiters when fixed-size host blocks return to the pool. They are carried as
`third_party_patches/cucascade_direct_io_and_wake.patch`; `third_party_patches/apply.sh` checks the
submodule is at the recorded commit and applies the patch once. Afterwards `git status` shows
`cucascade` as modified, which is expected. Run it again after any `git submodule update`, which
resets the submodule.

### Unit gate

```bash
ulimit -n $(ulimit -Hn)
NVIDIA_TF32_OVERRIDE=0 pixi run build/release/test/cpp/sirius_unittest "[vss],[vector_join]"
```

On a GH200: 351 test cases, 349 passed, 2 skipped. Compare case counts, not assertion totals: one
case's assertion count varies from run to run. On an A100 40 GB, the same cases fail as on the base
branch (see Known issues): 4 planner-path cases always, and a near-tie case in some runs.

## Config

`SIRIUS_CONFIG_FILE` points at a YAML file. The fields that matter for this join (values are the
ones used for the 465 GB runs on a GH200 96 GB / 480 GB host):

```yaml
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.563     # or usage_limit_bytes: <bytes>
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 80000000000     # host pool; must leave room for DuckDB and per-row state
      downgrade_trigger_fraction: 0.85
      downgrade_stop_fraction: 0.7
      initial_number_pools: 10
      pool_size: 512
      block_size: 1048576
    disk:
      disk_id: 0
      capacity_bytes: 700000000000    # below the drive's free space, see Sizing
      downgrade_root_dirs: /mnt/tier/disk_tier
      direct_io: true                 # read the corpus back with O_DIRECT, not through the page cache
  executor:
    scan_manager:
      uring_n_reactors: 16
    pipeline:
      num_threads: 8
    task_creator:
      num_threads: 4
    downgrade:
      num_threads: 1
      monitor_period: 10ms
  operator_params:
    scan_task_batch_size: 100000000
    max_sort_partition_bytes: 4000000000  # build's cluster-range size; 0 = 2 GiB, at most 4 GiB
    concat_batch_bytes: 250000000         # 250 MB with a half-precision tier, 500 MB without
    hash_partition_bytes: 100000000
    max_build_hash_table_bytes: 90000000
```

Run DuckDB with its own memory limit as well (`SET memory_limit='30GB'`); the host pool is separate
from DuckDB's buffer manager. Empty `downgrade_root_dirs` before each run: a killed run leaves its
spill files.

## Running the CLI

The binary needs the pixi environment's libraries:

```bash
ulimit -n $(ulimit -Hn)
pixi run env SIRIUS_CONFIG_FILE=$PWD/run.yaml SIRIUS_LOG_BACKEND=spdlog SIRIUS_LOG_LEVEL=info \
  SIRIUS_LOG_DIR=$PWD/logs SIRIUS_VECTOR_JOIN_PHASE_DEBUG=1 <switches> \
  sirius-duckdb/build/release/duckdb -unsigned \
  -cmd "LOAD 'sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension'" \
  -cmd "SET memory_limit='30GB'" /path/to/db.duckdb < run.sql
```

`SIRIUS_VECTOR_JOIN_PHASE_DEBUG=1` prints the block plan and per-phase times; it costs nothing
measurable and is the main progress signal. On a GH200 we ran under `numactl --membind=0` so host
memory comes from the CPU's NUMA node.

## Loading a corpus and a clustering

The vectors as `FLOAT[d]` in a DuckDB table, checkpointed (pins and scans read checkpointed rows):

```sql
SET force_compression='uncompressed'; SET preserve_insertion_order=false;
SET temp_directory='/nvme/duckdb_tmp'; SET threads=64;
CREATE TABLE base AS SELECT id::BIGINT AS id, vec::FLOAT[1024] AS vec FROM read_parquet('/data/pq/part-*.parquet');
CHECKPOINT;
SELECT count(*), count(DISTINCT id) FROM base;
```

Do not load while a build runs on the same machine: our first load beside a build was OOM-killed.
113.5M rows x 1,024-d (465 GB) loaded in 563 s with `memory_limit` 120 GB.

Fit a clustering on a ~1% sample and keep the centroids as a table:

```sql
CREATE TABLE sample AS SELECT id, vec FROM base WHERE hash(id) % 100 = 0; CHECKPOINT;
SELECT * FROM pin_table(name => 'sample', tier => 'gpu', format => 'duckdb');
SELECT * FROM sirius_kmeans_fit('sample', 'vec', name => 'cvec', n_clusters => 8192);
CREATE TABLE cents AS SELECT cluster_id AS id, list(value ORDER BY dim_index)::FLOAT[1024] AS vec
  FROM sirius_kmeans_centroids('cvec') GROUP BY cluster_id ORDER BY cluster_id;
CHECKPOINT;
```

In each later session, recreate the clustering from the table without refitting (both statements
go in the same script as the join):

```sql
SELECT * FROM pin_table(name => 'cents', tier => 'gpu', format => 'duckdb');
SELECT * FROM sirius_kmeans_fit('cents', 'vec', name => 'cs', rows_are_centroids => true);
```

## The join

Self-join, 10 nearest neighbours per row, 32 clusters probed:

```sql
SELECT count(*) AS pairs, sum(CASE WHEN left_id = right_id THEN 1 ELSE 0 END) AS self_hits,
       round(sum(distance), 3) AS dist_sum
FROM sirius_knn_join('base', 'vec', 'base', 'vec', probe_source => 'scan', build_source => 'scan',
                     metric => 'l2', k => 10, search_mode => 'approx', clustering => 'cs', n_probes => 32);
```

Two tables: the first pair of arguments is the probe table and column, the second the corpus:
`sirius_knn_join('queries', 'vec', 'base', 'vec', ...)` with the same options. The same table on both
sides with `SIRIUS_VSS_SELF_JOIN_SHARE=0` runs the two-table shape. Omitting `cluster_column` is what
selects the one-pass build; a table already written in cluster order can pass
`cluster_column => 'cluster'` instead.

To keep the pairs, write them out (`COPY (SELECT ...) TO 'pairs.parquet'`) instead of aggregating.

## Switches

Environment variables, read at plan or run time. Defaults are what you get with the variable unset.

| Variable | Default | What it does, when to use it |
|---|---|---|
| `SIRIUS_VSS_TIER_HALF` | unset (FP32 tier) | `1`: store the corpus in half precision, halving tier bytes and reads. Only for data FP16 holds exactly: the build fails with "a row is ... from its half-precision rounding" otherwise. `2`: any FP32 data, top-k L2 only (any other join keeps FP32). Each row is stored rounded with its rounding error; the FP32 rows go to side batches and the few pairs still open at block end are settled by reading their FP32 rows. Same neighbours as the FP32 tier; a reported distance is to the half rows, within the rows' rounding error. Both need the one-pass build (no `cluster_column`). |
| `SIRIUS_VSS_HALF_EXACT_DISTANCES` | `0` | With `TIER_HALF=2`: report FP32 distances. |
| `SIRIUS_VSS_HALF_SIDE` | `1` | With `TIER_HALF=2`: `0` keeps the FP32 rows inside the corpus batches instead of side batches (a shared self-join always does). Use if a probe row outside FP16's range is refused. |
| `SIRIUS_VSS_PAIR_ONCE` | `0` | `1`: self-join only. Each pair of rows is compared once instead of twice; the corpus is read about (B+1)/2 times for B blocks. Needs host memory outside the pools for every row's running top-k and bound: rows x (8k + 4) bytes + rows x n_probes x 2 bytes (148 B/row at k 10, 32 probes). Refused (logged, normal schedule) when that passes `SIRIUS_VSS_PAIR_ONCE_HOST_SHARE` of the process's memory limit or the pool plus it passes the limit, with `TIER_HALF=2`, for threshold, cosine or inner product, or when the bounded FP16 search is off. |
| `SIRIUS_VSS_PAIR_ONCE_HOST_SHARE` | `0.25` | Share of the process's memory limit (host RAM, or the cgroup limit if lower) pair-once state may use. |
| `SIRIUS_VSS_SELF_JOIN_SHARE` | `1` | `0`: give each side of a self-join its own scan (the two-table shape; twice the tier bytes). |
| `SIRIUS_VSS_FIRST_MET` | auto | First bound of each probe row from the first cluster run the chunk read meets, so a block reads the corpus once. Auto takes it when the first sweep would read more than half the chunks; `1`/`0` force it. Top-k only (not inner product), probe blocks held on host. |
| `SIRIUS_VSS_READ_AHEAD` | `1` | `0`: read corpus chunks on the task thread. |
| `SIRIUS_VSS_READ_AHEAD_DEPTH` | `3` | Chunk reads queued at once. |
| `SIRIUS_VSS_RESCORE_FROM_HALF` | `1` | With a half tier, re-score from the probe rows' half copies on the device; `0` gathers FP32 probe rows from host, a gather and a copy per chunk. |
| `SIRIUS_VSS_INGEST_SORT` | `0` | `1`: build with the engine's sampled range sort instead of fixed cluster-id ranges (one more pass over the parked corpus). |
| `SIRIUS_VSS_SPILL_DIRECT` | `1` | `0`: the join's own reads of spilled batches go through the page cache. |
| `SIRIUS_VSS_PROBE_BLOCK_ROWS` | planner | Force the probe block size. |
| `SIRIUS_VSS_DEVICE_RATES` | measured | The planner's cost model uses the GPU's measured FP32/FP16/PCIe/HBM rates, measured once per GPU and cached in `~/.cache/sirius/device_rates` (`SIRIUS_VSS_DEVICE_RATES_FILE` to move it). `remeasure` measures again, `reference` uses the RTX A5000 constants the model was fitted on (needed for 4 unit tests on an A100). |
| `SIRIUS_VSS_BUILD_IN_QUERY` | unset | Only affects the SQL rewrite of plain distance queries, not `sirius_knn_join`; our runs set `0`. |

## Sizing (measured unless marked)

MS MARCO v2.1 passages, 113.5M x 1,024-d FP32 (465 GB as rows), 8,192 clusters, k 10, 32 probes,
GH200 (96 GB HBM, 480 GB host), GPU limit 0.563, host pool 80 GB, process inside a 150 GB cgroup:

| Run | Wall | Build | Join | Blocks | Tier peak | Tier read |
|---|---|---|---|---|---|---|
| self-join, `TIER_HALF=1` | 1,478 s | 489 s | 970 s | 11 | 238 GB | 3,045 GB |
| self-join, `TIER_HALF=1`, `PAIR_ONCE=1` | 1,342 s | | | 12 | | 1,804 GB |
| two-table, `TIER_HALF=1` | 1,757 s | 455 s | 974 s (+281 s probe scan) | 11 | 706 GB | 3,039 GB |

- Tier: a self-join parks the corpus once, in its stored precision (~0.5x the FP32 bytes with a half
  tier, ~1.0x without). A two-table join parks the probe side in FP32 as well. With `TIER_HALF=2`
  the corpus side also keeps its FP32 rows (~1.5x FP32 corpus bytes, inferred from the layout, not
  measured at full size). Leave headroom: the two-table run had 113 GB free at its peak on a 700 GB
  cap. Without a half tier the self-join peak was 472 GB.
- Host: pool + DuckDB `memory_limit` + pair-once state (if on) + `TIER_HALF=2`'s contender log
  (not budgeted by the engine) must fit under the process limit.
- Blocks: about one pass over the corpus per block; the block count follows from the GPU limit.
- Quarter scale (28.4M rows) on an A100 40 GB, GPU limit 14.47 GB, host pool 20 GB, 60 GB cgroup:
  self-join `TIER_HALF=1` 577-617 s, 13 blocks; with `PAIR_ONCE=1` 495-528 s.

## Checking the answer

- Pairs = probe rows x k for a top-k join. `self_hits` is below the row count when the table has
  duplicate vectors.
- `round(sum(distance), 3)` is a fingerprint: the same data, clustering and switches give the same
  value. `TIER_HALF=1` and the FP32 tier give the same sum on FP16-exact data. `TIER_HALF=2` moves it
  in the third decimal (distances to half rows).
- Row by row, on a sample (we used 5M rows): run the reference (FP32 tier, switches off) with k and
  k + 1, the candidate with k, write all three to parquet, then count probe rows whose neighbour id
  set differs, leaving out rows whose k-th and (k+1)-th reference distances are equal (ties):

```sql
CREATE TABLE tied AS SELECT left_id FROM (SELECT left_id, distance,
    row_number() OVER (PARTITION BY left_id ORDER BY distance) AS rk FROM 'ref_k11.parquet')
  WHERE rk IN (10, 11) GROUP BY left_id HAVING count(*) = 2 AND min(distance) = max(distance);
CREATE TABLE d AS (SELECT left_id, right_id FROM 'ref.parquet' EXCEPT SELECT left_id, right_id FROM 'test.parquet')
  UNION ALL (SELECT left_id, right_id FROM 'test.parquet' EXCEPT SELECT left_id, right_id FROM 'ref.parquet');
SELECT count(DISTINCT left_id) FROM d WHERE left_id NOT IN (SELECT left_id FROM tied);  -- expect 0
```

## Smoke test on a new machine

Before a large run: the unit gate, then a quarter of the MS MARCO corpus, then a slice of the real
corpus to calibrate tier peak and time.

```sql
CREATE TABLE base_q AS SELECT id, vec FROM base WHERE hash(id) % 4 = 0;  -- 28,378,095 rows
CHECKPOINT;
```

Self-join with `SIRIUS_VSS_TIER_HALF=1`, k 10, 32 probes: 283,780,950 pairs. With our quarter
centroids (`cents_q`, ask for the parquet file) loaded as the clustering, the distance sum is
175,910,598.078 in every `TIER_HALF=1` run we made (GH200 and A100; with and without pair-once;
self-join or two-table over `base_q`). With a clustering of your own only the pair count
is comparable. Our quarter caps: GPU `usage_limit_bytes: 14470348800`, host pool 20 GB, tier 200 GB
(330 GB for two tables), the process in a 60 GB cgroup.

## What to watch during a run

- **Tier drive temperature.** At full size the GH200's tier drive sat at 84 C or more for most of the
  run and fell to about 1 GB/s when hot. Log `/sys/class/nvme/<ctrl>/hwmon*/temp1_input` every few
  seconds and start a run from a cool drive when comparing times.
- **OOM retries.** `OOM at operator <NAME>` and `reschedule (retry n/m)` in the engine log are normal
  under pressure; a count close to the retry limit, or a phase line that stops advancing, is not.
  Run-to-run, the same quarter run had 6 retries once and none the next time.
- **Progress.** With `SIRIUS_VECTOR_JOIN_PHASE_DEBUG=1`: the `probe blocks:` line when the join
  starts, then one `search+fold` line per block.
- **Disk tier full.** The query fails after repeated failed conversions naming the disk tier. Watch
  `du -sb` of the tier directory against `capacity_bytes` and the drive's free space.
- **File handles.** Raise `ulimit -n` to the hard limit before every run; the build parks many range
  pieces as files.

## Known issues

- **Unit gate on a non-reference GPU.** 4 cases assume a planner path (brute force over host lists)
  that the GPU's measured rates may not pick (`test_gpu_execution_kmeans.cpp` lines 5073, 5178,
  5319; `..._exact_per_row.cpp` line 781). A fifth (`test_gpu_execution_kmeans.cpp` line 3346) has
  one probe row whose two nearest centroids differ by 2.5e-6 relative; it fails in some runs and
  passes in others. On an A100 every case passes with the reference rates (349 passed, 2 skipped):
  `SIRIUS_VSS_DEVICE_RATES=reference NVIDIA_TF32_OVERRIDE=0 pixi run build/release/test/cpp/sirius_unittest "[vss],[vector_join]"`.
- **Scan batch size and the build.** With `scan_task_batch_size` large enough for a scan batch to
  hold 3 DuckDB row groups (122,880 rows each, ~0.5 GB at 1,024-d FP32), the full-size build took
  377 s instead of 462 s on a GH200 (single runs), and 7% less at quarter scale on an A100 (x86).
  4 row groups gave no further gain. Larger batches raise a scan task's device peak (~1.0 GB to
  ~3.6 GB at 3 row groups) and brought MERGE_SORT OOM retries in the first wave; the retries are
  safe and cost only time.
- **Approximate threshold joins.** Membership and reported distances come from an FP32 expanded
  form with no exact re-score, so pairs within ~1e-6 of the threshold, and the reported distances
  (by up to ~1e-3), can differ with batching and from run to run. Top-k joins re-score and are
  exact within the probed clusters.
- **Pair-once** is for self-joins only and needs host memory outside the pools: rows x (8k + 4)
  bytes + rows x n_probes x 2 bytes (148 B/row at k 10 and 32 probes). With `TIER_HALF=2` it is not
  taken; the join logs why and runs the usual schedule.

## Tested and not tested

Tested:
- 113.5M x 1,024-d (465 GB FP32), self-join and two-table, top-k L2, `TIER_HALF=1`, pair-once on the
  self-join, on a GH200 96 GB / 480 GB host (aarch64, sm_90a, one NVMe tier drive).
- Quarter scale (28.4M rows) on an A100 40 GB x86 host: self-join and two-table, `TIER_HALF=1` and
  `=2`, pair-once.
- 5M-row samples, row by row against the FP32 tier: `TIER_HALF=1`, `=2` (on data made truly FP32),
  pair-once with forced OOM retries and its fallback, first-met on and off.

Not tested:
- **Any corpus larger than 465 GB.** A 1.5 TB corpus is unverified. Check first:
  - tier capacity: about the corpus's stored size for a self-join, plus the probe side in FP32 for
    two tables, plus the FP32 side batches with `TIER_HALF=2`, plus headroom;
  - cuDF list columns hold 2^31 child values, ~2.09M rows per batch at 1,024-d. The build's cluster
    ranges are sized under `max_sort_partition_bytes` but only roughly even in rows, and there can
    be no more ranges than clusters, so corpus rows / clusters must stay well under that
    (inferred from the code);
  - cluster count and rows per cluster: only 8,192 clusters over 113.5M rows (~13.9k rows each on
    average) have run at size;
  - host memory for per-row state: pair-once at 148 B/row is ~54 GB for 366M rows, which under the
    default share needs a ~216 GB process limit (arithmetic, not run);
  - the build's parked ranges: they take tier space before the join starts, and their count grows
    with the corpus.
- Blackwell (sm_100) has never been built or run.
- `TIER_HALF` (both forms), `PAIR_ONCE` and `FIRST_MET` are top-k L2 only. Threshold joins, cosine
  and inner product run with the FP32 tier and the usual schedule; they are covered by unit tests,
  not at disk-tier scale.
- More than one GPU, more than one tier drive.
