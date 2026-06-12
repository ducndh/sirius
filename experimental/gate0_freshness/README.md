# Gate-0: freshness-under-churn harness

Make-or-break experiment for the "freshness for GPU-resident data" paper
direction (builds on sirius#819, which supplies the merge-on-read baseline
design — see the issue and PR #823 for the Phase-1 helper).

**Claim under test:** keeping a GPU-resident table fresh under a sustained
CPU-side INSERT/DELETE stream requires a cost-model-driven policy — no fixed
strategy is good across churn regimes.

**PASS:** there exists a realistic churn rate where merge-on-read (arm_b) read
latency degrades monotonically with accumulated delta AND invalidate-and-
reupload (arm_a) is worse.
**KILL:** a trivial fixed policy (e.g. arm_c with any K) flattens all curves
at negligible cost → the problem is too easy to carry a paper.

## Arms

| arm | policy | freshness | who pays |
|---|---|---|---|
| `arm_a` | CHECKPOINT + engine restart + re-warm per write batch | always fresh | full re-upload per batch |
| `arm_b` | stale GPU snapshot + CPU delta query + harness merge | fresh (merged) | every read, growing with delta |
| `arm_c` | arm_b + fold every K ticks | fresh (merged) | reads + periodic pause |
| `cpu`   | plain DuckDB, no GPU | always fresh | every read (CPU speed) |

`arm_b` simulates sirius#819's MVCC delta provider at the harness level with
explicit `delta_ins`/`delta_del` tables (the writer double-books each batch).
The GPU result is the *stale pin-epoch snapshot*; the harness runs the same
query over the delta tables on CPU and merges (decomposable aggregates only:
SUM/COUNT, AVG derived — same restriction a real merge operator would start
with). Every merged result is verified against CPU truth (`verify_every`).

In IVM terms: arm_a = full recompute, arm_b = deferred/lazy maintenance,
arm_c = lazy + periodic reorganization, patch-in-place (eager) is future work
once a real delta-apply kernel exists.

## Phases

```bash
# from the repo root, after building (see below)
cd experimental/gate0_freshness
python3 gate0_driver.py --config config.a100.json --phase setup    # ~minutes, once
python3 gate0_driver.py --config config.a100.json --phase probe    # current-dev semantics
python3 gate0_driver.py --config config.a100.json --phase sweep    # full grid (hours)
# single cells:
python3 gate0_driver.py --config config.a100.json --phase arm_b --rho 0.01 --ticks 10
python3 plot_gate0.py results/sweep.jsonl                          # figures
```

`probe` documents what current dev actually does after writes (stale serve /
fresh / error) at the three stages: un-checkpointed writes, after CHECKPOINT
without restart, after restart. Run it first and read the verdicts — they are
finding #1 of the paper and they pin down what the `table_gpu` cache keys on.

## Data

Synthetic 8-column table (`churn`), default 20M rows (~1.2 GB), generated
deterministically via `hash(range)` — no external datasets needed (the box's
benchmark data is on ephemeral storage and gets wiped). An `incoming`
reservoir (60% of N) feeds inserts. Churn shapes: `fifo` (sliding window —
clustered deletes + appends, matches append-heavy workloads and the NeurIPS'23
streaming-runbook structure) and `uniform` (scattered deletes — adversarial
for granular invalidation; the dirty-granule fraction follows
`1-(1-1/G)^U`). Scale via `scale.n_rows`.

## Output

JSONL, one record per measurement. Every record carries
`(box, interconnect, n_rows, churn_mode, arm, rho, tick)`; `granules` records
carry dirty-rowgroup/vector counts (id-based approximation, documented in the
driver) so the same data later feeds the general cost model with granule size
and interconnect as parameters — do not strip these fields.

## Building (per box)

```bash
git submodule update --init --recursive       # once per fresh worktree
ln -s ~/.pixi-shared .pixi                    # dnguyen56 box: share the pixi env
# build dir on local FS, not network FS:
mkdir -p /tmp/sirius-build-gate0 && ln -s /tmp/sirius-build-gate0 build
# single-arch build (~7x faster CUDA compile): A100=80-real, GH200/H100=90a-real
~/.pixi/bin/pixi run sh -c 'cd duckdb && cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=80-real && cd .. && CMAKE_BUILD_PARALLEL_LEVEL=24 make release'
```

### GH200 port checklist

1. Clone fork, `git checkout gate0-freshness-harness`, submodules, pixi
   (install pixi >= 0.68; `ln -sf ~/.pixi/bin/pixi ~/.local/bin/pixi` so env
   activation finds it).
2. Build with `-DCMAKE_CUDA_ARCHITECTURES=90a-real`.
3. Use `config.gh200.json` (already labels records `interconnect:
   nvlink-c2c`). Check `data_dir` points at fast local storage.
4. Run the same `setup -> probe -> sweep`. The cross-box comparison of the
   SAME jsonl schema is the interconnect-parameter evidence for the cost
   model (arm_a re-upload and arm_b delta-transfer terms should compress by
   the C2C bandwidth ratio; arm_b's merge growth and the dirty-granule curves
   should NOT move — that split is the prediction to check).

## Known limitations (deliberate, Gate-0 scope)

- merge-on-read is simulated at the harness level (explicit delta tables +
  Python merge), not inside the engine. Fine for Gate-0: we are measuring the
  *cost structure* (stale-GPU query + delta query scaling with delta size),
  not engine implementation quality. Engine-level = #819 Phase 2+.
- Engine restart is the cache-invalidation mechanism for arm_a/arm_c (the
  cache is in-process and has no SQL-level drop). `restart_s` is logged
  separately so the re-upload cost can be isolated from process startup.
- Decomposable aggregates only; no UPDATE stream yet (#819 scopes it out too;
  add as delete+insert later).
- Single GPU, single writer session.
