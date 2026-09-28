# 2026-08-25 — I1: one cuVS call per corpus slice, not one per probe run

**Shipped as `3c6db00b`.** This is the regression evidence for the change; the recall/time
consequences are in [../2026-08-25-post-i1-pareto/](../2026-08-25-post-i1-pareto/).

## The change

The clustered fold called `brute_force_knn` once per **(probe run x corpus slice)**, each call
doing a fresh index build and search. Two prior measurements said that cost is per-*call*, not
per-*pair*:

* **X1** fitted `a = 278 us/call` at **42-96% of runtime**.
* **S1** held the pairs scored constant, raised span-visits **16x**, and paid **7.4x** the time.

Every run wanting a slice searches the *same dataset*, so their probe rows are gathered into one
query matrix and searched once. The output is row-major, so each run's rows are a contiguous block
and the split back is a slice, not a gather. **Calls drop from `clusters x n_probes` to
`clusters`.** A slice wanted by exactly one run is searched in place, so `n_probes=1` pays no
gather and no copy.

`brute_force_knn` also gained a prebuilt-index overload. **That hoist alone was a wash** -- it was
measured separately first and came out 0-8% *slower*, because L2-unexpanded precomputes no norms,
so there was nothing to hoist. The entire win is the search batching. Worth recording: the handoff
listed the build hoist as step (a), "cheapest first", and it was worth nothing.

## Result -- SIFT1M, 10k probes x 1M corpus, k=10, A100, warm, best of three

| clusters / probes | calls before | calls after | before | after | speedup |
|---|---|---|---|---|---|
| 64 / 1 | 64 | 64 | 0.081 s | 0.081 s | 1.00x |
| 64 / 2 | 128 | 64 | 0.121 s | 0.090 s | 1.34x |
| 64 / 4 | 256 | 64 | 0.182 s | 0.112 s | 1.63x |
| 64 / 8 | 512 | 64 | 0.315 s | 0.142 s | 2.22x |
| 64 / 16 | 1024 | 64 | 0.598 s | 0.202 s | 2.96x |
| 256 / 32 | 8192 | 256 | 3.183 s | 0.486 s | **6.55x** |

The speedup tracks the call-count reduction, which is what the cost model predicted.
`search+merge` is still ~95% of the operator's device time; what changed is how many calls that is
spread over. Residual per-call floor is about **26 ms at 64 clusters** -- 16% of runtime at p16,
75% at p1 -- which is why **I1(c), concurrent streams, was not done: it is now a minority term.**

## Correctness -- proven, not asserted

`sum(distance)` is **bit-identical** between the unmodified fold and the batched one at every
configuration, and row counts match. 38-77 pairs per 100,000 differ, and **every one is an exact
distance tie**: for the differing pairs both members' distances were recomputed from the vectors
themselves (`array_distance` over `probe*`/`corpus*`), and the two `(left_id, distance)` multisets
are equal -- `EXCEPT ALL` in both directions returns 0 rows, in all five approximate configs.

| config | pairs differing / 100,000 | tie check (a\b, b\a) |
|---|---|---|
| c64_p1 | 0 (byte-identical) | -- |
| c64_p2 | 38 | 0, 0 |
| c64_p4 | 77 | 0, 0 |
| c64_p8 | 70 | 0, 0 |
| c64_p16 | 68 | 0, 0 |
| c256_p32 | 55 | 0, 0 |

## Two traps, one cycle each -- read before benchmarking this operator

**1. A build tree that "matches HEAD" by mtime is not a build of HEAD.** The first baseline was
taken with an existing `build/` on a clean tree at `642e21c7`, with no source file newer than the
binary. Rebuilding that same commit from scratch produced **different results** -- and I had
already started attributing the difference to my own change. That baseline was discarded.
**Rebuild before baselining.**

**2. `cuvs::cluster::kmeans::fit` is not bit-stable across processes.** Roughly **one session in
six** converges to a different centroid set from the same data, same seed, same binary. The corpus
is *ordered* by the fitting session's labels and the probe side is *assigned from the centroids at
join time*, so a re-fit silently changes which pairs are scored and reads as a result regression.
Diagnosed by hashing the centroids, the neighbour list and the cluster row counts separately: only
the **centroid** hash moved.

The harness fix is to hash the centroids under `SIRIUS_VECTOR_JOIN_PRUNE_DEBUG` (now in the
operator as `[vecjoin] centroids=<hash>`) and diff only runs whose hash agrees. `base_B` and
`i1b2_B` share `f86e27d40fbf1dc9` / `feea0ff6ad63df1f`, which is what makes the table above a valid
comparison. **Better still -- and what the Pareto experiment does -- build the cluster-ordered
corpus inside the measuring session.**

## Files
`run_bench.sh` (one session, 6 configs), `setup.sql`, `prelude.sql`, and per-tag
`results/<tag>/SUMMARY.txt` with the centroid hashes, timings, aggregate fingerprints and the md5
of the full ordered result. The 100k-row dumps themselves were deleted after the tie check -- 52 MB,
and the conclusion they supported is recorded above.

## Reproduce
```bash
./run_bench.sh <tag>
```
Compare two tags only when their `results/<tag>/centroid_hashes.txt` agree.
