# 2026-08-23 — Out-of-core: the corpus does not fit the GPU

**Question.** The coverage matrix called out-of-core "our only unique claim, and it has zero
numbers." What does it actually buy, and what does the alternative cost?

⚠️ **Do not state this as "cuVS/FAISS cannot do it."** That is false and a reviewer will say so:

| who | is corpus > GPU memory a problem? |
|---|---|
| FAISS-CPU, DuckDB HNSW | **No.** They live in host RAM; 48 GiB fits in 120 GB with no special handling. |
| cuVS / FAISS-GPU | Yes by default, but there are at least four standard escapes: manual **sharding**, RMM **managed memory (UVM)**, **IVF-PQ** compression (48 GiB fp32 → ~1 GiB, so it simply fits), and **multi-GPU**. |

So out-of-core is a capability gap **only against GPU libraries**, and even there it is a
*usability and exactness* argument, not an impossibility one. The defensible claim is:

> Corpus > GPU memory, handled in one SQL statement at recall 1.0, where every alternative
> requires choosing and implementing a strategy that costs recall, throughput, or engineering.

`cuvs_strategies.py` measures UVM and IVF-PQ so that claim is backed rather than asserted.
IVF-PQ is the strongest counter and must be reported: it fits easily, and its cost is recall.

**Status: Sirius side COMPLETE. cuVS side PARTIAL (8 GiB only) — the box rebooted mid-run and
took the corpora with it.** See "What is missing" below.

## Environment

A100-SXM4-40GB (**39.5 GiB usable**), 64 cores, 120 GB RAM. Sirius `vecjoin-approx-cluster` @
`ae5e3d0d` built `80-real`, under `../../vecjoin_bench.yaml` (GPU pool 0.85). Corpora synthesized
by `gen_corpus.py` (see caveats). 10k probes, k=10, L2. Exact ground truth computed by GPU brute
force over the full corpus.

## Results — Sirius, exact, corpus host-pinned and streamed to the device

| corpus | vs device | time | recall@10 | rows |
|---|---|---|---|---|
| 8 GiB | fits | 5.53 s | **1.0** | 100,000 |
| 24 GiB | fits | **16.14 s** (`run_24gib.log`) / 19.98 s | **1.0** | 100,000 |
| **48 GiB** | **exceeds 39.5 GiB** | **32.39 s** / 39.09 s | **1.0** | 100,000 |

The 48 GiB row was measured twice, on two separate boots of the same box: **32.39 s**
(`run_48gib.log`, committed) and 39.09 s (first run, log lost to a wipe). ~17% run-to-run
spread, most likely page-cache state on the corpus file. **Quote 39.09 s if you want the
conservative figure**; both passed the recall-1.0 gate. The 8 and 24 GiB rows still have no
committed log — re-run them before those two are quoted externally.

Scaling is near-linear (8→24 GiB: 3.61× for 3× data; 24→48: 1.96× for 2×). **Recall is exactly
1.0 at every size**, which is the correctness gate: anything below 1.0 would mean the streaming
fold silently dropped corpus chunks, and a fast wrong answer is the failure mode this experiment
exists to catch.

## Results — cuVS IVF-Flat, 8 GiB (the only size measured before the reboot)

Timing starts at **host-resident data** for both systems: Sirius's timed statement excludes its
`CREATE TABLE` + `pin_table`, so cuVS's parquet decode is excluded too. What is counted is the
work from host memory onwards.

| stage | time |
|---|---|
| host→device | 1.42 s |
| index build | 4.77 s |
| search | 0.55 s |
| **GPU-side total** | **6.75 s** (recall 1.0) |

## Results — every way cuVS can go past the device limit, 48 GiB (E3, `cuvs_strategies.py`)

| strategy | build | search | total | recall@10 | user effort |
|---|---|---|---|---|---|
| **Sirius exact** | — | — | **32.39 s** | **1.0** | one SQL statement |
| cuVS sharded | 29.25 s (+15.12 h2d) | 2.93 s | 47.29 s | 1.0 | hand-written 6-shard loop |
| cuVS IVF-PQ (pq_dim=32, UVM) | 70.79 s | **0.33 s** | 71.1 s | **0.408** | index params + UVM |
| cuVS UVM IVF-Flat | 98.20 s | 16.49 s | 114.7 s | 1.0 | allocator swap only |
| cuVS direct | — | — | **OOM** | — | — |

**Sirius is the fastest option that returns an answer at all, and the only exact one that is also
the fastest.** Both cuVS escapes work — the claim is emphatically not "cuVS cannot do this" — but
each costs something: sharding costs user code and 1.46× the time, UVM costs 3.5×, and IVF-PQ
costs **recall 0.41 against our 1.0**.

### The sharpest version of the out-of-core argument

**At this size cuVS cannot amortize its index either.** An IVF-Flat index over a 48 GiB corpus is
itself ~48 GiB, so it cannot stay resident between query batches — the build cost recurs. The
*only* cuVS configuration whose index can stay on the device is IVF-PQ, whose compressed index is
~16× smaller — and that is the configuration that drops to recall 0.408.

So out-of-core is the one regime where the amortization argument does not rescue an ANN library,
because residency is exactly what is unavailable. That is a much stronger and more specific claim
than "we are out-of-core and they are not".

⚠️ pq_dim=32 was the only PQ point measured. A larger pq_dim trades compression for recall and
would land somewhere between; a refinement pass over the raw vectors recovers recall at extra
cost. Neither was run, so do not present 0.408 as *the* IVF-PQ number — present it as the cost of
the compression ratio that makes the index resident.

⚠️ The IVF-PQ build (70.79 s) and UVM build (98.20 s) are both inflated by paging the raw fp32
corpus in through managed memory. That is intrinsic to feeding a builder more data than the device
holds, but it is a property of the *approach*, not of cuVS's kernels.

## Results — E2: FAISS-CPU at 48 GiB does NOT fit a 120 GB box

The rival with no device limit. `IndexIVFFlat` was fed by streaming the corpus from parquet so the
raw vectors and the index never coexist in host RAM; an RSS guard aborted rather than letting the
machine die.

| stage | result |
|---|---|
| train (2M sampled rows, 64 threads) | 306.3 s |
| add, streamed | **ABORTED at 88M of 100.7M rows, RSS 100.7 GB** |

RSS by progress: 20M → 25.6 GB · 40M → 48.7 GB · 60M → 65.6 GB · 80M → 94.4 GB · **88M → 100.7 GB**.

**`IndexIVFFlat` carries ~1.9× the raw corpus in RAM.** The index holds 512 B of codes plus an
8 B id per vector — ~46 GB at 88M rows — and the rest is `std::vector` growth slack in the
inverted lists. Extrapolated, the full 100.7M rows need **~115 GB**, leaving nothing for the search
phase on a 120 GB box.

**So the CPU rival has a capacity ceiling too — just a higher one.** That matters for how the
out-of-core claim is worded: it is not "GPU libraries can't and CPU can". FAISS's own answer for
corpora this size is `OnDiskInvertedLists` or PQ compression — escapes that exist and cost
something, exactly like cuVS's sharding, UVM and IVF-PQ.

⚠️ **This is a memory-fit finding, not a speed finding.** No FAISS-CPU *time* exists at 48 GiB
because the index never finished building. A 24 GiB run follows for an actual number. And the
abort threshold is our chosen guard, not a hard failure — a box with more RAM, or
`OnDiskInvertedLists`, would complete.

⚠️ An earlier run aborted at 68M/85.4 GB and was retried after adding
`pa.default_memory_pool().release_unused()` per batch, in case Arrow's retained buffers were
inflating RSS. They were not: 40M rows held 48.7 GB both before and after. The growth is FAISS's.

## Results — E2 completed at 24 GiB, where FAISS-CPU fits

Both systems, one box, recall 1.0, 10k probes, k=10.

| | build | search | **total, one-shot** | peak RSS |
|---|---|---|---|---|
| **Sirius** (host-pinned, streamed) | — | — | **16.14 s** | — |
| FAISS-CPU `IndexIVFFlat`, 64 threads | 1925.9 s (train 362.9 + add 1563.0) | 58.6 s | **1984.5 s** | 57.7 GB |

- **One-shot join: Sirius is 123× faster.** A join cannot amortize a build it must pay every time.
- **Even with the index free**, Sirius 16.14 s beats FAISS's search alone at 58.6 s — **3.6×**.
- **FAISS's build alone (1926 s) is 119× the entire Sirius join.**

So the CPU rival is not competitive at this scale under either accounting, and the earlier
worry — that "GPU speed at host scale" might not be earned — does not survive contact with a
measurement. It is earned by a wide margin.

## The out-of-core picture, complete

| corpus | Sirius | cuVS | FAISS-CPU |
|---|---|---|---|
| 24 GiB | **16.14 s** @ 1.0 | — | 1984.5 s @ 1.0 |
| **48 GiB** | **32.39 s** @ 1.0 | sharded 47.29 s @ 1.0 · UVM 114.7 s @ 1.0 · IVF-PQ 71.1 s @ **0.408** · direct **OOM** | **does not fit** (~115 GB needed) |

**Every alternative has an escape, and each escape costs something** — user code (sharding),
throughput (UVM), recall (IVF-PQ), or memory headroom it does not have (FAISS-CPU). Sirius runs
it in one SQL statement at recall 1.0 and is the fastest option that returns an answer at all.

## The finding: which number you may quote depends on the workload

At 8 GiB, where cuVS still fits comfortably:

- **One-shot join** — index build is not amortizable, so it counts:
  **Sirius 5.53 s vs cuVS 6.75 s. We win, narrowly.**
- **Served index** — build happens once and many query batches follow:
  **cuVS search alone is 0.55 s vs our 5.53 s. cuVS wins by 10×.**

Both are honest; they answer different questions. This is the "build cost belongs in the
comparison" item from the handoff turning out to be decisive rather than bookkeeping. **State
which regime a number belongs to, every time.** Note also that our 5.53 s is a brute-force exact
scan with no index at all, which is why it is competitive against build-inclusive cuVS and not
against build-amortized cuVS.

## ⚠ Caveats that must travel with these numbers

1. **The corpus is synthetic and unusually easy.** 4096 well-separated Gaussian clusters, probes
   drawn from the same centers. cuVS reached recall 1.0 probing only 16 of 4096 lists — real
   embeddings do not behave that way. This data is appropriate for a **capacity and throughput**
   claim and is **not evidence about recall quality**. Do not quote recall from this experiment
   as an ANN quality result.
2. **"Out-of-core" here means the corpus exceeds GPU memory, not host memory.** It is pinned to
   the HOST tier and streamed to the device. Measured 2026-08-23: the **exact** path still
   requires the corpus pinned to some tier and refuses an unpinned child scan (`right table
   'corpus' must be pinned`); only the **approx** path was freed from pinning, in `edecf159`.
   This narrows the claim in `active_vector_join_paper.md` ("neither side needs to fit GPU
   memory — both stream from host"), which reads as broader than what the exact path does.
3. **Only the exact path is measured out-of-core.** Approximate out-of-core needs kmeans over a
   48 GiB corpus, which may itself hit the device limit. Untested.

## What is missing (the box rebooted and wiped `/var/tmp`)

- **cuVS at 24 and 48 GiB** — the sharded path, and the point where `direct` fails outright.
- **cuVS UVM and IVF-PQ** (`cuvs_strategies.py`) — the escapes above. IVF-PQ especially: if it
  reaches useful recall at 48 GiB in under a second, the out-of-core framing needs rewriting
  around exactness rather than capacity.
- **FAISS-CPU at 48 GiB** — the *real* competitor in this regime, since a CPU index has no device
  limit. "GPU speed at host scale" is the claim, and it is untested against the obvious rival.
- The 80 GiB of corpora. They are **not backed up** (they exceed the 64 GB JuiceFS quota) and are
  **deterministic from `--seed 0`** — regenerate with `gen_corpus.py 8`, `24`, `48`. ~30 min.

## Reproduce

```bash
python3 gen_corpus.py 8   --queries 10000      # then 24, 48
bash    sirius_ooc.sh 8 24 48
python3 cuvs_ooc.py   8  --shard-gib 8         # then 24, 48
```

## Traps

1. **The exact path needs the corpus pinned.** An unpinned child scan is refused. Pin to
   `tier => 'host'` for out-of-core; `tier => 'gpu'` cannot hold 48 GiB.
2. **Do not answer a device-capacity question by exhausting the host.** The first version of
   `cuvs_ooc.py`'s `direct` path built a host list of parquet batches and then `np.concatenate`d
   it, holding the corpus in host RAM TWICE — 96 GiB for the 48 GiB case on a 120 GB box. The
   machine went down mid-run. It now streams straight into a preallocated device array, so the
   allocation that fails is the device one, which is the thing being measured.
3. **pyarrow refuses a batch_size of tens of millions** (`OSError: List index overflow`).
   Assemble shards from ordinary batches instead.
4. **Arrow→numpy via `np.stack(col.to_numpy())`** builds one Python object per row and is ~24×
   slower than reading the flat value buffer. At 48 GiB that is the difference between 2 minutes
   and an hour.
5. **Never edit a shell script while it is running.** bash reads a script incrementally by byte
   offset, so an in-place edit shifts what it reads next and it parses garbage mid-run — this
   produced a bogus `syntax error near unexpected token` at line 82 of `sirius_ooc.sh` during
   the 48 GiB run. The measurement had already completed, but it could just as easily have
   corrupted the run. Copy the script, edit the copy, or wait.
6. **Two allocators, one ceiling.** `sirius_ooc.sh` originally let DuckDB default to ~80% of RAM
   for its buffer pool while Sirius separately pinned the whole corpus to the host tier — on a
   48 GiB corpus that reached 98.9 GB RSS against a 120 GiB cgroup ceiling. `memory_limit` and
   `temp_directory` are now set. The lesson from the earlier crash had been applied to
   `bf_anchor.sql` and not to its siblings; out-of-core is exactly where host memory is most
   contended, and it was the one place left uncapped.
7. **cupy's pool retains temporaries between iterations.** A 10k×500k distance tile is 20 GB on
   device; tile the probes as well as the corpus and free the pool inside the inner loop.
