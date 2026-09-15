"""The practitioner path for the same pipeline: DuckDB filters the probes, exports them; cuVS
brute force (k=65 trimmed, the tuned exact call) finds the matches; ids are imported back into
DuckDB, which does the equi-join and aggregate. Wall time of the whole pipeline, per stage."""
import time, json, numpy as np, duckdb, cupy as cp
from cuvs.neighbors import brute_force
OUT="/var/tmp/vj/compose"; K=10
def sync(): cp.cuda.Device().synchronize()
con=duckdb.connect(f"{OUT}/cuvs.db")
con.execute("SET memory_limit='32GB'")
con.execute("CREATE OR REPLACE TABLE base AS SELECT id, vec::FLOAT[128] AS vec, (id % 97)::INTEGER AS category FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_base.parquet')")
con.execute("CREATE OR REPLACE TABLE query AS SELECT id, vec::FLOAT[128] AS vec, (id % 4 = 0) AS flagged FROM read_parquet('/var/tmp/vj/data/parquet/sift-128-euclidean_query.parquet')")
# the corpus is assumed already resident on the device (amortized) OR exported per run (one-shot)
def run(export_corpus):
    t={}
    t0=time.perf_counter()
    if export_corpus:
        B=np.stack(con.execute("SELECT vec FROM base ORDER BY id").fetchnumpy()["vec"]); Bd=cp.asarray(B); sync()
        t["export_corpus"]=time.perf_counter()-t0
    t0=time.perf_counter()
    q=con.execute("SELECT id, vec FROM query WHERE flagged ORDER BY id").fetchnumpy(); qid=q["id"]; Q=np.stack(q["vec"]); Qd=cp.asarray(Q); sync()
    t["filter+export_probes"]=time.perf_counter()-t0
    t0=time.perf_counter()
    idx=brute_force.build(run.Bd, metric="sqeuclidean"); d,n=brute_force.search(idx, Qd, 65); sync()
    t["search"]=time.perf_counter()-t0
    t0=time.perf_counter()
    nb=cp.asnumpy(n)[:, :K]; rows=np.column_stack([np.repeat(qid, K), nb.reshape(-1)])
    import pyarrow as pa
    m = pa.table({"left_id": pa.array(rows[:, 0].astype(np.int32)), "right_id": pa.array(rows[:, 1].astype(np.int32))})
    con.register("m", m)
    con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM m")
    res=con.execute("SELECT b.category, count(*) AS matches FROM matches m JOIN base b ON b.id = m.right_id GROUP BY b.category ORDER BY matches DESC, b.category").fetchall()
    t["import+join+aggregate"]=time.perf_counter()-t0
    t["total"]=sum(t.values()); return t,res
B=np.stack(con.execute("SELECT vec FROM base ORDER BY id").fetchnumpy()["vec"]); run.Bd=cp.asarray(B); sync()
out={}
for label,exp in (("amortized_corpus_resident",False),("one_shot_export_corpus",True)):
    best=None
    for i in range(3):
        t,res=run(exp)
        if best is None or t["total"]<best["total"]: best=t; best_res=res
    out[label]=best; print(label, {k:round(v,3) for k,v in best.items()})
json.dump(out, open(f"{OUT}/cuvs_compose.json","w"), indent=1)
with open(f"{OUT}/pipeline_cuvs.csv","w") as f:
    f.write("category,matches\n"); [f.write(f"{c},{m}\n") for c,m in best_res]
