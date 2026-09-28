"""Build a REAL two-sided vector-join dataset: entity resolution with labelled matches.

Why this and not SIFT/GIST/GloVe: see ../../../design_vecjoin_demo_datasets.md. In one line —
ANN benchmarks give one table plus a query file, so every "join" measured on them is search with
the query set relabelled. Entity resolution gives:

  * TWO real relations (tableA, tableB) with their own schemas
  * SCALAR columns (name, description, price) for the hybrid equality/range + vector predicate,
    instead of the synthetic `id % 10 < 3` that J1/J2 had to use
  * LABELLED ground-truth matches, so we can report PRECISION / RECALL / F1 for the join --
    a real answer, not recall@k against our own exhaustive run
  * a naturally THRESHOLD-shaped query ("all pairs above similarity tau"), which is the mode
    Andy's radius kernel now makes fast (ef56272b)

Encoder: all-MiniLM-L6-v2, 384-dim, L2-normalised output -> cosine distance = 1 - dot, which is
what `metric => 'cosine'` expects. CPU because the tables are thousands of rows, not millions;
this is the semantics/F1 dataset, SIFT stays the scale dataset.
"""
import os, sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

SRC = "/var/tmp/vj/er"
OUT = "/var/tmp/vj/data/parquet"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


def text_of(df):
    """One string per record. Concatenating the descriptive columns is what practitioners do; the
    price is deliberately LEFT OUT of the text so it stays a usable scalar predicate."""
    cols = [c for c in ("name", "title", "description", "manufacturer") if c in df.columns]
    return (df[cols].fillna("").astype(str).agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    for ds in ("Abt-Buy", "Amazon-Google"):
        d = f"{SRC}/{ds}"
        if not os.path.exists(f"{d}/tableA.csv"):
            print(f"skip {ds}: not downloaded"); continue
        a = pd.read_csv(f"{d}/tableA.csv"); b = pd.read_csv(f"{d}/tableB.csv")
        pairs = pd.concat([pd.read_csv(f"{d}/{s}.csv") for s in ("train", "valid", "test")
                           if os.path.exists(f"{d}/{s}.csv")], ignore_index=True)
        print(f"\n{ds}: A={len(a):,} B={len(b):,} labelled pairs={len(pairs):,} "
              f"(positives={int(pairs.label.sum()):,})", flush=True)
        print(f"  A columns: {list(a.columns)}", flush=True)

        for name, df in (("a", a), ("b", b)):
            vec = model.encode(text_of(df).tolist(), batch_size=256,
                               normalize_embeddings=True, show_progress_bar=False)
            vec = np.ascontiguousarray(vec, dtype="float32")
            assert vec.shape[1] == DIM, vec.shape
            tbl = {"id": pa.array(df["id"].to_numpy(np.int64), pa.int64()),
                   "vec": pa.FixedSizeListArray.from_arrays(
                       pa.array(vec.reshape(-1), pa.float32()), DIM)}
            # keep the scalars: they are the point -- hybrid predicates need real attributes
            if "price" in df.columns:
                tbl["price"] = pa.array(pd.to_numeric(df["price"], errors="coerce")
                                        .to_numpy(np.float64), pa.float64())
            if "name" in df.columns:
                tbl["name"] = pa.array(df["name"].fillna("").astype(str).tolist(), pa.string())
            if "manufacturer" in df.columns:
                tbl["manufacturer"] = pa.array(
                    df["manufacturer"].fillna("").astype(str).tolist(), pa.string())
            stem = f"{ds.lower()}_{name}"
            pq.write_table(pa.table(tbl), f"{OUT}/{stem}.parquet")
            print(f"  wrote {stem}.parquet  rows={len(df):,} dim={DIM} "
                  f"cols={list(tbl.keys())}", flush=True)

        pq.write_table(pa.table({
            "left_id": pa.array(pairs["ltable_id"].to_numpy(np.int64), pa.int64()),
            "right_id": pa.array(pairs["rtable_id"].to_numpy(np.int64), pa.int64()),
            "label": pa.array(pairs["label"].to_numpy(np.int8), pa.int8())}),
            f"{OUT}/{ds.lower()}_labels.parquet")
        print(f"  wrote {ds.lower()}_labels.parquet  pairs={len(pairs):,}", flush=True)


if __name__ == "__main__":
    main()
