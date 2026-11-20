"""Write-side script: build a per-file IVF index and embed it in the footer.

    python -m lakehouse_ann.build IN.parquet [IN2.parquet ...] --out-dir DIR \
        --column emb --n-clusters 64

Each input file is rewritten (same rows) with the blob attached. By default the
rows are reordered by IVF cluster, a performance optimization that makes each
posting list contiguous; --keep-row-order skips it, and the result is still
correct. The index stores file-local row ordinals of the file as written.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .format import IndexBlob
from .footer import write_indexed_parquet


def embedding_matrix(arr: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """FixedSizeList<float32>/List<float32> column -> (n, dim) float32, no nulls."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if arr.null_count:
        raise ValueError("embedding column has nulls; not supported in this prototype")
    if pa.types.is_fixed_size_list(arr.type):
        dim = arr.type.list_size
        flat = arr.flatten()
    else:
        flat = arr.flatten()
        dim = len(flat) // max(len(arr), 1)
    return flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(len(arr), dim)


def _sqdist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (np.einsum("ij,ij->i", x, x)[:, None] - 2.0 * x @ c.T
            + np.einsum("ij,ij->i", c, c)[None, :])


def assign(x: np.ndarray, centroids: np.ndarray, chunk: int = 65536) -> np.ndarray:
    return np.concatenate([np.argmin(_sqdist(x[i:i + chunk], centroids), axis=1)
                           for i in range(0, len(x), chunk)]) if len(x) else np.empty(0, np.int64)


def kmeans(x: np.ndarray, k: int, *, iters: int = 20, sample: int = 65536, seed: int = 42) -> np.ndarray:
    """Plain Lloyd's k-means on a sample, k-means++ initialisation."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(x)))
    s = x[rng.choice(len(x), min(sample, len(x)), replace=False)]
    cent = [s[rng.integers(len(s))]]
    d2 = ((s - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(s[rng.choice(len(s), p=p)])
        d2 = np.minimum(d2, ((s - cent[-1]) ** 2).sum(1))
    cent = np.stack(cent).astype(np.float32)
    for _ in range(iters):
        a = assign(s, cent)
        sums = np.zeros_like(cent)
        np.add.at(sums, a, s)
        counts = np.bincount(a, minlength=k)
        empty = counts == 0
        cent[~empty] = sums[~empty] / counts[~empty, None]
        # Re-seed empty clusters with random sample points rather than leaving dead centroids.
        if empty.any():
            cent[empty] = s[rng.choice(len(s), empty.sum(), replace=False)]
    return cent


def build_blob(vectors: np.ndarray, n_clusters: int, *, seed: int = 42) -> IndexBlob:
    cent = kmeans(vectors, n_clusters, seed=seed)
    a = assign(vectors, cent)
    order = np.argsort(a, kind="stable")          # ascending row ids within each cluster
    counts = np.bincount(a, minlength=len(cent))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint64)
    mean = vectors.mean(0).astype(np.float32)
    radius = float(np.sqrt(((vectors - mean) ** 2).sum(1).max()))
    return IndexBlob(dim=vectors.shape[1], n_rows=len(vectors), centroids=cent,
                     offsets=offsets, postings=order.astype(np.uint64),
                     file_geom=(mean, radius), seed=seed)


def index_table(table: pa.Table, out_path: str, column: str, n_clusters: int,
                *, row_group_size: int = 8192, seed: int = 42, cluster_rows: bool = True,
                compress_embedding: bool = False) -> int:
    """Index `table` and write it to `out_path`.

    `cluster_rows` (default) also SORTS the rows by cluster before writing, so
    each posting list is one contiguous run of rows — and of vector bytes. A
    query then fetches about `nprobe` byte ranges per file instead of
    thousands of scattered vectors. Row order inside a data file carries no
    meaning in Iceberg (the file is new, so no position delete refers to it),
    but it does mean the output is not row-for-row the input.

    `compress_embedding` writes the embedding column zstd instead of
    uncompressed PLAIN. Byte offsets can then not be verified, so the file
    carries no IFVR runs and readers fetch vectors by row group — keep the
    row group size small enough that `nprobe` clusters span only a few.
    """
    vecs = embedding_matrix(table.column(column))
    blob = build_blob(vecs, n_clusters, seed=seed)
    if cluster_rows:
        perm = blob.postings.astype(np.int64)        # rows grouped by cluster
        table = table.take(pa.array(perm))
        vecs = vecs[perm]
        blob.postings = np.arange(len(perm), dtype=np.uint64)
    return write_indexed_parquet(table, out_path, column, blob, vecs,
                                 row_group_size=row_group_size,
                                 compress_embedding=compress_embedding)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--n-clusters", type=int, default=0,
                    help="clusters per file (default: ~sqrt(rows))")
    ap.add_argument("--row-group-size", type=int, default=8192)
    ap.add_argument("--keep-row-order", action="store_true",
                    help="do not sort rows by cluster (scattered posting lists)")
    ap.add_argument("--compress-embedding", action="store_true",
                    help="zstd the embedding column (no IFVR runs; readers "
                         "fetch vectors by row group)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.inputs:
        t0 = time.perf_counter()
        table = pq.read_table(path)
        k = args.n_clusters or max(1, int(np.sqrt(table.num_rows)))
        out = os.path.join(args.out_dir, os.path.basename(path))
        n = index_table(table, out, args.column, k, row_group_size=args.row_group_size,
                        cluster_rows=not args.keep_row_order,
                        compress_embedding=args.compress_embedding)
        print(f"{out}: {table.num_rows} rows, {k} clusters, {n / 1e6:.1f} MB, "
              f"{time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
