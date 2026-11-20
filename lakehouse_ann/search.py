"""Per-file filtered IVF search — the kernel one Daft task runs on one file.

Order of work, cheapest first:

  1. Footer      — already read to open the file; holds the index pointers.
  2. Filter      — read only the filter columns, evaluate the Daft predicate
                   to a row mask. Zero matches ⇒ done, index never touched.
  3. Plan        — two candidate sets, both from metadata alone:
                     exact: every matching row;
                     IVF:   probe clusters nearest-first, drop non-matching ids
                            from each posting list BEFORE scoring, and keep
                            probing until enough clusters with matches are
                            covered (filter during ranking).
                   Price each set's vector reads (bytes + per-request cost) and
                   take exact unless IVF is clearly cheaper.
  4. Fetch       — read only the candidate vectors (byte ranges via the IFVR
                   runs; row groups if the file has none).
  5. Score       — exact L2, keep this file's top-k.
  6. Project     — read the requested columns for those k rows only.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .format import IndexBlob, ValueRuns
from .footer import read_index, read_runs

DISTANCE_COL = "_distance"


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    k: int = 10
    nprobe: int = 8
    # Filtered plans: score every matching row exactly unless the IVF plan's
    # vector reads are this many times cheaper (see _read_cost).
    exact_bias: float = 1.5
    # Ranged reads closer than this are merged into one request. None = pick by
    # filesystem: 128 KiB local (Python per-request overhead), 1 MiB on object
    # storage (one ~30 ms round trip costs about as much as ~1-3 MB of transfer).
    coalesce_gap: int | None = None
    io_threads: int = 16
    # How filter columns are read from a many-row-group file:
    #   "batched" every filter column chunk's byte range comes from the footer;
    #             all of them are fetched as raw ranged reads on a dedicated
    #             pool (filter_io_threads wide), then decoded in ONE
    #             ParquetFile.read over that in-memory cache;
    #   "per_rg"  one ParquetFile.read_row_group per row group on the shared
    #             IO pool (the original path).
    # Both issue one request per chunk — chunks of one column are ~a row
    # group's embedding bytes apart, so merging them would read the file —
    # but raw reads are cheaper per request and the pool is wider.
    filter_read: str = "batched"
    filter_io_threads: int = 64
    # Filtered probing: "matched" counts only clusters holding >= 1 matching
    # row toward nprobe; "all" counts every probed cluster.
    probe_rule: str = "matched"
    # Filtered IVF keeps probing until it holds at least overfetch * k matching
    # candidates (the k nearest MATCHING rows sit farther out than the k nearest
    # rows, so the same nprobe sees too few of them).
    filtered_overfetch: int = 16
    # Count every read the task makes (footer, filter, blobs, vectors,
    # projection). Costs a Python hop per read; for benchmarks.
    trace_io: bool = False
    # Which index to use when a file has one in a Puffin file and/or its footer:
    # "auto" (Puffin, else footer), "puffin" (ignore footers), "footer"
    # (ignore Puffin). For A/B comparisons; results are correct in every mode.
    index_location: str = "auto"


@dataclasses.dataclass
class FileStats:
    path: str
    n_rows: int = 0
    n_match: int | None = None
    plan: str = ""
    clusters_probed: int = 0
    candidates: int = 0
    vector_bytes_read: int = 0
    vector_requests: int = 0
    filter_bytes: int = 0          # compressed size of the filter column chunks read
    index_bytes: int = 0           # IFVI + IFVR blobs
    index_source: str = ""         # "puffin", "footer" or "" (no index)
    index_requests: int = 0        # reads made to fetch the index blobs
    io_bytes: int | None = None    # with trace_io: every byte the task read
    io_requests: int | None = None
    ms: float = 0.0


STATS: list[FileStats] = []          # per-process trace, for benchmarks and demos
_STATS_LOCK = threading.Lock()

# One ranged-read pool per process, shared by every concurrently running task.
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
_TLS = threading.local()            # per-IO-pool-thread ParquetFile (see _pooled_pf)


_FILTER_POOL: ThreadPoolExecutor | None = None


def _io_pool(n: int) -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(n, thread_name_prefix="ivf-io")
        return _POOL


def _filter_pool(n: int) -> ThreadPoolExecutor:
    """Separate from _io_pool: filter reads are many tiny requests whose cost
    is pure round-trip latency, so they want far more outstanding at once — and must
    not queue behind a concurrent task's large vector reads."""
    global _FILTER_POOL
    with _POOL_LOCK:
        if _FILTER_POOL is None:
            _FILTER_POOL = ThreadPoolExecutor(n, thread_name_prefix="ivf-filter")
        return _FILTER_POOL


def _chunk_ranges(md: pq.FileMetaData, columns) -> list[tuple[int, int]]:
    """(offset, length) of every column chunk of `columns` (top-level names,
    nested leaves included), in file order. A chunk starts at its dictionary
    page when it has one, else at its first data page."""
    want = set(columns)
    out = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema.split(".")[0] not in want:
                continue
            starts = [o for o in (c.dictionary_page_offset, c.data_page_offset) if o]
            out.append((min(starts), c.total_compressed_size))
    return sorted(out)


def _merge_ranges(ranges: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Merge sorted (offset, length) ranges whose holes are <= gap bytes —
    several filter columns of one row group usually sit next to each other,
    and pyarrow asks for them as one span."""
    out: list[list[int]] = []
    for off, ln in ranges:
        if out and off - (out[-1][0] + out[-1][1]) <= gap:
            out[-1][1] = max(out[-1][1], off + ln - out[-1][0])
        else:
            out.append([off, ln])
    return [(o, n) for o, n in out]


class _RangeCache:
    """File-like over prefetched byte ranges; anything else falls through to
    the underlying file. Lets pyarrow decode many column chunks in one call
    without issuing a request per chunk itself."""

    def __init__(self, src, ranges: list[tuple[int, int]], blobs: list[bytes]):
        self._src, self._pos, self._size = src, 0, src.size()
        self._starts = np.array([o for o, _ in ranges], dtype=np.int64)
        self._ranges, self._blobs = ranges, blobs
        self.misses = 0
        self.closed = False

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence=0):
        self._pos = pos if whence == 0 else (self._pos + pos if whence == 1 else self._size + pos)
        return self._pos

    def tell(self):
        return self._pos

    def size(self):
        return self._size

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        lo, hi = self._pos, min(self._pos + n, self._size)
        i = max(int(np.searchsorted(self._starts, lo, side="right")) - 1, 0)
        R, parts, at = self._ranges, [], lo
        while at < hi:                      # cached segments; holes come from the file
            while i < len(R) and R[i][0] + R[i][1] <= at:
                i += 1
            if i < len(R) and R[i][0] <= at:
                off, ln = R[i]
                end = min(hi, off + ln)
                parts.append(self._blobs[i][at - off:end - off])
            else:
                end = min(hi, R[i][0]) if i < len(R) else hi
                self.misses += 1
                parts.append(self._src.read_at(end - at, at))
            at = end
        self._pos = hi
        return parts[0] if len(parts) == 1 else b"".join(parts)


def _read_filter_batched(md: pq.FileMetaData, src, cols: list[str], cfg: SearchConfig) -> pa.Table:
    ranges = _merge_ranges(_chunk_ranges(md, cols), 64 << 10)
    blobs = list(_filter_pool(cfg.filter_io_threads).map(lambda r: src.read_at(r[1], r[0]), ranges))
    cache = _RangeCache(src, ranges, blobs)
    return pq.ParquetFile(pa.PythonFile(cache, mode="r"), metadata=md).read(
        columns=cols, use_threads=False)


def _pooled_pf(src):
    """A ParquetFile for the current pool thread, over the task's open `src`.

    pyarrow's `ParquetFile.read_row_group` is NOT safe to call from several
    threads at once, but the ranged reads it issues (`NativeFile.read_at`)
    are — that is also what the IFVR path relies on. So every IO-pool thread
    keeps its own reader over the same open file (one extra footer read per
    thread), and the task's own `pf` stays on the task thread alone."""
    ent = getattr(_TLS, "ent", None)
    if ent is None or ent[0] is not src:
        ent = _TLS.ent = (src, pq.ParquetFile(src))
    return ent[1]


def column_bytes(md: pq.FileMetaData, columns) -> int:
    """Compressed bytes of the named columns' chunks (nested leaves included)."""
    want = set(columns)
    n = 0
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema.split(".")[0] in want:
                n += c.total_compressed_size
    return n


def blob_bytes(md: pq.FileMetaData) -> int:
    import json
    from .format import POINTER_KEY, RUNS_POINTER_KEY
    kv = md.metadata or {}
    return sum(int(json.loads(kv[k.encode()])["length"])
               for k in (POINTER_KEY, RUNS_POINTER_KEY) if k.encode() in kv)


def _filter_mask(pf: pq.ParquetFile, filter_expr, filter_cols: list[str],
                 cfg: SearchConfig | None = None, src=None) -> np.ndarray:
    """Row mask for `filter_expr`: a pyarrow compute Expression (pyiceberg
    path) or a Daft Expression (Daft path). Null counts as no match.

    The filter columns are read one row group at a time so the reads stay
    small; with many-row-group files (small groups are how compressed
    embedding files are written) that is hundreds of requests, so they go
    through the IO pool — each thread with its own reader — instead of
    one-at-a-time."""
    md = pf.metadata
    if cfg is not None and src is not None and md.num_row_groups > 1 and cfg.filter_read == "batched":
        t = _read_filter_batched(md, src, filter_cols, cfg)
    elif cfg is not None and src is not None and md.num_row_groups > 1:
        def read_rg(g):
            return _pooled_pf(src).read_row_group(g, columns=filter_cols, use_threads=False)
        tables = list(_io_pool(cfg.io_threads).map(read_rg, range(md.num_row_groups)))
        t = pa.concat_tables(tables)
    else:
        t = pf.read(columns=filter_cols)
    if isinstance(filter_expr, pc.Expression):
        t = t.append_column("__pos", pa.array(np.arange(t.num_rows, dtype=np.int64)))
        mask = np.zeros(t.num_rows, bool)
        mask[t.filter(filter_expr)["__pos"].to_numpy()] = True
        return mask
    from daft.recordbatch import RecordBatch
    m = RecordBatch.from_arrow_table(t).eval_expression_list([filter_expr.alias("_m")]).to_arrow_table().column(0)
    return m.fill_null(False).to_numpy(zero_copy_only=False).astype(bool)


def _probe(blob: IndexBlob, q: np.ndarray, cfg: SearchConfig,
           mask: np.ndarray | None) -> tuple[np.ndarray, int]:
    diff = blob.centroids - q
    order = np.argsort(np.einsum("ij,ij->i", diff, diff))
    nprobe = min(cfg.nprobe, blob.n_clusters)
    if mask is None:
        parts = [blob.posting(c) for c in order[:nprobe]]
        return np.sort(np.concatenate(parts)), nprobe
    # Same coverage the unfiltered search gets over all rows (≈ nprobe/C),
    # but measured over the MATCHING rows: a filter that is anti-correlated
    # with the query keeps probing outward instead of coming back short.
    target = max(cfg.k * cfg.filtered_overfetch, math.ceil(int(mask.sum()) * nprobe / blob.n_clusters))
    parts, got, probed, useful = [], 0, 0, 0
    for c in order:
        p = blob.posting(c)
        p = p[mask[p]]
        parts.append(p)
        got += len(p)
        probed += 1
        useful += len(p) > 0
        if (useful if cfg.probe_rule == "matched" else probed) >= nprobe and got >= target:
            break
    return np.sort(np.concatenate(parts)), probed


def _rows_by_row_group(md: pq.FileMetaData, rows: np.ndarray):
    starts = np.cumsum([0] + [md.row_group(g).num_rows for g in range(md.num_row_groups)])
    g_of = np.searchsorted(starts, rows, side="right") - 1
    for g in np.unique(g_of):
        sel = g_of == g
        yield int(g), np.flatnonzero(sel), (rows[sel] - starts[g]).astype(np.int64)


def _ranges(offs: np.ndarray, vb: int, gap: int) -> list[np.ndarray]:
    """Split sorted byte offsets into coalesced read groups."""
    return np.split(np.arange(len(offs)), np.flatnonzero(np.diff(offs) > vb + gap) + 1)


def _read_cost(md: pq.FileMetaData, runs: ValueRuns | None, rows: np.ndarray,
               vb: int, gap: int) -> float:
    """Bytes to fetch `rows`' vectors, plus `gap` bytes per request — the same
    trade the coalescer makes: a gap is merged when it is cheaper than a request."""
    if len(rows) == 0:
        return 0.0
    if runs is None:
        groups = list(_rows_by_row_group(md, rows))
        return sum(md.row_group(g).num_rows * vb for g, _, _ in groups) + len(groups) * gap
    offs = runs.offsets_of(rows).astype(np.int64)
    groups = _ranges(offs, vb, gap)
    return float(sum(offs[g[-1]] - offs[g[0]] + vb for g in groups) + len(groups) * gap)


def _vectors_by_ranges(src, runs: ValueRuns, rows: np.ndarray, cfg: SearchConfig) -> tuple[np.ndarray, int]:
    vb = runs.dim * 4
    offs = runs.offsets_of(rows.astype(np.uint64)).astype(np.int64)
    # rows are sorted and runs are laid out in row order, so offs ascend.
    groups = _ranges(offs, vb, cfg.coalesce_gap or 0)
    out = np.empty((len(rows), runs.dim), np.float32)

    def read(idx):
        lo, hi = int(offs[idx[0]]), int(offs[idx[-1]]) + vb
        buf = np.frombuffer(src.read_at(hi - lo, lo), dtype=np.uint8)
        rel = offs[idx] - lo
        out[idx] = buf[rel[:, None] + np.arange(vb)].view("<f4")
        return hi - lo

    if len(groups) == 1:
        nbytes = read(groups[0])
    else:
        nbytes = sum(_io_pool(cfg.io_threads).map(read, groups))
    return out, nbytes, len(groups)


def _vectors_by_row_group(pf: pq.ParquetFile, column: str, rows: np.ndarray,
                          cfg: SearchConfig | None = None,
                          src=None) -> tuple[np.ndarray, int, int]:
    """Candidate vectors via whole row groups — the path for files with no
    IFVR runs (any compressed embedding column). Row groups are fetched
    concurrently through the IO pool (one reader per thread): a probe of
    scattered clusters is dozens of requests, and on object storage each
    request costs a round trip."""
    from .build import embedding_matrix

    def fetch(item):
        g, pos, local = item
        reader = _pooled_pf(src) if src is not None else pf
        m = embedding_matrix(reader.read_row_group(g, columns=[column], use_threads=False).column(0))
        return pos, m[local], m.nbytes

    groups = list(_rows_by_row_group(pf.metadata, rows))
    if cfg is not None and src is not None and len(groups) > 1:
        results = list(_io_pool(cfg.io_threads).map(fetch, groups))
    else:
        results = [fetch(item) for item in groups]
    parts = [r[1] for r in results]
    order = [r[0] for r in results]
    nbytes = sum(r[2] for r in results)
    out = np.empty((len(rows), parts[0].shape[1]), np.float32)
    out[np.concatenate(order)] = np.concatenate(parts)
    return out, nbytes, len(parts)


def _take_rows(pf: pq.ParquetFile, columns: list[str], rows: np.ndarray) -> pa.Table:
    pieces, order = [], []
    for g, pos, local in _rows_by_row_group(pf.metadata, rows):
        pieces.append(pf.read_row_group(g, columns=columns).take(pa.array(local)))
        order.append(pos)
    t = pa.concat_tables(pieces)
    return t.take(pa.array(np.argsort(np.concatenate(order))))


class _CountingReader:
    """File-like wrapper that counts read calls and bytes (see trace_io)."""

    def __init__(self, src):
        self._src, self._pos = src, 0
        self.requests = self.bytes = 0
        self._lock = threading.Lock()
        self.closed = False

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence=0):
        self._pos = pos if whence == 0 else (self._pos + pos if whence == 1 else self._src.size() + pos)
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._src.size() - self._pos
        b = self._src.read_at(n, self._pos)
        self._pos += len(b)
        with self._lock:
            self.requests += 1
            self.bytes += len(b)
        return b

    def size(self):
        return self._src.size()


def search_file(src, path: str, column: str, query: np.ndarray, cfg: SearchConfig,
                out_schema: pa.Schema, filter_expr=None, filter_cols=(),
                deleted: np.ndarray | None = None, index_ref=None, index_io=None) -> pa.Table:
    """Top-k rows of one file, as a table shaped like `out_schema` (which may
    include the embedding column and must end with `_distance`).

    `deleted` holds file positions removed by Iceberg positional deletes or
    deletion vectors. They are the same ordinals the index stores, so they
    simply join the filter mask: a deleted row is never a candidate.

    `index_ref` (a puffin.IndexRef, already validated against this data file)
    points at the file's index in a Puffin file, read through `index_io`.
    Without it, the index comes from the data file's own footer, if any."""
    t0 = time.perf_counter()
    counter = None
    if cfg.trace_io:
        counter = _CountingReader(src)
        src = pa.PythonFile(counter, mode="r")
    pf = pq.ParquetFile(src)
    md = pf.metadata
    st = FileStats(path=path, n_rows=md.num_rows)
    # the counting wrapper is position-based and not pool-safe; pooled reads
    # need the raw NativeFile (and we accept losing per-request counts there)
    pool_src = None if counter is not None else src
    empty = out_schema.empty_table()
    try:
        mask = None
        if filter_expr is not None:
            mask = _filter_mask(pf, filter_expr, list(filter_cols), cfg, pool_src)
            st.filter_bytes = column_bytes(md, filter_cols)
        if deleted is not None and len(deleted):
            if mask is None:
                mask = np.ones(md.num_rows, bool)
            mask[np.asarray(deleted, dtype=np.int64)] = False
        if mask is not None:
            st.n_match = int(mask.sum())
            if st.n_match == 0:
                st.plan = "skip:no-match"
                return empty
            if st.n_match == md.num_rows:
                mask = None          # every row qualifies: the filter is a no-op here

        if index_ref is not None:
            from .puffin import read_index_blobs
            blob, runs, st.index_bytes, st.index_requests = read_index_blobs(index_io, index_ref)
            st.index_source = "puffin"
        else:
            blob = read_index(src, md) if cfg.index_location != "puffin" else None
            runs = read_runs(src, md) if blob is not None else None
            st.index_bytes = blob_bytes(md)
            if blob is not None:
                st.index_source = "footer"
                st.index_requests = 1 + (runs is not None)
        if blob is not None and blob.dim != query.shape[0]:
            raise ValueError(f"{path}: index dim {blob.dim} != query dim {query.shape[0]}")
        if blob is not None and blob.n_rows != md.num_rows:
            blob = runs = None       # index built for different contents: search exactly
        vb = query.shape[0] * 4
        if mask is None:
            if blob is None:
                cand, st.plan = np.arange(md.num_rows, dtype=np.uint64), "exact:no-index"
            else:
                cand, st.clusters_probed = _probe(blob, query, cfg, None)
                st.plan = "ivf"
        else:
            exact = np.flatnonzero(mask).astype(np.uint64)
            cand, st.plan = exact, "exact:filtered"
            if blob is not None and st.n_match > cfg.k:
                # Both candidate sets come from metadata alone, so price each
                # one's vector reads before doing any. Exact is never wrong, so
                # it wins unless IVF is clearly cheaper.
                ivf, probed = _probe(blob, query, cfg, mask)
                c_exact = _read_cost(md, runs, exact, vb, cfg.coalesce_gap or 0)
                c_ivf = _read_cost(md, runs, ivf, vb, cfg.coalesce_gap or 0)
                if c_exact > c_ivf * cfg.exact_bias:
                    cand, st.plan, st.clusters_probed = ivf, "ivf:filtered", probed
        st.candidates = len(cand)
        if len(cand) == 0:
            return empty

        if runs is not None:
            vecs, st.vector_bytes_read, st.vector_requests = _vectors_by_ranges(src, runs, cand, cfg)
        else:
            vecs, st.vector_bytes_read, st.vector_requests = _vectors_by_row_group(pf, column, cand, cfg, pool_src)

        diff = vecs - query
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        k = min(cfg.k, len(cand))
        top = np.argpartition(dist, k - 1)[:k]
        top = top[np.argsort(dist[top])]
        winners = cand[top]

        cols = [n for n in out_schema.names if n not in (DISTANCE_COL, column)]
        order = np.argsort(winners, kind="stable")     # read in file order, then restore
        arrays = {}
        if cols:
            t = _take_rows(pf, cols, winners[order])
            back = np.argsort(order)
            arrays = {n: t.column(n).take(pa.array(back)) for n in cols}
        if column in out_schema.names:
            v = vecs[top]
            arrays[column] = pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), v.shape[1])
        arrays[DISTANCE_COL] = pa.array(dist[top].astype(np.float32))
        return pa.table({n: arrays[n] for n in out_schema.names}).cast(out_schema)
    finally:
        st.ms = (time.perf_counter() - t0) * 1000
        if counter is not None:
            st.io_bytes, st.io_requests = counter.bytes, counter.requests
        with _STATS_LOCK:
            STATS.append(st)
