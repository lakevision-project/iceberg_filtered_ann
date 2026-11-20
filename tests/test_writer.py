"""Writer tests: the compressed-embedding write path.

`index_table(..., compress_embedding=True)` zstds the embedding column, so
vector bytes cannot be needle-verified: the file must carry an IFVI blob but
NO IFVR runs, and search must fall back to row-group fetches. What these
tests pin down, on a small local file (no catalog, no network):

  1. layout      — embedding chunk is ZSTD, footer has the IFVI pointer and
                   no IFVR pointer (the uncompressed write has both);
  2. blob        — the spliced IFVI parses, with the right shape;
  3. parity      — compressed and uncompressed writes of the SAME table with
                   the same seed produce identical search results: the blob
                   (hence candidate set) is identical, only the vector fetch
                   path differs;
  4. correctness — top-k against a brute-force scan of the source rows.

Run from the repo root:  .venv/bin/pytest -q tests/test_writer.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", message=".*statistics missing.*")

from lakehouse_ann.build import embedding_matrix, index_table          # noqa: E402
from lakehouse_ann.format import POINTER_KEY, RUNS_POINTER_KEY         # noqa: E402
from lakehouse_ann.footer import read_index, read_runs                 # noqa: E402
from lakehouse_ann.search import DISTANCE_COL, SearchConfig, search_file  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SRC = os.path.join(_ROOT, "data", "plain", "part-000.parquet")
COLUMN = "emb"
N_CLUSTERS = 64
ROW_GROUP_SIZE = 1024          # small, so the RG fallback spans several groups
K = 10

fs = pafs.LocalFileSystem()


def _open(path):
    return fs.open_input_file(path)          # random access: parquet needs seek


def _leaf(md: pq.FileMetaData, column: str) -> int:
    rg = md.row_group(0)
    for j in range(rg.num_columns):
        p = rg.column(j).path_in_schema
        if p == column or p.startswith(column + "."):
            return j
    raise KeyError(column)


@pytest.fixture(scope="module")
def src_table():
    if not os.path.exists(_SRC):
        pytest.skip("local demo data missing (data/plain)")
    return pq.read_table(_SRC)


@pytest.fixture(scope="module")
def writes(src_table, tmp_path_factory):
    """The same table written both ways, same seed."""
    d = tmp_path_factory.mktemp("writes")
    plain = str(d / "plain.parquet")
    zstd = str(d / "zstd.parquet")
    index_table(src_table, plain, COLUMN, N_CLUSTERS,
                row_group_size=ROW_GROUP_SIZE, seed=7)
    index_table(src_table, zstd, COLUMN, N_CLUSTERS,
                row_group_size=ROW_GROUP_SIZE, seed=7, compress_embedding=True)
    return plain, zstd


@pytest.fixture(scope="module")
def q(src_table):
    rng = np.random.default_rng(123)
    m = embedding_matrix(src_table.column(COLUMN))
    return m[rng.integers(len(m))] + rng.standard_normal(m.shape[1]).astype(np.float32) * 0.3


def _search(path, query, **cfg):
    out_schema = pa.schema([src_field for src_field in pq.read_schema(path).remove_metadata()]
                           + [pa.field(DISTANCE_COL, pa.float32())])
    with _open(path) as src:
        t = search_file(src, path, COLUMN, query,
                        SearchConfig(k=K, nprobe=8, **cfg), out_schema)
    return t


def test_layout(writes):
    plain, zstd = writes
    md_p, md_z = pq.ParquetFile(plain).metadata, pq.ParquetFile(zstd).metadata
    kv_p, kv_z = md_p.metadata or {}, md_z.metadata or {}
    assert POINTER_KEY.encode() in kv_z, "compressed write must still splice the IFVI blob"
    assert RUNS_POINTER_KEY.encode() not in kv_z, "compressed bytes cannot be verified; no IFVR"
    assert POINTER_KEY.encode() in kv_p and RUNS_POINTER_KEY.encode() in kv_p
    # the embedding leaf really is compressed now, and nothing else changed shape
    assert md_z.row_group(0).column(_leaf(md_z, COLUMN)).compression == "ZSTD"
    assert md_p.row_group(0).column(_leaf(md_p, COLUMN)).compression == "UNCOMPRESSED"
    assert md_z.num_rows == md_p.num_rows == pq.ParquetFile(_SRC).metadata.num_rows
    assert md_z.num_row_groups > 3          # several small groups: the fallback path


def test_blob_parses(writes):
    _, zstd = writes
    with _open(zstd) as src:
        md = pq.ParquetFile(src).metadata
        blob = read_index(src, md)
        assert read_runs(src, md) is None    # no IFVR: readers use row groups
    assert blob is not None
    assert blob.dim == 128 and blob.n_clusters == N_CLUSTERS
    assert blob.n_rows == pq.ParquetFile(_SRC).metadata.num_rows
    assert blob.file_geom is not None


def test_parity_compressed_vs_plain(writes, q):
    """Same blob + same vectors ⇒ identical results; only the fetch path
    (ranged reads vs row groups) differs."""
    plain, zstd = writes
    tp, tz = _search(plain, q), _search(zstd, q)
    assert tp.num_rows == tz.num_rows == K
    assert tp.column("id").to_pylist() == tz.column("id").to_pylist()
    np.testing.assert_allclose(tp.column(DISTANCE_COL).to_pylist(),
                               tz.column(DISTANCE_COL).to_pylist(), rtol=1e-6)


def test_recall_vs_bruteforce(writes, q, src_table):
    _, zstd = writes
    m = embedding_matrix(src_table.column(COLUMN))
    top = np.argsort(np.linalg.norm(m - q, axis=1), kind="stable")[:K]
    got = _search(zstd, q).column("id").to_pylist()
    want = src_table.column("id").take(pa.array(top)).to_pylist()
    recall = len(set(got) & set(want)) / K
    assert recall >= 0.8, f"recall@{K} = {recall}: {got} vs {want}"


@pytest.mark.parametrize("cols,expr", [
    (["tier"], "tier == 'p10'"),
    (["price", "tier"], "(price < 5) & (tier == 'p10')"),
])
def test_batched_filter_read_matches_per_row_group(writes, q, cols, expr):
    """filter_read="batched" (ranged prefetch + one decode) must give the same
    row mask as the per-row-group path, never fall back to the file, and the
    search results built on it must be identical."""
    import pyarrow.compute as pc
    from lakehouse_ann import search as S
    _, zstd = writes
    e = eval(expr, {}, {c: pc.field(c) for c in cols})
    with _open(zstd) as src:
        pf = pq.ParquetFile(src)
        md = pf.metadata
        assert md.num_row_groups > 1
        chunks = S._chunk_ranges(md, cols)
        assert len(chunks) == md.num_row_groups * len(cols)
        ranges = S._merge_ranges(chunks, 64 << 10)
        # several filter columns of one row group merge into one read; the
        # embedding chunk between row groups keeps the groups apart
        assert len(ranges) == md.num_row_groups
        blobs = [src.read_at(n, o) for o, n in ranges]
        cache = S._RangeCache(src, ranges, blobs)
        t = pq.ParquetFile(pa.PythonFile(cache, mode="r"), metadata=md).read(columns=cols)
        assert cache.misses == 0
        assert t.equals(pf.read(columns=cols))
        m_b = S._filter_mask(pf, e, cols, SearchConfig(filter_read="batched"), src)
        m_r = S._filter_mask(pf, e, cols, SearchConfig(filter_read="per_rg"), src)
    assert m_b.any() and np.array_equal(m_b, m_r)
    with _open(zstd) as src:
        out_schema = pa.schema(list(pq.read_schema(zstd).remove_metadata())
                               + [pa.field(DISTANCE_COL, pa.float32())])
        a, b = (search_file(src, zstd, COLUMN, q, SearchConfig(k=K, nprobe=8, filter_read=fr),
                            out_schema, e, cols) for fr in ("batched", "per_rg"))
    assert a.equals(b)
