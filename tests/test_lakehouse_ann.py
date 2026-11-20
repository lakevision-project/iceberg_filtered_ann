import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from daft import col  # noqa: E402

from lakehouse_ann import STATS, index_table, ivf_search, read_ivf  # noqa: E402
from lakehouse_ann.build import build_blob, embedding_matrix  # noqa: E402
from lakehouse_ann.format import POINTER_KEY, FormatError, parse, serialize  # noqa: E402
from lakehouse_ann.footer import read_index, read_runs  # noqa: E402

DIM = 16


def make_table(n, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((20, DIM)).astype(np.float32) * 4
    topic = rng.integers(0, 20, n)
    v = centers[topic] + rng.standard_normal((n, DIM)).astype(np.float32)
    return pa.table({
        "id": np.arange(n, dtype=np.int64) + seed * 1_000_000,
        "cat": rng.choice(["a", "b", "c", "d"], n, p=[0.6, 0.3, 0.09, 0.01]),
        "emb": pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), DIM),
    })


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("ivf")
    tables = []
    for i in range(3):
        t = make_table(5000, seed=i)
        index_table(t, str(d / f"f{i}.parquet"), "emb", 32, row_group_size=1024)
        tables.append(t)
    return str(d / "*.parquet"), pa.concat_tables(tables)


def truth(table, q, k, cat=None):
    v = embedding_matrix(table["emb"])
    ids = table["id"].to_numpy()
    m = np.ones(len(ids), bool) if cat is None else np.array(table["cat"].to_pylist()) == cat
    d = np.linalg.norm(v[m] - q, axis=1)
    return ids[m][np.argsort(d)[:k]].tolist()


def test_blob_roundtrip_and_crc():
    v = np.random.default_rng(1).standard_normal((300, DIM)).astype(np.float32)
    b = build_blob(v, 8)
    raw = serialize(b)
    b2 = parse(raw)
    assert b2.n_clusters == 8 and b2.n_rows == 300
    assert np.array_equal(b2.postings, b.postings) and np.array_equal(b2.offsets, b.offsets)
    assert sorted(b2.postings.tolist()) == list(range(300))
    with pytest.raises(FormatError):
        parse(raw[:-1] + bytes([raw[-1] ^ 1]))


def test_indexed_file_is_plain_parquet(files):
    pattern, table = files
    path = pattern.replace("*", "f0")
    back = pq.read_table(path)
    assert back.num_rows == 5000
    # Rows are re-ordered by cluster, but the multiset of rows is unchanged.
    assert sorted(back["id"].to_pylist()) == sorted(make_table(5000, 0)["id"].to_pylist())
    md = pq.read_metadata(path)
    assert POINTER_KEY.encode() in md.metadata
    json.loads(md.metadata[POINTER_KEY.encode()])


def test_value_runs_point_at_the_vectors(files):
    pattern, _ = files
    path = pattern.replace("*", "f1")
    md = pq.read_metadata(path)
    with pa.OSFile(path) as src:
        runs = read_runs(src, md)
        blob = read_index(src, md)
    raw = open(path, "rb").read()
    vecs = embedding_matrix(pq.read_table(path)["emb"])
    offs = runs.offsets_of(np.arange(len(vecs), dtype=np.uint64))
    for r in range(len(vecs)):
        assert np.array_equal(np.frombuffer(raw[int(offs[r]):int(offs[r]) + DIM * 4], "<f4"), vecs[r])
    # cluster-sorted: every posting list is one contiguous run of rows
    for c in range(blob.n_clusters):
        p = blob.posting(c)
        assert len(p) == 0 or np.array_equal(p, np.arange(p[0], p[0] + len(p)))



def test_unfiltered_matches_exact_with_full_probe(files):
    pattern, table = files
    q = embedding_matrix(table["emb"])[123] + 0.1
    got = ivf_search(pattern, "emb", q, 10, nprobe=32, select=["id"]).to_pydict()["id"]
    assert got == truth(table, q, 10)


@pytest.mark.parametrize("cat", ["a", "c", "d"])
def test_filtered_returns_only_matching_rows_with_high_recall(files, cat):
    pattern, table = files
    q = embedding_matrix(table["emb"])[7] + 0.1
    out = ivf_search(pattern, "emb", q, 10, where=col("cat") == cat,
                     select=["id", "cat"]).to_pydict()
    assert len(out["id"]) == 10
    assert set(out["cat"]) == {cat}
    assert out["_distance"] == sorted(out["_distance"])
    assert len(set(out["id"]) & set(truth(table, q, 10, cat))) >= 8


def test_filter_is_pushed_into_tasks(files):
    pattern, table = files
    q = embedding_matrix(table["emb"])[0]
    n0 = len(STATS)
    ivf_search(pattern, "emb", q, 5, where=col("cat") == "d").collect()
    traced = STATS[n0:]
    assert len(traced) == 3 and all(s.n_match is not None for s in traced)


def test_post_filter_underfills(files):
    """A where() after sort/limit is not pushed down: it filters the top-k."""
    pattern, table = files
    q = embedding_matrix(table["emb"])[0]
    out = read_ivf(pattern, "emb", q, 10).sort("_distance").limit(10) \
        .where(col("cat") == "d").to_pydict()
    assert len(out["id"]) < 10


def test_no_match_skips_index(files):
    pattern, table = files
    q = embedding_matrix(table["emb"])[0]
    n0 = len(STATS)
    out = ivf_search(pattern, "emb", q, 5, where=col("cat") == "zzz").to_pydict()
    assert out["id"] == []
    assert {s.plan for s in STATS[n0:]} == {"skip:no-match"}
