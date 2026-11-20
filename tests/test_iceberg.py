import os
import sys
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyiceberg.catalog.sql import SqlCatalog  # noqa: E402

from lakehouse_ann import index_table  # noqa: E402
from lakehouse_ann.build import embedding_matrix  # noqa: E402
from lakehouse_ann.iceberg import STATS, create_table_from_files, vector_search  # noqa: E402
from lakehouse_ann.search import SearchConfig, search_file  # noqa: E402

from test_lakehouse_ann import DIM, make_table, truth  # noqa: E402

warnings.filterwarnings("ignore", message=".*statistics missing.*")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("ice")
    cat = SqlCatalog("t", uri=f"sqlite:///{d}/cat.db", warehouse=f"file://{d}/wh")
    cat.create_namespace("vec")
    full = pa.concat_tables([make_table(4000, seed=i) for i in range(3)])
    mixed = []
    for i in range(3):
        p = str(d / f"mixed{i}.parquet")
        index_table(full.slice(i * 4000, 4000), p, "emb", 24, row_group_size=1024)
        mixed.append(p)
    pure = []
    for c in ["a", "b", "c", "d"]:
        p = str(d / f"cat_{c}.parquet")
        index_table(full.filter(pc.equal(full["cat"], c)), p, "emb", 16, row_group_size=1024)
        pure.append(p)
    return (cat, full, create_table_from_files(cat, "vec.mixed", mixed),
            create_table_from_files(cat, "vec.pure", pure, partition_by=["cat"]))


def test_filtered_search_matches_truth(tables):
    _, full, mixed, _ = tables
    q = embedding_matrix(full["emb"])[11] + 0.1
    out = vector_search(mixed, "emb", q, 10, row_filter="cat = 'c'", selected_fields=("id", "cat"))
    assert set(out["cat"].to_pylist()) == {"c"} and out.num_rows == 10
    assert out["_distance"].to_pylist() == sorted(out["_distance"].to_pylist())
    assert len(set(out["id"].to_pylist()) & set(truth(full, q, 10, "c"))) >= 8


def test_partition_pruning_makes_the_filter_vanish(tables):
    _, full, _, pure = tables
    q = embedding_matrix(full["emb"])[3]
    n0 = len(STATS)
    out = vector_search(pure, "emb", q, 10, row_filter="cat = 'd'", selected_fields=("id", "cat"))
    st = STATS[n0:]
    assert len(st) == 1 and st[0].plan == "ivf" and st[0].filter_bytes == 0
    assert set(out["cat"].to_pylist()) == {"d"}


def test_copy_on_write_delete_falls_back_to_exact(tables):
    cat, full, _, pure = tables
    before = pure.current_snapshot().snapshot_id
    pure.delete("cat = 'b' and id < 1000000")      # rewrites the cat=b file, which loses its index
    pure = cat.load_table("vec.pure")
    q = embedding_matrix(full["emb"])[5]
    n0 = len(STATS)
    out = vector_search(pure, "emb", q, 10, row_filter="cat = 'b'", selected_fields=("id",))
    assert [s.plan for s in STATS[n0:]] == ["exact:no-index"]
    assert all(i >= 1_000_000 for i in out["id"].to_pylist())
    keep = full.filter(pc.field("id") >= 1_000_000)
    assert out["id"].to_pylist() == truth(keep, q, 10, "b")
    old = vector_search(pure, "emb", q, 10, row_filter="cat = 'b'", snapshot_id=before)
    assert old["id"].to_pylist() == truth(full, q, 10, "b")


def test_positional_deletes_are_never_candidates(tmp_path):
    t = make_table(3000)
    p = str(tmp_path / "f.parquet")
    index_table(t, p, "emb", 16, row_group_size=512)
    import pyarrow.parquet as pq
    back = pq.read_table(p)
    vecs = embedding_matrix(back["emb"])
    q = vecs[100]
    schema = pa.schema([back.schema.field("id"), pa.field("_distance", pa.float32())])
    with pa.OSFile(p) as src:
        first = search_file(src, p, "emb", q, SearchConfig(k=5, nprobe=16), schema)
    pos = np.flatnonzero(np.isin(back["id"].to_numpy(), first["id"].to_numpy()))
    with pa.OSFile(p) as src:
        second = search_file(src, p, "emb", q, SearchConfig(k=5, nprobe=16), schema, deleted=pos)
    assert not set(first["id"].to_pylist()) & set(second["id"].to_pylist())
    assert second.num_rows == 5
