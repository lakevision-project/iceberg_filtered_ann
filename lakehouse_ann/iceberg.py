"""Filtered vector search on an Iceberg table with nothing but pyiceberg
(+ pyarrow + numpy). No query engine.

    from pyiceberg.catalog import load_catalog
    from lakehouse_ann.iceberg import vector_search

    tbl = load_catalog("default").load_table("vec.docs")
    vector_search(tbl, "emb", q, k=10, row_filter="region = 'r3'",
                  selected_fields=("id", "region"))

What pyiceberg contributes, and what this module adds on top:

  pyiceberg  plan_files(row_filter)  partition + manifest-statistics pruning;
             task.residual           what is left of the filter for THIS file
                                     (AlwaysTrue for a file whose partition
                                     already proves the filter — no filter
                                     column is read, the file is "pure");
             task.delete_files       positional deletes / deletion vectors;
             snapshot_id             time travel;
             add_files()             register indexed Parquet files as-is.
  here       per-file IVF search (search.py) on each planned file, in a
             thread pool, then a k-way merge on `_distance`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.expressions import AlwaysTrue, BooleanExpression
from pyiceberg.exceptions import NoSuchNamespaceError
from pyiceberg.expressions.parser import parse
from pyiceberg.expressions.visitors import bind, extract_field_ids
from pyiceberg.io.pyarrow import _read_deletes, expression_to_pyarrow
from pyiceberg.manifest import DataFileContent
from pyiceberg.table import Table

from .search import DISTANCE_COL, STATS, SearchConfig, search_file

__all__ = ["vector_search", "create_table_from_files", "STATS"]


def _arrow_filter(tbl: Table, residual: BooleanExpression):
    """Residual → (pyarrow expression, column names), or (None, []) if no-op."""
    if isinstance(residual, AlwaysTrue):
        return None, []
    schema = tbl.schema()
    bound = bind(schema, residual, case_sensitive=True)    # residuals come back unbound
    cols = sorted(schema.find_column_name(i).split(".")[0] for i in extract_field_ids(bound))
    return expression_to_pyarrow(bound, schema), cols


def _deleted_positions(tbl: Table, task) -> np.ndarray | None:
    out = []
    for df in task.delete_files:
        if df.content == DataFileContent.EQUALITY_DELETES:
            # Same limitation as pyiceberg's own reader.
            raise NotImplementedError(f"equality deletes are not supported: {df.file_path}")
        pos = _read_deletes(tbl.io, df).get(task.file.file_path)
        if pos is not None:
            out.append(np.asarray(pos, dtype=np.int64))
    return np.unique(np.concatenate(out)) if out else None


def vector_search(tbl: Table, column: str, query, k: int = 10, *,
                  row_filter: str | BooleanExpression = AlwaysTrue(),
                  selected_fields: tuple[str, ...] | None = None,
                  snapshot_id: int | None = None, nprobe: int = 8,
                  max_workers: int = 16, **cfg_overrides) -> pa.Table:
    """Top-k rows of `tbl` nearest to `query` (L2) among rows matching
    `row_filter`, with a `_distance` column, nearest first.

    Every planned file is searched; files without an index (e.g. ones
    pyiceberg rewrote on delete) fall back to an exact scan of that file,
    so the answer stays correct while the index is stale.
    """
    if isinstance(row_filter, str):
        row_filter = parse(row_filter)
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    is_remote = "://" in tbl.metadata.location and not tbl.metadata.location.startswith("file:")
    cfg = SearchConfig(k=k, nprobe=nprobe, coalesce_gap=(1024 if is_remote else 128) * 1024,
                       **cfg_overrides)

    scan = tbl.scan(row_filter=row_filter, snapshot_id=snapshot_id)
    tasks = list(scan.plan_files())
    if not tasks:
        return pa.table({})

    from .puffin import index_refs, ref_for
    refs = index_refs(tbl, column)         # Puffin indexes, if any

    def one(task):
        path = task.file.file_path
        expr, cols = _arrow_filter(tbl, task.residual)
        deleted = _deleted_positions(tbl, task)
        with tbl.io.new_input(path).open() as src:
            file_schema = pq.read_schema(src).remove_metadata()
            names = selected_fields or tuple(n for n in file_schema.names)
            keep = [file_schema.field(n) for n in file_schema.names if n in names]
            out_schema = pa.schema(keep + [pa.field(DISTANCE_COL, pa.float32())])
            return search_file(src, path, column, query, cfg, out_schema, expr, cols, deleted,
                               index_ref=ref_for(refs, task.file), index_io=tbl.io)

    with ThreadPoolExecutor(max_workers) as ex:
        parts = [t for t in ex.map(one, tasks) if t.num_rows]
    if not parts:
        return pa.table({})
    merged = pa.concat_tables(parts, promote_options="permissive")
    order = np.argsort(merged[DISTANCE_COL].to_numpy(), kind="stable")[:k]
    return merged.take(pa.array(order))


def create_table_from_files(catalog, identifier: str, paths: list[str], *,
                            partition_by: list[str] = (), properties: dict | None = None) -> Table:
    """Create an Iceberg table whose schema comes from the indexed Parquet
    files, partitioned by identity on `partition_by`, and register the files
    with add_files() — no rewrite, so the footer indexes survive. The parent
    namespace is created if it does not exist yet. For an identity partition,
    every file must hold a single value of that column (pyiceberg infers it
    from the file's min/max statistics)."""
    arrow_schema = pq.read_schema(paths[0]).remove_metadata()
    try:
        tbl = catalog.create_table(identifier, schema=arrow_schema, properties=properties or {})
    except NoSuchNamespaceError:
        parts = identifier.split(".") if isinstance(identifier, str) else list(identifier)
        if not parts[:-1]:
            raise
        catalog.create_namespace(tuple(parts[:-1]))
        tbl = catalog.create_table(identifier, schema=arrow_schema, properties=properties or {})
    if partition_by:
        with tbl.update_spec() as spec:
            for name in partition_by:
                spec.add_identity(name)
    tbl.add_files(list(paths))
    return tbl
