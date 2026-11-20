'''Option A — SQL over an Iceberg-planned, Daft-executed filtered ANN.

This is the third front end; it combines the other two:

  * pyiceberg does the **planning** — everything `lakehouse_ann.iceberg` already
    uses: catalog + snapshot, `scan(row_filter).plan_files()` for partition
    and manifest-statistics pruning, per-file `task.residual`, and
    `task.delete_files`.
  * Daft does the **execution** — SQL parsing, predicate/projection pushdowns,
    task scheduling (Ray when configured), and the global sort + limit merge.

The glue is one `DataSource`. When Daft optimizes a query it pushes the SQL's
WHERE into `pushdowns.filters`; `get_tasks()` converts that into a pyiceberg
`row_filter` — with Daft's own converter from its native Iceberg source
(`daft.io.iceberg._expressions.convert_row_filter`) — plans the files, and
yields one task per planned file carrying that file's residual (as a pyarrow
expression, via `lakehouse_ann.iceberg._arrow_filter`) and deleted positions.

    from pyiceberg.catalog import load_catalog
    from lakehouse_ann.iceberg_sql import ann_sql

    tbl = load_catalog("demo").load_table("vec.docs")
    ann_sql(tbl, "emb", q, k=10, sql="""
        SELECT id, tier, _distance FROM docs
        WHERE tier = 'p10' AND price < 5
        ORDER BY _distance LIMIT 10""").show()

or with the lower-level source, composing Daft operators yourself:

    read_ivf_iceberg(tbl, "emb", q, k=10) \\
        .where(daft.col("tier") == "p10") \\
        .sort("_distance").limit(10)

Semantics (same rules as `lakehouse_ann.source`):

  * A WHERE that sits directly on the source is evaluated DURING ranking.
    One placed after `ORDER BY ... LIMIT` (e.g. in a wrapping subquery)
    cannot be pushed down — that is post-filtering, and at low selectivity
    it returns too few rows.
  * `k` (the per-file top-k, set at registration) must be >= the SQL LIMIT:
    Daft does not push a LIMIT through a Sort, so the source cannot learn
    it from pushdowns. `ann_sql` scans the statement for `LIMIT n` and
    raises k to cover the largest one; `read_ivf_iceberg` callers must keep
    the invariant themselves.
  * A WHERE that references `_distance` itself cannot be evaluated before
    scoring; it is applied after ranking (with a warning). Daft re-applies
    it post-scan, so results stay correct.
  * The SQL LIMIT is never given to pyiceberg planning: planning stops
    after enough *rows* for a limit, but a top-k needs every planned file
    searched.
'''
from __future__ import annotations

import asyncio
import dataclasses
import logging
import re

import numpy as np
import pyarrow as pa

import daft
from daft.io.iceberg._expressions import convert_row_filter
from daft.io.source import DataSource, DataSourceTask
from daft.recordbatch import RecordBatch
from pyiceberg.expressions import AlwaysTrue
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.table import Table

from .iceberg import _arrow_filter, _deleted_positions
from .puffin import index_refs, ref_for
from .search import DISTANCE_COL, SearchConfig, search_file

__all__ = ["IcebergFilesSource", "read_ivf_iceberg", "ann_sql"]

log = logging.getLogger(__name__)

_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)


class IcebergScanTask(DataSourceTask):
    """One planned Iceberg file: residual filter + deletes + IVF search."""

    def __init__(self, io, path, column, query, cfg, out_schema, filter_expr, filter_cols, deleted,
                 index_ref=None):
        self._io, self._path, self._column = io, path, column
        self._index_ref = index_ref
        self._query, self._cfg = query, cfg
        self._out_schema = out_schema
        self._filter, self._filter_cols = filter_expr, filter_cols
        self._deleted = deleted

    @property
    def schema(self):
        return daft.Schema.from_pyarrow_schema(self._out_schema)

    async def read(self):
        def run():
            with self._io.new_input(self._path).open() as src:
                return search_file(src, self._path, self._column, self._query, self._cfg,
                                   self._out_schema, self._filter, self._filter_cols, self._deleted,
                                   index_ref=self._index_ref, index_io=self._io)
        # search_file is synchronous IO-heavy work; to_thread lets the local
        # runner's event loop interleave several files' tasks instead of
        # running each one to completion before the next
        t = await asyncio.to_thread(run)
        yield RecordBatch.from_arrow_table(t)


class IcebergFilesSource(DataSource):
    """A Daft DataSource over an Iceberg table, planned by pyiceberg and
    searched by the per-file IVF kernel. What each side contributes:

        pyiceberg  catalog + snapshot, scan(row_filter).plan_files():
                   partition & manifest-statistics pruning, task.residual,
                   task.delete_files
        Daft       SQL → expressions, predicate/projection pushdowns, task
                   scheduling, the global sort + limit over per-file top-k
        kernel     filter during ranking, ranged vector reads, per-file top-k

    Deleting rows copy-on-write rewrites a file without an index; the kernel
    falls back to an exact scan of that file, so answers stay correct while
    the index is stale (same as `lakehouse_ann.iceberg.vector_search`).
    """

    def __init__(self, table: Table, column: str, query, cfg: SearchConfig,
                 snapshot_id: int | None = None):
        self._table = table
        self._column = column
        self._query = np.asarray(query, dtype=np.float32).reshape(-1)
        self._snapshot_id = snapshot_id
        arrow = schema_to_pyarrow(table.schema(), include_field_ids=False)
        if column not in arrow.names:
            raise KeyError(f"column {column!r} not in {arrow.names}")
        if cfg.coalesce_gap is None:
            remote = "://" in table.metadata.location and not table.metadata.location.startswith("file:")
            cfg = dataclasses.replace(cfg, coalesce_gap=(1024 if remote else 128) * 1024)
        self._cfg = cfg
        self._arrow_schema = arrow.append(pa.field(DISTANCE_COL, pa.float32()))

    @property
    def name(self) -> str:
        return "iceberg-ivf"

    @property
    def schema(self):
        return daft.Schema.from_pyarrow_schema(self._arrow_schema)

    async def get_tasks(self, pushdowns):
        names = self._arrow_schema.names
        filter_cols = sorted(pushdowns.filter_required_column_names()) if pushdowns.filters is not None else []
        use_filter = pushdowns.filters is not None or pushdowns.partition_filters is not None
        if DISTANCE_COL in filter_cols:
            # A predicate on the distance can't be evaluated before scoring;
            # Daft re-applies it after the scan, so results stay correct.
            log.warning("filter references %s; applying it after ranking", DISTANCE_COL)
            filter_cols, use_filter = [], False
        wanted = set(pushdowns.columns or names) | set(filter_cols) | {DISTANCE_COL}
        out_schema = pa.schema([f for f in self._arrow_schema if f.name in wanted])

        # SQL WHERE → pyiceberg row_filter → partition/statistics pruning.
        # convert_row_filter handles both pushdowns.filters and the
        # partition_filters Daft's optimizer may move identity predicates
        # into, and falls back to AlwaysTrue when conversion fails.
        row_filter = AlwaysTrue()
        if use_filter:
            row_filter = convert_row_filter(pushdowns._to_pypushdowns(), self._table.schema())
        planned = self._table.scan(row_filter=row_filter,
                                   snapshot_id=self._snapshot_id).plan_files()

        cfg = self._cfg
        if pushdowns.limit is not None:      # not pushed through a Sort, but be safe
            cfg = dataclasses.replace(cfg, k=max(cfg.k, pushdowns.limit))

        refs = index_refs(self._table, self._column)      # Puffin indexes, if any
        for task in planned:
            expr, cols = _arrow_filter(self._table, task.residual)
            deleted = _deleted_positions(self._table, task)
            yield IcebergScanTask(self._table.io, task.file.file_path, self._column,
                                  self._query, cfg, out_schema, expr, cols, deleted,
                                  index_ref=ref_for(refs, task.file))


def read_ivf_iceberg(table: Table, column: str, query, k: int = 10, *,
                     snapshot_id: int | None = None, nprobe: int = 8,
                     **cfg_overrides) -> daft.DataFrame:
    """A DataFrame over `table`'s current (or `snapshot_id`'s) files: each
    planned file contributes its top-k nearest rows to `query` with a
    `_distance` column. Compose `.where()` / `.sort()` / `.limit()` on it —
    a `.where()` directly on this DataFrame is pushed into every per-file
    task and applied DURING ranking. Keep `k` >= any limit you apply later.
    """
    cfg = SearchConfig(k=k, nprobe=nprobe, **cfg_overrides)
    return IcebergFilesSource(table, column, query, cfg, snapshot_id).read()


def _limit_k(sql: str, k: int) -> int:
    """Per-file k must cover every LIMIT in the statement (Daft does not
    push a limit through the Sort, so the source cannot learn it)."""
    limits = [int(n) for n in _LIMIT_RE.findall(sql)]
    if limits and max(limits) > k:
        log.info("raising per-file k %d -> %d to cover the SQL LIMIT", k, max(limits))
        return max(limits)
    return k


def ann_sql(table: Table, column: str, query, sql: str, k: int = 10, *,
            alias: str | None = None, snapshot_id: int | None = None,
            nprobe: int = 8, bindings: dict | None = None,
            **cfg_overrides) -> daft.DataFrame:
    '''Run `sql` with the ANN source registered as a SQL table.

    The source is bound under `alias` (default: the table's last name
    segment, e.g. "docs" for vec.docs — write `FROM docs`). Other
    DataFrames the SQL references can be passed via `bindings` or left in
    the caller's globals (`daft.sql` picks those up too). Every `LIMIT n`
    in the statement raises the per-file k to at least n.

        ann_sql(tbl, "emb", q, k=10, sql="""
            SELECT d.id, m.label, a._distance
            FROM docs a JOIN meta m ON a.region = m.region
            WHERE a.tier = 'p10' ORDER BY a._distance LIMIT 10""")
    '''
    name = table.name()
    alias = alias or (name.split(".")[-1] if isinstance(name, str) else name[-1])
    k = _limit_k(sql, k)
    df = read_ivf_iceberg(table, column, query, k, snapshot_id=snapshot_id,
                          nprobe=nprobe, **cfg_overrides)
    return daft.sql(sql, **{**(bindings or {}), alias: df})
