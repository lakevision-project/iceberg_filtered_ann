"""Daft DataSource for filtered vector search over footer-indexed Parquet.

    from daft import col
    from lakehouse_ann import ivf_search

    ivf_search("data/*.parquet", "emb", q, k=10,
               where=col("category") == "tech",
               select=["id", "title"]).show()

or with the lower-level source, composing Daft operators yourself:

    read_ivf("data/*.parquet", "emb", q, k=10) \
        .where(col("category") == "tech") \
        .sort("_distance").limit(10)

A `.where()` directly on the source is pushed into every per-file task and
applied DURING ranking. One placed after `.sort().limit()` cannot be pushed
down — that is post-filtering, and at low selectivity it returns too few rows.
"""
from __future__ import annotations

import dataclasses
import glob as _glob
import logging

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import daft
from daft.io.source import DataSource, DataSourceTask
from daft.recordbatch import RecordBatch

from .footer import resolve
from .search import DISTANCE_COL, SearchConfig, search_file

log = logging.getLogger(__name__)


def _expand(paths, filesystem):
    if isinstance(paths, str):
        paths = [paths]
    fs, _ = resolve(paths[0], filesystem)
    out = []
    for p in paths:
        _, local = resolve(p, filesystem)
        if any(ch in local for ch in "*?["):
            if isinstance(fs, pafs.LocalFileSystem):
                out += sorted(_glob.glob(local))
            else:
                base = local.split("*")[0].rsplit("/", 1)[0]
                import fnmatch
                infos = fs.get_file_info(pafs.FileSelector(base, recursive=True))
                out += sorted(i.path for i in infos if fnmatch.fnmatch(i.path, local))
        else:
            out.append(local)
    if not out:
        raise FileNotFoundError(f"no files match {paths}")
    return fs, out


class IvfScanTask(DataSourceTask):
    def __init__(self, fs, path, column, query, cfg, out_schema, filter_expr, filter_cols):
        self._fs, self._path, self._column = fs, path, column
        self._query, self._cfg = query, cfg
        self._out_schema = out_schema
        self._filter, self._filter_cols = filter_expr, filter_cols

    @property
    def schema(self):
        return daft.Schema.from_pyarrow_schema(self._out_schema)

    async def read(self):
        with self._fs.open_input_file(self._path) as src:
            t = search_file(src, self._path, self._column, self._query, self._cfg,
                            self._out_schema, self._filter, self._filter_cols)
        yield RecordBatch.from_arrow_table(t)


class IvfSource(DataSource):
    """Each task searches one file and returns its top-k with `_distance`.
    The caller sorts by `_distance` and limits to k for the global answer."""

    def __init__(self, paths, column: str, query, cfg: SearchConfig, filesystem=None):
        self._fs, self._paths = _expand(paths, filesystem)
        self._column = column
        self._query = np.asarray(query, dtype=np.float32).reshape(-1)
        if cfg.coalesce_gap is None:
            local = isinstance(self._fs, pafs.LocalFileSystem)
            cfg = dataclasses.replace(cfg, coalesce_gap=(128 if local else 1024) * 1024)
        self._cfg = cfg
        with self._fs.open_input_file(self._paths[0]) as f:
            file_schema = pq.read_schema(f).remove_metadata()
        if column not in file_schema.names:
            raise KeyError(f"column {column!r} not in {file_schema.names}")
        self._arrow_schema = file_schema.append(pa.field(DISTANCE_COL, pa.float32()))

    @property
    def name(self) -> str:
        return "ivf"

    @property
    def schema(self):
        return daft.Schema.from_pyarrow_schema(self._arrow_schema)

    async def get_tasks(self, pushdowns):
        names = self._arrow_schema.names
        filter_expr, filter_cols = pushdowns.filters, sorted(pushdowns.filter_required_column_names())
        if filter_expr is not None and DISTANCE_COL in filter_cols:
            # A predicate on the distance itself can't be evaluated before
            # scoring; Daft re-applies it after the scan, so results stay
            # correct, only the per-file pre-filter is lost.
            log.warning("filter references %s; applying it after ranking", DISTANCE_COL)
            filter_expr, filter_cols = None, []
        wanted = set(pushdowns.columns or names) | set(filter_cols) | {DISTANCE_COL}
        out_schema = pa.schema([self._arrow_schema.field(n) for n in names if n in wanted])
        for path in self._paths:
            yield IvfScanTask(self._fs, path, self._column, self._query, self._cfg,
                              out_schema, filter_expr, filter_cols)


def read_ivf(paths, column: str, query, k: int = 10, *, nprobe: int = 8,
             filesystem=None, **cfg_overrides) -> daft.DataFrame:
    cfg = SearchConfig(k=k, nprobe=nprobe, **cfg_overrides)
    return IvfSource(paths, column, query, cfg, filesystem).read()


def ivf_search(paths, column: str, query, k: int = 10, *, where=None, select=None,
               nprobe: int = 8, filesystem=None,
               **cfg_overrides) -> daft.DataFrame:
    df = read_ivf(paths, column, query, k, nprobe=nprobe,
                  filesystem=filesystem, **cfg_overrides)
    if where is not None:
        df = df.where(where)
    if select is not None:
        df = df.select(*select, DISTANCE_COL)
    return df.sort(DISTANCE_COL).limit(k)
