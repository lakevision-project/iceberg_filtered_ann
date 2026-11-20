"""Filtered vector search over Parquet files that carry a per-file IVF index
in their footer (the IFVI v1 format, see lakehouse_ann/format.py).

Three front ends share one per-file kernel (search.py):
  lakehouse_ann.read_ivf / ivf_search   a Daft DataSource           (needs daft)
  lakehouse_ann.iceberg.vector_search   pyiceberg, no query engine  (needs pyiceberg)
  lakehouse_ann.iceberg_sql.ann_sql     SQL, Daft executes          (needs both)
Imports are lazy so each works without the others installed.
"""
from .build import index_table
from .search import DISTANCE_COL, STATS, SearchConfig

__all__ = ["DISTANCE_COL", "STATS", "IvfSource", "IcebergFilesSource", "SearchConfig",
           "index_table", "ivf_search", "read_ivf", "read_ivf_iceberg", "ann_sql"]


def __getattr__(name):
    if name in ("IvfSource", "ivf_search", "read_ivf"):
        from . import source
        return getattr(source, name)
    if name in ("IcebergFilesSource", "read_ivf_iceberg", "ann_sql"):
        from . import iceberg_sql
        return getattr(iceberg_sql, name)
    raise AttributeError(name)
