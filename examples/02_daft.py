"""Mode 2 — Daft DataFrame: the index is a Daft DataSource.

A .where() placed directly on the source is pushed into every per-file task
and applied DURING ranking. One placed after .sort().limit() is post-filtering
and returns too few rows when the filter is selective (shown below).
"""
import daft
from daft import col
from _demo import catalog, query_vector

from lakehouse_ann.iceberg_sql import read_ivf_iceberg

tbl = catalog().load_table("vec.docs")
q = query_vector(tbl)

during = (read_ivf_iceberg(tbl, "emb", q, k=10)
          .where(col("tier") == "p1")                      # pushed down: filter during ranking
          .sort("_distance").limit(10)
          .select("id", "tier", "_distance"))
print("filter during ranking:", during.count_rows(), "rows")
during.show()

after = (read_ivf_iceberg(tbl, "emb", q, k=10)
         .sort("_distance").limit(10)
         .where(col("tier") == "p1"))                       # NOT pushed: post-filter
print("post-filter (what an index blind to filters gives you):", after.count_rows(), "rows")

# The same source also works without a catalog, over a glob of indexed Parquet:
#   from lakehouse_ann import ivf_search
#   ivf_search("data/indexed/*.parquet", "emb", q, k=10, where=col("tier") == "p1")
