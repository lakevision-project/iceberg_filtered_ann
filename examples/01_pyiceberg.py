"""Mode 1 — pyiceberg only: no query engine.

pyiceberg plans the scan (partition + statistics pruning, per-file residual
filters, delete files, snapshots); the per-file IVF kernel runs on each
surviving file in a thread pool, and the results are k-way merged.
"""
from _demo import catalog, query_vector

from lakehouse_ann.iceberg import STATS, vector_search

tbl = catalog().load_table("vec.docs_by_region")        # partitioned identity(region)
q = query_vector(tbl)

STATS.clear()
top = vector_search(tbl, "emb", q, k=10, row_filter="region = 'r3' and tier = 'p10'",
                    selected_fields=("id", "region", "tier"))
print(top.to_pandas())
print(f"files searched: {len(STATS)} (others pruned by the partition), "
      f"plans: {sorted({s.plan for s in STATS})}")
