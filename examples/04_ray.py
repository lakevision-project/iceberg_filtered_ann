"""Mode 4 — distributed: the same SQL on Daft's Ray runner.

Nothing in the query changes. Each planned file becomes a task on a Ray
worker; each worker returns only its file's top-k, and Daft merges them.

    python examples/04_ray.py                      # starts a local Ray cluster
    python examples/04_ray.py ray://head:10001     # or an existing one
    RAY_ADDRESS=auto python examples/04_ray.py     # from a node of the cluster
"""
import sys

import daft
from _demo import catalog, query_vector

daft.set_runner_ray(address=sys.argv[1] if len(sys.argv) > 1 else None)

from lakehouse_ann import ann_sql  # noqa: E402

tbl = catalog().load_table("vec.docs")
q = query_vector(tbl)
ann_sql(tbl, "emb", q, k=10, sql="""
    SELECT id, tier, region, _distance FROM docs
    WHERE tier = 'p10' ORDER BY _distance LIMIT 10""").show()

import ray  # noqa: E402
print("ran on", len(ray.nodes()), "Ray node(s)")
