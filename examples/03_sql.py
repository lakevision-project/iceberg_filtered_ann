"""Mode 3 — SQL: Daft parses and executes, pyiceberg plans.

The ANN source is bound as a SQL table (named after the table's last name
segment). The WHERE on it is pushed into every per-file task; joins and
aggregations over the ANN result work in the same statement.
"""
import daft
from _demo import catalog, query_vector

from lakehouse_ann import ann_sql

cat = catalog()
tbl = cat.load_table("vec.docs_by_region")
q = query_vector(tbl)

ann_sql(tbl, "emb", q, k=10, sql="""
    SELECT id, region, tier, _distance FROM docs_by_region
    WHERE region = 'r3' AND price < 50
    ORDER BY _distance LIMIT 10""").show()

labels = daft.from_pydict({"tier": ["p50", "p10", "p1", "p01"],
                           "label": ["common", "frequent", "rare", "very rare"]})
ann_sql(tbl, "emb", q, k=20, bindings={"labels": labels}, sql="""
    SELECT l.label, COUNT(*) AS n, MIN(a._distance) AS best
    FROM (SELECT * FROM docs_by_region WHERE region = 'r3'
          ORDER BY _distance LIMIT 20) a
    JOIN labels l ON a.tier = l.tier
    GROUP BY l.label ORDER BY best""").show()

# time travel: the same query against the table's first snapshot
first = tbl.metadata.snapshots[0].snapshot_id
ann_sql(tbl, "emb", q, k=10, snapshot_id=first, sql="""
    SELECT id, _distance FROM docs_by_region WHERE region = 'r3'
    ORDER BY _distance LIMIT 3""").show()
