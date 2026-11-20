# Querying: four front ends, one kernel

Every front end ends in the same per-file kernel (`lakehouse_ann/search.py`).
They differ in who plans the files and who executes the tasks.

| | planning | execution | interface |
|---|---|---|---|
| (i) `lakehouse_ann.iceberg.vector_search` | pyiceberg | local thread pool | function |
| (ii) `lakehouse_ann.read_ivf_iceberg` / `read_ivf` | pyiceberg / file glob | Daft | DataFrame |
| (iii) `lakehouse_ann.ann_sql` | pyiceberg | Daft | SQL |
| (iv) any of (ii)/(iii) after `daft.set_runner_ray()` | pyiceberg | Daft on Ray | DataFrame / SQL |

All of them return the requested columns plus `_distance` (L2), smallest first.

## (i) pyiceberg only

```python
from lakehouse_ann.iceberg import vector_search
top = vector_search(tbl, "embedding", q, k=10,
                    row_filter="category = 'tech' and customer_id < 200",
                    selected_fields=("id", "title"), nprobe=32, snapshot_id=None)
```

pyiceberg is a planner and a reader, not a query engine, so this is a function,
not SQL. `plan_files(row_filter)` prunes by partition and manifest statistics, each
planned file is searched in a thread pool with its own residual filter and deletes,
and the results are k-way merged. It needs no Daft and is useful when embedding
the search in an application.

## (ii) Daft DataFrame

```python
from daft import col
from lakehouse_ann import read_ivf_iceberg

df = read_ivf_iceberg(tbl, "embedding", q, k=10, nprobe=32)
df.where(col("category") == "tech").sort("_distance").limit(10).show()
```

`read_ivf_iceberg` returns a DataFrame over a custom Daft `DataSource`. When
Daft optimizes the query, it pushes the `.where()` into `pushdowns.filters`.
The source turns that into a pyiceberg `row_filter`, plans the files, and emits
one task per file. Each task returns its file's top-k, and Daft does the global
`sort().limit()`.

Without a catalog, the same kernel runs over a glob of indexed Parquet files:

```python
from lakehouse_ann import ivf_search
ivf_search("s3://bucket/table/data/*.parquet", "embedding", q, k=10,
           where=col("category") == "tech", select=["id", "title"])
```

The glob front end has no partition pruning, no deletes and no snapshots. It is
a convenient way to search indexed files that aren't (yet) in a table.

## (iii) SQL

```python
from lakehouse_ann import ann_sql
ann_sql(tbl, "embedding", q, k=10, sql="""
    SELECT d.id, m.label, d._distance
    FROM docs d JOIN meta m ON d.region = m.region
    WHERE d.category = 'tech'
    ORDER BY d._distance LIMIT 10""", bindings={"meta": meta_df})
```

The ANN source is bound as a SQL table named after the table's last name
segment (`vec.docs` → `docs`; override with `alias=`). Other DataFrames for
joins go in `bindings=` or the caller's globals. Joins and aggregations over
the ANN result work in the same statement, and the WHERE on the source is still
pushed into the kernel.

## (iv) Distributed with Ray

```python
import daft
daft.set_runner_ray(address="ray://head:10001")   # or None for a local cluster
```

Nothing else changes. Each planned file becomes a task on a Ray worker. Tasks
carry the file's residual filter, deleted positions and the table's FileIO, so
workers need no catalog access or credentials of their own. Only each file's
top-k rows travel back for the merge. Results are identical to the native
runner (verified in [benchmarks §5](benchmarks.md#5-native-vs-ray-runner-5m-rows)).
Ray helps once a query touches many files. With few files left after pruning,
the scheduling overhead outweighs the gain.

## Semantics

1. **A WHERE directly on the source is evaluated during ranking.** A WHERE
   after `ORDER BY … LIMIT` (in a wrapping subquery, or a `.where()` after
   `.limit()`) can't be pushed down. That is post-filtering, and it returns too
   few rows when the filter is selective.
2. **`k` must be ≥ the SQL `LIMIT`.** `k` is the per-file top-k. Daft doesn't
   push a limit through a sort, so the source can't learn it. `ann_sql` scans
   the statement for `LIMIT n` and raises `k` to the largest. If you use
   `read_ivf_iceberg` directly, keep that invariant yourself.
3. **A WHERE on `_distance`** can't be evaluated before scoring. It is applied
   after ranking (with a warning); Daft re-applies it after the scan, so results
   stay correct.
4. **The `LIMIT` never reaches pyiceberg planning.** Planning stops once it has
   enough *rows* for a limit, but a top-k needs every planned file searched.
5. **Deletes.** Positional deletes and deletion vectors are masked out. A
   copy-on-write rewrite has no index and is scanned exactly. Equality deletes
   raise an error.
6. **Time travel.** `snapshot_id=` on every Iceberg front end.

## Knobs (`SearchConfig`, passed as keyword arguments)

| knob | default | meaning |
|---|---|---|
| `k` | 10 | per-file top-k (≥ LIMIT) |
| `nprobe` | 8 | clusters probed per file (filtered: clusters *with matches*) |
| `filtered_overfetch` | 16 | filtered IVF keeps probing until `overfetch·k` matching candidates |
| `exact_bias` | 1.5 | prefer the exact plan unless IVF reads are this many times cheaper |
| `coalesce_gap` | auto | merge ranged reads closer than this (128 KiB local, 1 MiB object storage) |
| `io_threads` | 16 | ranged-read pool for vectors |
| `filter_read` | `"batched"` | filter-column reads: batched ranged prefetch or `"per_rg"` |
| `filter_io_threads` | 64 | pool for the batched filter reads |
| `trace_io` | False | count every read (bytes, requests) per file, for benchmarks |

Per-file statistics (`lakehouse_ann.search.STATS`) record the plan, candidates,
bytes and requests of every file searched in the current process.
