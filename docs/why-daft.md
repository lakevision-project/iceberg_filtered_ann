# Why Daft

lakehouse-ann needs three things from a query engine:

1. **The query's filter must reach the per-file search, before ranking.** That is
   the whole point of filtering during ranking (see [design §3](design.md#3-why-filter-during-ranking)).
   The engine has to hand the WHERE clause, as an expression, to a custom data
   source, which then runs it inside each file's IVF probe.
2. **The per-file kernel is Python** (numpy + pyarrow): k-means, posting lists,
   ranged reads. The engine has to call it without a foreign-language bridge on
   every batch.
3. **The same code should run on a laptop and on a cluster**, and be reachable
   from SQL as well as from a DataFrame API.

## What Daft provides

* **A Python `DataSource` with pushdowns.** `DataSource.get_tasks(pushdowns)`
  receives the optimized query's filter (`pushdowns.filters`), partition
  filters and projected columns as Daft expressions. Our source
  (`lakehouse_ann/iceberg_sql.py`) turns the filter into a pyiceberg
  `row_filter`, lets Iceberg plan and prune, and emits one task per surviving
  file carrying that file's residual predicate. That is requirement 1, with no engine changes.
* **SQL for free.** `daft.sql` plans over the same DataFrame, so
  `SELECT … WHERE … ORDER BY _distance LIMIT k` reaches the same pushdowns, and
  joins and aggregations over the ANN result work in the same statement.
* **Local and distributed from one code path.** The same source runs on Daft's
  native runner and on its Ray runner without modification. Tasks are pickled
  to Ray workers, which run the kernel next to the data and return only each
  file's top-k. We measured identical results on both runners and 1.8–4.2×
  speedups on queries that touch many files ([benchmarks §5](benchmarks.md#5-native-vs-ray-runner-5m-rows)).
* **Arrow-native, Rust-executed.** Record batches cross between Daft and the
  kernel as Arrow, and Daft's own operators (sort, limit, joins) run in Rust.
* **Iceberg awareness.** We reuse Daft's own expression-to-pyiceberg converter
  (`daft.io.iceberg._expressions.convert_row_filter`), so SQL predicates and
  pyiceberg planning agree.
* **Model inference in the write path.** Daft class UDFs load an embedding
  model once per worker, batch the input, and place workers on GPUs on Ray.
  The embed stage is about 30 lines (`lakehouse_ann/embed.py`).

## Alternatives we considered

This is our assessment for this project's needs, not a general ranking of
engines.

* **pyiceberg alone.** It is an excellent planner and reader, but not a query
  engine: no SQL, no joins, no distributed execution. We kept it as front end (i),
  `vector_search`, which needs no engine at all.
* **Apache Spark.** Spark has a mature Iceberg integration and DataSource V2
  filter pushdown, but custom sources are JVM code. A Python kernel would sit
  behind Arrow UDF hops, and a cluster is heavyweight for single-node use.
* **DuckDB.** Fast and embeddable, with Iceberg support, but single-node. A
  data source that receives pushed-down filters is a C++ extension rather than
  Python.
* **Ray Data alone.** Distributed and Python-native, but without SQL, and we'd
  have built the planning and filter-pushdown plumbing ourselves. With Daft we
  get Ray's scheduling through Daft's Ray runner, and Daft's planner on top.

Nothing in the index format depends on Daft. The footer blobs are documented
([design §1](design.md#1-an-indexed-parquet-file)) and every file stays plain
Parquet, so another engine could implement the same per-file kernel behind its
own data source API. [Other engines](other-engines.md) describes what that takes.
