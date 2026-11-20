# lakehouse-ann — filtered vector search on an Iceberg lakehouse

Find the *k* nearest neighbours of a query vector **among the rows that match a
SQL predicate**, directly over an Apache Iceberg table of Parquet files on object
storage. There's no vector database and no second copy of the data.

```sql
SELECT id, title, _distance FROM docs
WHERE tenant = 'acme' AND lang = 'en'
ORDER BY _distance LIMIT 10
```

This repository is an open implementation of the approach in

> **Filtered Vector Search in a Disaggregated Lakehouse: Composing Table-Format
> Pruning with Per-File ANN.** Rakesh Jain, Thomas Griffin, Syed Zawad.
> arXiv:2608.05441, 2026. <https://arxiv.org/abs/2608.05441>

built on open engines: [pyiceberg](https://py.iceberg.apache.org/),
[Daft](https://www.getdaft.io) and [Ray](https://www.ray.io).

## The idea in five points

1. **A small IVF index for every data file.** By default it is stored in an
   Iceberg [Puffin](https://iceberg.apache.org/puffin-spec/) file next to the
   data, one blob per data file (the same pattern v3 deletion vectors use). The
   data files stay untouched plain Parquet, so an index can be **attached to an
   existing table written by any engine** without rewriting data. Optionally, the index can
   instead be spliced into each Parquet file, outside its footer. Either way an
   index-aware reader fetches it with one ranged read, and every other reader
   ignores it.
2. **Iceberg prunes first.** Partition and column-statistics pruning drop whole
   files before any vector is touched. As the paper puts it, the payoff comes
   from file pruning.
3. **Filter during ranking, not after.** Inside each surviving file, rows that
   fail the predicate are removed from the IVF candidate lists *before*
   scoring. Top-k-then-filter (post-filtering) silently returns too few rows
   when the filter is selective. This doesn't.
4. **Layout makes it fast.** Partition by the columns you filter on. A file
   whose partition proves the predicate needs no filter at all. Optionally,
   sort each file's rows by cluster (the writer does this by default). It isn't
   needed for correct results, but a probe then reads a few contiguous byte
   ranges instead of most of the file.
5. **It is still plain Iceberg.** Snapshots, time travel and deletes keep
   working. A file without an index (for example, one rewritten by a delete) is
   scanned exactly, so answers stay correct while the index catches up.

Details are in [docs/design.md](docs/design.md).

## Four ways to query

All four share one per-file search kernel (`lakehouse_ann/search.py`).

```python
from pyiceberg.catalog import load_catalog
tbl = load_catalog("prod").load_table("vec.docs")      # any pyiceberg catalog

# (i) pyiceberg only, no query engine. Iceberg plans, a thread pool searches.
from lakehouse_ann.iceberg import vector_search
vector_search(tbl, "embedding", q, k=10, row_filter="category = 'tech'",
              selected_fields=("id", "title"))

# (ii) Daft DataFrame. The index is a Daft DataSource, and .where() is pushed into every file.
from daft import col
from lakehouse_ann import read_ivf_iceberg
read_ivf_iceberg(tbl, "embedding", q, k=10) \
    .where(col("category") == "tech").sort("_distance").limit(10).show()

# (iii) SQL. Daft parses and executes, pyiceberg plans.
from lakehouse_ann import ann_sql
ann_sql(tbl, "embedding", q, k=10, sql="""
    SELECT id, title, _distance FROM docs
    WHERE category = 'tech' AND customer_id < 200
    ORDER BY _distance LIMIT 10""").show()

# (iv) Distributed. The same code on Daft's Ray runner: one task per planned file.
import daft
daft.set_runner_ray(address="ray://head:10001")
ann_sql(tbl, "embedding", q, k=10, sql="...")            # unchanged
```

Runnable versions are `examples/01_pyiceberg.py` … `examples/04_ray.py`. The
semantics (what is pushed down, `k` vs `LIMIT`, time travel, deletes) are covered in
[docs/querying.md](docs/querying.md).

## Writing an indexed table

There are two steps: **embed**, then **index + register**. Both run on one machine or on Ray.

```bash
# single node
python -m lakehouse_ann.write embed docs.parquet --text-col text --out staging/
python -m lakehouse_ann.write index --catalog prod --dest vec.docs \
    --input 'staging/*.parquet' --partition-by category

# distributed: the same commands plus --ray (embedding on GPUs with --gpus 1)
python -m lakehouse_ann.write embed 's3://bucket/raw/*.parquet' --text-col text \
    --out s3://bucket/staging/ --ray ray://head:10001 --gpus 1
python -m lakehouse_ann.write index --catalog prod --dest vec.docs \
    --input 's3://bucket/staging/*.parquet' --partition-by 'customer_id:truncate[200]' \
    --ray ray://head:10001
```

`index` builds a k-means index per output file, sorts rows by cluster (optional;
`--no-cluster-sort` keeps the input order), uploads, writes the indexes to Puffin
files (or into each data file with `--index-location footer`), and registers
everything in one Iceberg commit. Every output file holds a single partition value.
It can re-index an existing table into a new one (`--from-table`) and is safe to re-run.

Already have a table? Index it **in place**, without rewriting any data file:

```bash
python -m lakehouse_ann.write attach --catalog prod --table vec.docs --column embedding [--ray]
```

The same API
is available from Python (`lakehouse_ann.write.write_indexed_table`,
`lakehouse_ann.embed.embed_text`). See [docs/writing.md](docs/writing.md) and
`examples/write_local.sh` / `examples/write_ray.sh`.

## Results

Recall@10 is measured against exact search. Full setups, caveats and more cases are in
[docs/benchmarks.md](docs/benchmarks.md).

| Setup | Query | Baseline | lakehouse-ann | |
|---|---|---|---|---|
| 1M × 128 synthetic, local, no pruning possible | unfiltered | brute force 522 MB read | **52 MB**, recall 1.00 | 10× less I/O |
| same | `tier` = 1% of rows | post-filter returns **0.1 of 10** rows | 10 rows, recall 0.83 | filter during ranking |
| same, partitioned by the filter column | correlated filter | 0.50 recall, 16 files | **0.97 recall, 1 file**, 6 ms | Iceberg pruning |
| 1M, index location | sorted rows, unfiltered | footer: 52 MB, 187 req | **Puffin: 52 MB, 187 req** | same cost, no data rewrite to (re)index |
| 5M × 384 Granite, IBM COS, laptop | `category = 'tech'` | 77 s, 1.5 GB | **12.6 s, 190 MB** | 6.1× |
| 5M, in-cluster, native → Ray (5 workers) | scalar filter, all files | 35.4 s | **8.4 s**, identical top-k | 4.2× |
| 50M × 384, Ray (5 workers × 8 vCPU) | unfiltered (250 files) | 83.1 s | **24.9 s**, recall 1.00 | 3.3× |
| same | partition band (50 files) | 23.6 s | **6.0 s**, recall 1.00 | 3.9× |
| same | band + scalar filter | 29.4 s | **8.2 s**, recall 1.00 | 3.6× |

"Baseline" means brute force or post-filtering in the synthetic rows. In the
5M/50M rows it means the same table indexed by a writer that keeps the original row
order and large row groups, except the native → Ray row, which compares Daft
runners on the same table.

## Why Daft

The design needs an engine that hands **the query's filter to a custom data
source as an expression**, so it reaches each per-file search before ranking.
Daft's Python `DataSource` API does exactly that, along with column projection.
The same source then works with no extra code as a DataFrame, behind SQL
(`daft.sql`) and on Ray. Daft is Arrow-native with a Rust engine, so the kernel
stays plain numpy/pyarrow, and its class UDFs run embedding models on GPUs for the
write path. A longer comparison with Spark, DuckDB, Ray Data and pyiceberg on
its own is in [docs/why-daft.md](docs/why-daft.md).

The index itself is engine-neutral: every file stays plain Parquet. For what
Spark, Trino, DuckDB or any other engine would add to search it fast, see
[docs/other-engines.md](docs/other-engines.md). For where GPU acceleration with NVIDIA
cuVS could fit (index builds, compressed codes, batched filtered search), see
[docs/cuvs.md](docs/cuvs.md).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"          # core; add ,ray and/or ,embed for those modes
python bench/make_data.py --out data          # synthetic 1M x 128, ~1 min
python bench/iceberg_demo.py                  # local SQLite catalog + the pyiceberg demo
pytest -q                                     # 44 tests, no network (data-dependent ones skip without the two steps above)
python examples/01_pyiceberg.py               # ... 02_daft, 03_sql, 04_ray
examples/write_local.sh                       # embed -> index -> register -> query
```

To query your own catalog (REST, Glue, Hive, SQL, …), configure it the
standard pyiceberg way and run `examples/remote_catalog.py`. For a Ray cluster on
Kubernetes, see [deploy/kubernetes](deploy/kubernetes/README.md).

## Repository layout

```
lakehouse_ann/   format.py              IFVI/IFVR index blobs
                 puffin.py  footer.py   where they live: Puffin files (default) or the Parquet footer
                 build.py               k-means, cluster-sort, write one indexed file
                 search.py              per-file kernel: filter → plan → fetch → score → project
                 iceberg.py             (i)   pyiceberg-only vector_search
                 source.py              (ii)  Daft DataSource over a glob of indexed files
                 iceberg_sql.py         (ii)/(iii)/(iv)  Daft DataSource planned by pyiceberg, ann_sql
                 embed.py  write.py     write path: embed, index, register, attach (local or Ray)
examples/        one script per query mode, write-path scripts, remote catalog
bench/           synthetic data, local benchmarks, SQL demo, Ray benchmark, results/*.json
deploy/          Ray on Kubernetes: manifest, deploy script, Dockerfile
docs/            design, querying, writing, benchmarks, why Daft, other engines, cuVS
tests/           format, footer, kernel, writer, all front ends, write path
```

## Status and limits

This is research code that accompanies the paper. It supports **L2 distance** only;
for cosine, normalize the vectors, which `embed` does by default. It has no PQ compression
(see [docs/cuvs.md](docs/cuvs.md) for a GPU path to it), and doesn't prune across files by vector distance: the per-file centroid and radius
are written but not yet used. Equality deletes are not supported (positional
deletes and deletion vectors are). See the limits sections of the docs.

## Citation

```bibtex
@article{jain2026filtered,
  title   = {Filtered Vector Search in a Disaggregated Lakehouse: Composing
             Table-Format Pruning with Per-File {ANN}},
  author  = {Jain, Rakesh and Griffin, Thomas and Zawad, Syed},
  journal = {arXiv preprint arXiv:2608.05441},
  year    = {2026}
}
```

Licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
