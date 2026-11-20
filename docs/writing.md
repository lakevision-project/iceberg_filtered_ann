# Writing an indexed table

Going from raw data to a searchable Iceberg table takes two stages. Each runs on one
machine or distributed on Ray, with the same code.

```
 raw rows (Parquet / CSV / JSON)                        any pyiceberg catalog
        │                                                         ▲
        ▼  1. embed   lakehouse_ann.embed                         │ one add_files() commit
 staging Parquet with an embedding column ──► 2. index  lakehouse_ann.write
                                               split by partition → k-means → cluster-sort
                                               → splice index into footer → verify → upload
```

Both stages are behind one CLI, `python -m lakehouse_ann.write {embed,index}`,
and two Python functions, `embed_text` and `write_indexed_table`.

## 1. Embed (`lakehouse_ann.embed`)

```bash
python -m lakehouse_ann.write embed 'raw/*.parquet' --text-col text --out staging/ \
    [--model ibm-granite/granite-embedding-97m-multilingual-r2] [--batch-size 64] \
    [--ray [ADDRESS] [--gpus 1] [--concurrency 8] [--rows-per-file 250000]]
```

```python
import daft
from lakehouse_ann.embed import embed_text

df = daft.read_parquet("raw/*.parquet")
df = embed_text(df, "text", "embedding", model="ibm-granite/granite-embedding-97m-multilingual-r2",
                num_gpus=1, concurrency=8)        # GPU/concurrency matter on Ray
df.write_parquet("staging/")
```

* The model runs in a Daft **class UDF**: it is loaded once per worker and fed
  batches. On the native runner that is one process. On Ray, `concurrency`
  workers each hold a copy, placed on GPUs with `num_gpus`.
* The output column is `FixedSizeList<float32, dim>`, **L2-normalized** by
  default. The index ranks by L2, which for unit vectors orders results the same
  as cosine similarity. Embed queries with the same model and the same
  normalization.
* Any sentence-transformers model works. Bring your own embeddings by skipping this
  stage: the index stage only needs a float32 vector column with no nulls.
* On Ray the output is split into files of about `--rows-per-file` rows. The
  native runner can't repartition; the index stage splits oversized files itself.

## 2. Index and register (`lakehouse_ann.write`)

```bash
python -m lakehouse_ann.write index --catalog prod --dest vec.docs \
    --input 'staging/*.parquet' \
    --partition-by category --partition-by 'customer_id:truncate[200]' \
    [--n-clusters 256] [--row-group-size 512] [--max-rows-per-file 500000] \
    [--no-compress-embedding] [--workers 4 | --ray [ADDRESS]]

# re-index an existing table into a new, partitioned one
python -m lakehouse_ann.write index --catalog prod --from-table vec.raw --dest vec.docs \
    --partition-by 'customer_id:truncate[200]' --ray
```

```python
from pyiceberg.catalog import load_catalog
from lakehouse_ann.write import write_indexed_table

write_indexed_table(load_catalog("prod"), "vec.docs", "s3://bucket/staging/*.parquet",
                    "embedding", partition_by=["category"], runner="ray")
```

Each input file is one unit of work, run on a thread with `runner="local"` or as a
Ray task with `runner="ray"`:

1. **read** the file through the catalog's FileIO (local, S3, …);
2. **split** it by partition value, and into pieces of at most `max_rows_per_file`;
3. **index** each piece: k-means on its own vectors (`n_clusters`, capped at
   about one cluster per 64 rows for small pieces), rows **sorted by cluster**,
   the IVF blob spliced in front of the footer;
4. **verify** the footer pointer, the blob's row and cluster counts, and statistics on the
   partition columns;
5. **upload** under `<table location>/data/<partition path>/`. A file already
   there with the same row count and an index is skipped, so a failed run can
   simply be re-run.

Then **one `add_files()` commit** registers every new file. pyiceberg infers
each file's partition from its column statistics, which is why every output
file holds a single partition value. Supported transforms are identity
(`col`) and truncate (`col:truncate[N]`), the order-preserving transforms that
statistics can prove. An existing table keeps its own partition spec. Asking for
a different one raises an error rather than producing mismatched files.

Catalogs come from pyiceberg's standard configuration (`~/.pyiceberg.yaml` or
`PYICEBERG_CATALOG__<NAME>__*` environment variables); no credentials appear in
code. On Ray, the catalog's FileIO travels inside every task, so workers need
no credentials of their own. Staging and table locations must be reachable from
every worker, which means object storage on a real cluster.

## Choosing the knobs

| knob | guidance |
|---|---|
| rows per file | **100k–1M.** Enough rows for per-file k-means to pay off, and small enough to prune and to parallelize across. |
| `n_clusters` | **A few hundred per file**, so that clusters are at least as large as a row group (`rows / n_clusters ≥ row_group_size`). More clusters give finer probes; fewer give more matching rows per probe. |
| `row_group_size` | With a compressed embedding column (the default), vectors are fetched by whole row group: match it to the cluster size (about 512) so a probe reads a few groups at about 1.3× amplification. With `--no-compress-embedding`, the writer also records IFVR byte runs and fetches exact ranges, so row-group size matters much less. |
| partitioning | **Partition by the columns you filter on.** Pure files are pruned by the catalog and need no filter inside. That is also the fix for filters correlated with the vectors ([design §4](design.md#4-layout-where-most-of-the-speed-comes-from)). |
| compression | Float32 embeddings barely compress (about 1% on the Granite tables), so the choice is about fetch granularity. zstd (default) means whole-row-group fetches, so keep the groups small. Uncompressed PLAIN adds IFVR exact ranged reads. |

## Scale reference

On a Kubernetes cluster with 5 Ray workers (8 vCPU / 32 GiB each), re-indexing
a 50M-row, 384-dim table (250 source files, ~119 GB) from object storage, with
k-means of 256 clusters per file, cluster-sorting, 512-row groups and zstd,
took **10.7 minutes**, with about 19 files processed concurrently
(5–10 s read, 35–45 s index, 2–8 s upload per file). The result had the same
50,016,000 rows and was partitioned into 5 bands of 10M rows each. See
[benchmarks §6](benchmarks.md#6-50m-rows-on-ray) for how it queries.

## Lower-level building blocks

* `lakehouse_ann.build.index_table(table, path, column, n_clusters, …)` writes one
  indexed file (also `python -m lakehouse_ann.build in.parquet --out-dir … --column …`).
* `lakehouse_ann.iceberg.create_table_from_files(catalog, name, paths, partition_by=[…])`
  registers already-indexed files as a new table with identity partitions.
