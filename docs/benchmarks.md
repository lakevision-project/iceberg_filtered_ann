# Benchmarks

Six experiments, from a laptop to a 50M-row table on a Ray cluster. Unless noted:
k = 10, L2 distance, and recall@10 against **exact** search over the same rows and
the same filter. Raw results are in `bench/results/`.

How to read the "baseline" columns:

* §1–§3 compare against **brute force** (exact, reads every vector) and
  **post-filtering** (top-k first, then the filter, which is what an index blind
  to filters gives you).
* §4 and §6 compare the **same table indexed two ways**. The baseline copy was
  written by a separate writer that keeps the original row order and large row
  groups, with its own per-file IVF index in the same footer format. The other
  copy was written by `lakehouse_ann.write` (rows sorted by cluster, small row
  groups, partitioned). Both are searched by the same kernel.
* §5 compares Daft's native and Ray runners on one table.

Datasets in §4–§6 (5M and 50M text rows embedded with
`ibm-granite/granite-embedding-97m-multilingual-r2`, 384 dims) are not public.
§1–§3 are fully reproducible with the scripts in `bench/`. To run §4–§6 on
your own data, see [reproducing](#reproducing-on-your-own-data).

---

## 1. 1M synthetic vectors, local

```bash
python bench/make_data.py --out data --also-scattered
python bench/bench.py                                   # -> bench/results/local_1m.json
python bench/bench.py --data data/indexed_scattered \
    --json bench/results/local_1m_scattered.json        # same index, rows NOT cluster-sorted
```

1M rows × 128 dims, 16 files of 62.5k rows, 250 clusters per file, nprobe = 8,
20 queries. The vectors are a 2000-topic Gaussian mixture. Rows are spread
across files at random, so **no file can be pruned**: the worst case for a
per-file index.

| filter | selectivity | brute recall | **index recall** | post-filter recall (rows returned) | I/O brute | **I/O index** | object-store *model* brute → index |
|---|---|---|---|---|---|---|---|
| none | 100% | 1.00 | **1.00** | 1.00 (10) | 522 MB / 272 req | **52 MB / 187** | 672 → 142 ms |
| tier = p50 | 50% | 1.00 | **1.00** | 0.47 (4.8) | 522 MB / 400 | **52 MB / 319** | 732 → 202 ms |
| tier = p10 | 10% | 1.00 | **1.00** | 0.11 (1.1) | 522 MB / 400 | **51 MB / 359** | 732 → 231 ms |
| tier = p1 | 1% | 1.00 | **0.83** | 0.02 (0.1) | 522 MB / 400 | **93 MB / 730** | 732 → 453 ms |
| tier = p01 | 0.1% | 1.00 | **1.00** (exact plan) | 0.00 (0) | 522 MB / 400 | **26 MB / 1066** | 732 → 536 ms |
| region = r3, correlated with the vector | 5% | 1.00 | **0.40** | 0.00 (0) | 522 MB / 400 | **52 MB / 342** | 732 → 232 ms |
| price < 5 & tier = p10 | 0.5% | 1.00 | **0.91** | 0.02 (0.1) | 531 MB / 528 | **108 MB / 1233** | 801 → 708 ms |

* "I/O index" is **measured**: every read the task made, including footer,
  filter column, index blobs, vectors and projection. "I/O brute" is a lower
  bound from the column-chunk sizes brute force must read.
* The object-store column is a **model, not a measurement**: 30 ms per GET, 64
  concurrent GETs, 1 GB/s per node. On a laptop everything sits in the page
  cache and Daft's Rust brute force takes 50–180 ms, so local wall-clock is not
  where an index wins. Bytes and requests are; §4–§6 measure the real thing.

**Findings**

1. **Filtering during ranking is essential.** Used the usual way (top-k, then
   filter), the same index returns 0.1 rows instead of 10 at 1% selectivity,
   with no error.
2. **10× less I/O** for unfiltered and moderately selective (≥ 10%) filters, at
   recall 1.00.
3. **Sorting rows by cluster is optional, but it drives the I/O saving.** Results
   are correct either way. In the original row order
   (`bench/results/local_1m_scattered.json`), candidates are spread through the
   file and the coalesced reads cover almost the whole column: 534 MB against
   52 MB unfiltered, close to brute force. Pruning doesn't replace it: in §4 both
   copies prune to the same 2 files, and the sorted copy with small row groups
   is still 6.1× faster, because pruning picks the files and the in-file layout
   sets the bytes read inside each one.
4. **Very selective filters (≤ 1%) trade cost against recall.** The planner
   switches to exact scoring when that is cheap (0.1% → recall 1.00), but
   scattered matching rows are expensive to fetch one by one. `filtered_overfetch`
   at 1%: 1× → 0.73, 16× → 0.87, 32× → 0.93 recall for 37 → 92 → 180 MB.
5. **Correlated filters are the hard case (0.40)**, because a mixed file's centroids rank
   clusters poorly for the matching subset. The answer is layout (§2).

## 2. pyiceberg front end: layout, deletes, time travel

`python bench/iceberg_demo.py` (local SQLite catalog, same 1M rows, 10 queries)

`vec.docs` has 16 files in a random layout. `vec.docs_by_region` has 20 files,
one per region, partitioned `identity(region)`.

| filter | table | recall | files searched | read | time |
|---|---|---|---|---|---|
| none | docs | 1.00 | 16 | 52 MB | 62 ms |
| tier = 'p10' | docs | 1.00 | 16 | 52 MB | 58 ms |
| tier = 'p1' | docs | 0.87 | 16 | 92 MB | 82 ms |
| region = 'r3' (correlated) | docs | **0.50** | 16 | 52 MB | 54 ms |
| region = 'r3' (correlated) | docs_by_region | **0.97** | **1** | **1.5 MB** | **6 ms** |
| region = 'r3' and tier = 'p10' | docs | 0.91 | 16 | 87 MB | 97 ms |
| region = 'r3' and tier = 'p10' | docs_by_region | 0.97 | 1 | 1.6 MB | 7 ms |

Partitioning by the filter column fixes the correlated case. Iceberg prunes 19 of 20
files, and in the remaining file the residual is `AlwaysTrue`, so the search is a
plain unfiltered IVF. The demo then deletes rows copy-on-write. The rewritten
file has no index and is scanned exactly, with correct results and the deleted
rows absent. Time travel to the earlier snapshot finds the indexed file again.

## 3. SQL front end

`python bench/sql_demo.py` (same catalog, one query)

| query | table | rows | recall | files searched | plan |
|---|---|---|---|---|---|
| `tier = 'p10'` | docs | 10 | 1.00 | 16 | ivf:filtered |
| `region = 'r3'` (correlated) | docs | 10 | 0.30* | 16 | ivf:filtered |
| `region = 'r3' and tier = 'p10'` | docs | 10 | 0.90 | 16 | ivf:filtered |
| `region = 'r3'` | docs_by_region | 10 | 1.00 | **1** | exact:no-index |
| `region = 'r3'`, first snapshot | docs_by_region | 10 | 1.00 | **1** | ivf |

\* single-query variance; §2's 0.50 is a 10-query mean. A test pins SQL results
to `vector_search()` exactly, so the SQL path adds no recall loss of its own.
Partition pruning works through SQL (19 of 20 files pruned), joins and
aggregations over the ANN result work in the same statement, and `LIMIT 15`
with `k=10` returns a correct 15 because `k` is raised automatically.

## 4. 5M rows on object storage, from a laptop

5,020,000 rows × 384 dims (Granite embeddings), 10 files, partitioned
`identity(category)` over 5 categories, on IBM Cloud Object Storage, behind
an Iceberg REST catalog. The client is a laptop over the internet (about 25–35
MB/s aggregate), with the SQL front end.

* **baseline copy**: original row order, uncompressed embedding,
  122,880-row groups, a 4096-cluster index per file, nprobe = 8;
* **lakehouse-ann copy**: 512 clusters per file, rows sorted by cluster, zstd
  embedding, 512-row groups, nprobe = 32. Same rows and partitions.

| query | baseline | lakehouse-ann | faster | less vector I/O |
|---|---|---|---|---|
| no filter (10 files) | 273 s, 7.7 GB | **44 s, 0.76 GB** | 6.2× | 10.1× |
| `category = 'tech'` (2 files) | 77 s, 1.5 GB | **12.6 s, 190 MB** | 6.1× | 8.1× |
| `+ customer_id < 200` | 79 s, 1.5 GB (exact:filtered) | **32 s, 186 MB** (ivf:filtered) | 2.5× | 8.3× |
| recall@10 (tech, vs exact) | 10/10 | 9/10 | | |

Both sides are bandwidth-bound, so the speedups track the byte reductions. The
one recall miss (exact rank 3) sits in a cluster outside the 32 nearest
centroids. Raising nprobe to 64 (333 MB) doesn't find it either. The
knobs are fewer clusters or a higher nprobe, traded against bytes.

## 5. Native vs Ray runner, 5M rows

The lakehouse-ann copy from §4, queried **inside the cloud region** (in-cluster
access to object storage), nprobe = 32, 3 query vectors × 2 runs, median.
Native means Daft's native runner in one 4-vCPU pod. Ray means 5 workers
× 8 vCPU / 32 GiB on separate Kubernetes nodes, with the head hosting only the
driver.

| query | files | native | Ray | same top-k |
|---|---|---|---|---|
| no filter | 10 | 4.4 s | **2.5 s** (1.8×) | 3/3 |
| `category = 'tech'` | 2 | **1.1 s** | 1.4 s | 3/3 |
| `category = 'tech' AND customer_id < 200` | 2 | 5.2 s | 5.1 s | 3/3 |
| `customer_id < 200` (all files) | 10 | 35.4 s | **8.4 s** (4.2×) | 3/3 |

* Ray helps when a query touches many files. After pruning to 2 files there is
  little to distribute, and scheduling overhead makes it about even.
* Work was spread evenly (33–37 tasks per worker). Each worker returns only its
  file's top-k (`PhysicalScan → TopN` runs on the worker).
* Running next to the data matters as much as the runner: the same unfiltered
  query took 44 s from the laptop (§4) and 4.4 s in-region.
* The scalar rows predate the batched filter read (§6), which would make them faster.

## 6. 50M rows on Ray

50,016,000 rows × 384 dims (Granite), on IBM Cloud Object Storage, in-region,
on the same 5-worker Ray cluster. `bench/ray_bench.py`, nprobe = 32, 3 query
vectors, median. Recall@10 is against exact ground truth over all 50M rows,
computed on the cluster in 223 s.

* **baseline copy**: 250 files of 150–250k rows, **one row group per file**,
  zstd embedding, original row order, 256-cluster index per file,
  unpartitioned (and without column statistics in its manifests, so nothing prunes);
* **lakehouse-ann copy**: re-indexed with `lakehouse_ann.write`: 250 files, 256
  clusters per file, rows sorted by cluster, 512-row groups, zstd,
  partitioned `truncate(200, customer_id)` into 5 bands of 10M rows. The build took
  10.7 minutes on the cluster ([writing](writing.md#scale-reference)).

| query | baseline | lakehouse-ann | faster | recall (both) |
|---|---|---|---|---|
| no filter | 83.1 s (250 files) | **24.9 s** (250 files) | 3.3× | 10/10 |
| band `customer_id ∈ [400, 600)` | 23.6 s (250 files) | **6.0 s** (50 files) | 3.9× | 10/10 |
| band + `customer_id < 450` | 29.4 s (250 files) | **8.2 s** (50 files) | 3.6× | 10/10 |
| `customer_id IN (5 ids)`, 0.5% of rows, all files | 78.1 s | **34.0 s** | 2.3× | 10/10 |

**The filter-column read** (`bench/results/ray_50m_filter_ab.json`). Small row
groups make vector reads cheap but split each filter column into hundreds of
tiny chunks per file. Reading them one row group at a time made filtered
queries slow. That configuration was slower than the baseline for the 0.5%
`IN` query, where the baseline's one-row-group files need a single filter read.
Batching the reads fixed it:

| query | `filter_read="per_rg"` | `"batched"` (default) | faster |
|---|---|---|---|
| band + `customer_id < 450` (50 files) | 19.1 s | **8.2 s** | 2.3× |
| `IN` 5 ids (250 files) | 87.7 s | **34.0 s** | 2.6× |

Recall was 10/10 in every run. The first query after a restart is slower (cold
start), so medians are reported.

**Caveats.** Three query vectors per case is a small sample, and object-storage
latency varies from run to run. Under Ray, per-file I/O counters stay on the
workers, so §5–§6 report time and recall rather than bytes.

---

## Reproducing on your own data

1. Build an indexed table: `python -m lakehouse_ann.write embed …` then `… index …`
   ([writing](writing.md)). To compare layouts, write the same rows twice, once
   with your current writer and once with `lakehouse_ann.write`.
2. Embed a few query vectors with the same model and save them as `.npy`.
3. Run the benchmark against any pyiceberg catalog, with exact ground truth:

```bash
python bench/ray_bench.py --catalog prod --table vec.docs_new --table vec.docs_old \
    --queries queries.npy --gt --gt-cache gt.json \
    --where "band=customer_id >= 400 AND customer_id < 600" \
    --where "rare=customer_id IN (7, 207, 407, 607, 807)" \
    --runner ray                      # or native, or both (compares top-k across runners)
python bench/ray_bench.py ... --filter-reads per_rg batched     # the §6 A/B
```

A Ray cluster on Kubernetes: [deploy/kubernetes](../deploy/kubernetes/README.md).
