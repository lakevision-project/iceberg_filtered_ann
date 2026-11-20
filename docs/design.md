# Design

lakehouse-ann searches an Iceberg table whose Parquet data files each carry a
small IVF index. A query does three things: Iceberg prunes files, the per-file
kernel searches each surviving file with the filter applied *during* ranking,
and the engine merges the per-file top-k. This page covers the file format, the
kernel and the layout choices that make it fast.

## 1. An indexed Parquet file

```
PAR1 [data pages] [IFVI blob] [IFVR blob] [FileMetaData] [len] PAR1
                                           KV  ifs.vector.index.v1 = {"offset", "length"}
                                           KV  ifs.vector.runs.v1  = {"offset", "length"}
```

* The blobs sit **outside** the Thrift footer, between the last data page and
  `FileMetaData`. The footer's key-value metadata holds `{offset, length}`
  pointers to them. A normal Parquet reader seeks to the end, reads
  `FileMetaData` and never touches the blobs, so the file is plain Parquet to
  every engine.
* **IFVI v1** (`lakehouse_ann/format.py`) is the index: header, `n_clusters × dim`
  float32 centroids, posting-list offsets, the posting lists (file-local row
  ordinals, ascending per cluster), an optional whole-file centroid and radius,
  and a CRC32.
* **IFVR** (optional) records where each run of vector values physically sits
  in the file. With it, a reader fetches single vectors with byte-range reads.
  It needs the embedding column written PLAIN and uncompressed, which the
  writer verifies byte by byte. With a compressed embedding column there is no
  IFVR, and the reader fetches vectors by whole row group (see §4).
* **Splicing without Thrift surgery** (`lakehouse_ann/footer.py`): pyarrow's
  writer is deterministic, so the table is written twice, the second time with
  the pointers in the KV metadata. The data pages of the two writes are
  identical (asserted), and the result is `[data pages] + blobs + [footer of write 2]`.

Row ordinals in the index are positions within the file. Iceberg positional
deletes and deletion vectors use the same ordinals, so deleted rows simply
join the filter mask.

## 2. The per-file kernel (`lakehouse_ann/search.py`)

One task searches one file, doing the cheapest work first:

1. **Footer.** Opening the file reads it anyway, and it carries the index pointers.
2. **Filter.** Read only the filter columns and evaluate the predicate (a Daft
   expression or a pyarrow expression) into a row mask, together with any
   deleted positions. If nothing matches, the file is done and its index is never read.
   If everything matches, the mask is dropped.
3. **Plan.** Build two candidate sets from metadata alone:
   * *exact*: every matching row;
   * *IVF*: probe clusters nearest-first, **drop non-matching ids from each
     posting list before scoring**, and keep probing until at least `nprobe`
     clusters with matches and `16·k` matching candidates are covered. Without
     this, a selective filter leaves too few candidates in the usual `nprobe`
     clusters.

   Each set's vector reads are priced (bytes plus a per-request cost), and the
   kernel takes *exact* unless IVF is clearly cheaper (`exact_bias`, 1.5×).
4. **Fetch.** Read only the candidate vectors: coalesced byte ranges with
   IFVR, or whole row groups without it.
5. **Score.** Exact L2, keeping this file's top-k.
6. **Project.** Read the requested columns for the k winners only.

The engine (Daft, or the pyiceberg thread pool) then merges the per-file
top-k on `_distance`.

Plans reported per file: `skip:no-match`, `exact:filtered`, `exact:no-index`
(a file without an index is scanned exactly), `ivf`, `ivf:filtered`.

### Reading filter columns from many-row-group files

Small row groups make vector fetches cheap (§4), but a filter column then
spans hundreds of tiny column chunks. `SearchConfig.filter_read="batched"`,
the default, takes every filter chunk's byte range from the footer, merges
chunks within 64 KiB (several filter columns of one row group become one
read), fetches them all as raw ranged reads on a dedicated wide pool
(`filter_io_threads`, 64), and decodes them in one `ParquetFile.read` over an
in-memory range cache. On a 50M-row table that cut filtered queries by 2.3–2.6×
compared with one `read_row_group` call per row group (`filter_read="per_rg"`).
See [benchmarks](benchmarks.md#6-50m-rows-on-ray).

## 3. Why filter during ranking

The usual way to combine an ANN index with a predicate is to take the top-k
nearest and then apply the filter. At 1% selectivity, that returns about
0.1 of 10 rows on average, with no error ([benchmarks §1](benchmarks.md#1-1m-synthetic-vectors-local)).
Pushing the filter into the candidate lists keeps the result size right. The
extra probing keeps recall up when the filter removes most of each cluster.

In Daft this is just operator order. A `.where()` directly on the source is
pushed into every task (`pushdowns.filters`); one placed after
`.sort().limit()` cannot be pushed and becomes a post-filter.

## 4. Layout: where most of the speed comes from

* **Partition by the filter columns.** Iceberg then drops whole files at
  planning time. In a file whose partition already proves the predicate, the
  per-file residual is `AlwaysTrue`, so no filter column is read and the search
  is a plain unfiltered IVF. This is the paper's purity argument. It also fixes
  the hardest case for a per-file index: a filter *correlated* with the vectors.
  There, a mixed file's centroids rank clusters poorly for the matching
  subset, and recall drops to about 0.4–0.5. Partitioned, the same query
  reaches 0.97 while reading 1 of 20 files.
* **Optionally, sort rows by cluster inside each file.** This isn't needed for
  correctness. Any row order is searched correctly with the same recall, and pruning
  and filter-during-ranking work the same. It is a performance optimization,
  and on object storage a large one. The writer (by default) reorders a file's rows by IVF
  cluster, so every posting list is one contiguous run of rows and of bytes,
  and a probe reads about `nprobe` ranges instead of thousands of scattered
  vectors. Pruning can't substitute for it: pruning decides *which* files are
  searched (by the scalar filter), while sorting decides how many bytes each
  searched file costs (by the vector). In the original row order, the probe
  still scores only a fraction of the vectors, but the coalesced reads cover
  almost the whole column (534 MB against 52 MB unfiltered), close to brute
  force. Row order inside a new data file has no meaning in Iceberg, so the
  sort costs nothing at write time. The one trade-off is that it uses up the file's
  sort order, so the file can't also be sorted by another column to sharpen
  row-group statistics. Turn it off with `cluster_sort=False` / `--no-cluster-sort`.
* **Row groups sized to clusters.** With a compressed embedding column,
  vectors are fetched by row group. Keep `row_group_size` near
  `rows / n_clusters` so a probe reads a few whole groups at about 1.3×
  amplification, not 20×.
* **File size.** Roughly 100k–1M rows per file: enough for per-file k-means to
  pay off, and small enough to prune and to parallelize across.

## 5. Iceberg integration

Planning is pyiceberg's `table.scan(row_filter, snapshot_id).plan_files()`:
partition and manifest-statistics pruning, per-file `task.residual` (what is
left of the predicate for that file) and `task.delete_files`. The Daft front
end converts the SQL WHERE into that `row_filter` with Daft's own Iceberg
expression converter, then emits one task per planned file carrying its
residual (as a pyarrow expression) and its deleted positions.

* **Time travel.** `snapshot_id=` plans an older snapshot. Indexes travel with
  the files.
* **Deletes.** Positional deletes and deletion vectors are applied as part of
  the mask. A copy-on-write delete rewrites the affected file *without* an
  index; that file is then scanned exactly (`exact:no-index`), so results stay
  correct until it is re-indexed. Equality deletes raise an error, as in
  pyiceberg's own reader.
* **Registration.** Indexed files are registered with `add_files()`, which
  infers partition values from column statistics. Nothing is rewritten, so the
  indexes survive.

## 6. Limits

* L2 distance only. Normalize vectors for cosine; the embed step does by default.
* No product quantization; vectors are scored at full float32 precision.
* No cross-file pruning by vector distance. Per-file centroid and radius are
  written, but tasks don't share a running top-k cutoff. A two-phase driver could use them.
* `k` must be at least the SQL `LIMIT`, because Daft doesn't push a limit through
  a sort. `ann_sql` raises `k` automatically ([querying](querying.md)).
* Columns are matched by name; schema evolution beyond that is untested.
* Under Ray, per-file statistics (`STATS`) stay in the worker processes.
