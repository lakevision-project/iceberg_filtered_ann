# GPU acceleration with cuVS

[cuVS](https://github.com/rapidsai/cuvs) is NVIDIA's GPU library for vector
search and clustering (k-means, IVF-Flat, IVF-PQ, CAGRA, brute force). This
page describes where it could plug into lakehouse-ann. **None of it is
implemented yet**: every measurement in this repository is CPU-only, and the
cuVS capabilities named below are our understanding of the library, to be
checked against the version you use.

The short version: the index *format* stays engine- and hardware-neutral
([other engines](other-engines.md)). cuVS is a fast way to **build** those
indexes and to **search** them where GPUs are available.

## Where the time goes today

| stage | what happens | where the cost is |
|---|---|---|
| write: embed | text → vectors in a Daft class UDF | already on GPU when available (`embed --gpus`) |
| write: build index | k-means per data file, assign rows, sort rows by cluster | about 35–45 s of each file's 45–60 s on CPU; 10.7 min for a 50M-row, 250-file rebuild on 5 × 8 vCPU ([writing](writing.md#scale-reference)) |
| query: plan and prune | Iceberg metadata drops files by predicate | small |
| query: probe and fetch | masked posting lists, then candidate vectors from object storage | most of the bytes: ~85% of the 52 MB read for an unfiltered 1M query are candidate vectors |
| query: score | L2 over candidates, per-file top-k, merge | small for one query |

Queries are bandwidth-bound on object storage, and index builds are
compute-bound. The four opportunities below target those two costs.

## 1. Build indexes on GPU

*Low effort, no format change.*

The per-file index is IVF-Flat in all but name: k-means centroids plus posting
lists of row positions ([design §1](design.md#1-an-indexed-parquet-file)).
cuVS k-means, or its IVF-Flat build, could compute the centroids and the
row-to-cluster assignment on GPU. `lakehouse_ann.build` would then serialize
them into the same IFVI blob, so readers see no difference.

* Embedding already runs on GPU in the same pipeline, so embedding and indexing
  could share one GPU pass per file.
* On Ray this is a `num_gpus` setting on the index task
  (`lakehouse_ann.write.index_one` / `attach_one`).
* Open question: the best call pattern for many independent builds of
  100k–300k vectors each, one call per file or batched.

A natural interface is a `--gpu` option on `write index` and `write attach`
that swaps `build.kmeans` / `build.assign` for their cuVS equivalents.

## 2. Compressed codes as a Puffin blob

*Largest query-side win; needs a new blob type.*

Candidates are ranked on full float32 vectors fetched from the data files:
1,536 bytes per vector at 384 dimensions. cuVS IVF-PQ compresses each vector to a
short code (for example 48 bytes). The idea:

1. build PQ codes per data file and store them as an additional Puffin blob
   next to the IVF blob (for example `lakehouse-ann-ivf-pq-v1`);
2. probe and rank candidates on the codes, read from the small Puffin blob;
3. re-rank the best few hundred with exact vectors from Parquet.

Ranking reads would shrink roughly by the compression ratio (up to ~30× for
48-byte codes at 384 dims), on top of pruning and cluster sorting. Because
queries are bandwidth-bound, that is a direct latency win. This is an estimate,
not a measurement.

Puffin makes this clean: a new blob type, no data rewrite, and engines that
don't understand the codes fall back to the IVF blob or an exact scan. For a
community standard, the codebook and code layout would need to be stable and
documented, so that CPU-only readers can score them too.

## 3. Filtered and batched scoring on GPU

*Medium effort; pays for batches of queries.*

As we understand it, cuVS search accepts a **bitset prefilter**, and candidates
outside it are skipped. The per-file row mask we already build from the
predicate and the deleted positions is that bitset, so filter-during-ranking
([design §3](design.md#3-why-filter-during-ranking)) maps directly onto it.

Where a GPU helps:

* **batches of queries** against the same files (recommendations, deduplication,
  evaluation runs): read a file's candidates once, score many queries;
* **exact scans of the files left after pruning**, with the Parquet decode
  also on GPU.

For a single interactive query, transfer and kernel-launch overhead likely
outweigh the gain, because object-storage reads dominate. Open question: how
to search many small per-file indexes, each with its own bitset, in one call.

## 4. CAGRA as an alternative per-file index

*Exploratory.*

A CAGRA graph per data file could be stored as another Puffin blob type. A
file can carry several typed index blobs; an engine uses the ones it
understands and otherwise searches exactly
([other engines](other-engines.md#four-rules-for-correct-results), rule 3).

Concerns:

* graph search touches scattered rows, which works against contiguous reads from
  object storage (the reason rows are cluster-sorted);
* graph indexes degrade under very selective filters;
* a library-specific serialized format is harder to standardize across engines;
* CPU engines would need a conversion path, for example CAGRA to HNSW.

## What stays neutral

The proposal to the Iceberg community covers only the IVF blob layout, Puffin
blob types, how a reader finds a data file's index, and the distance metric.
Nothing in it requires a GPU or a particular library. GPU-built indexes are
ordinary blobs that any engine can search on CPU, and GPU search is one more
implementation of the same per-file kernel.

## Open questions for the cuVS project

1. For many independent indexes of 100k–300k vectors, what is the most efficient
   build path: one call per file, or batched builds?
2. Is there, or could there be, a stable, documented serialization for IVF-PQ
   codebooks and codes that CPU readers could implement?
3. Which index types support bitset prefilters, and how does performance hold
   at 0.1–1% selectivity?
4. Can a batch of queries be searched across many small indexes in one launch,
   each with its own filter bitset?

## Suggested order

1. GPU index build (§1), measured on the 50M-row rebuild.
2. IVF-PQ codes in Puffin (§2), measuring bytes, latency and recall against the
   current layout.
3. Batched filtered scoring (§3), if batch workloads matter.
4. CAGRA (§4) only after the above, as a comparison point.
