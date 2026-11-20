# Vector search in other engines

The index is a property of **data files**, not of an engine. Every indexed
file is ordinary Parquet, so any engine can read these tables today. An engine
that also wants *fast filtered vector search* adds a small amount of code,
described on this page. Nothing here needs changes to the Iceberg spec or to
other engines, and engines can adopt it independently and incrementally.

## Level 0: read the table (no work)

Every engine that reads Iceberg + Parquet already reads these tables. The
index blobs sit outside the Parquet footer, where readers never look. An
engine with a vector distance function can already answer the query exactly by
brute force:

```sql
SELECT id, l2_distance(embedding, :q) AS d
FROM docs WHERE category = 'tech'
ORDER BY d LIMIT 10
```

That is correct, and Iceberg pruning still applies, but it reads every vector of
every surviving file. The levels below avoid that.

## Level 1: a per-file search (the core)

Add one function that searches **one data file**:

```
search_file(file, query_vector, k, predicate, deleted_positions) -> top-k rows + distance
```

A minimal correct version:

```
1. mask   = rows of `file` matching `predicate`, minus `deleted_positions`
            (read only the predicate's columns)
2. if the file has no index        -> score every masked row exactly, return top-k
3. index  = the file's blob in a Puffin file (from the table's index registry;
            use it only if the blob's referenced-data-file, size and row count
            match this file), else the footer KV pointer `ifs.vector.index.v1`;
            one ranged read either way
4. probe the clusters nearest to `query_vector`, in order; from each posting
   list keep only rows in `mask`; stop after `nprobe` clusters that had matches
   AND at least ~16·k candidates
5. read those candidates' vectors, compute L2 distance, keep the k smallest
6. read the requested columns for those k rows
```

* The index layout is documented in [design §1](design.md#1-an-indexed-parquet-file)
  and `lakehouse_ann/format.py`: a header, centroids, and posting lists of row
  positions within the file. Parsing it takes a few dozen lines.
* Row positions in the index are the same positions Iceberg positional deletes
  and deletion vectors use, so deletes are just part of the mask.
* Step 4 is what "filter during ranking" means: non-matching rows are dropped
  from the candidate lists **before** scoring. A file whose partition already
  proves the predicate gets an empty predicate, and step 1 is skipped.
* The reference implementation (`lakehouse_ann/search.py`) adds an exact-vs-IVF
  cost choice and read coalescing. Those are optimizations; the six steps above
  are enough for correct results.

## Level 2: call it from a query

The engine plans the scan as it already does: Iceberg pruning, residual filter
per file, delete files. Then, instead of scanning each file, it calls
`search_file` and merges the per-file results with its normal top-k
(`ORDER BY distance LIMIT k`). The simplest way to expose this is a
**table function** (or read option) that takes the vector and `k`:

```sql
SELECT * FROM vector_search('docs', 'embedding', :q, k => 10)
WHERE category = 'tech'
ORDER BY _distance LIMIT 10
```

It returns the table's columns plus `_distance`, with up to `k` rows per file.
Optionally, an optimizer rule can later recognise
`ORDER BY l2_distance(embedding, :q) LIMIT k` over a table scan and rewrite it to
the same thing, so plain SQL works.

## Three rules for correct results

1. **The predicate must be applied inside `search_file`.** If the engine pushes
   the top-k into the scan but applies the WHERE clause *after* it, results
   come back short with no error (post-filtering). Either pass the whole
   predicate into the per-file search, or don't push the top-k down.
2. **Per-file `k` must be at least the query's `LIMIT`.**
3. **A file without an index is searched exactly.** Files written or compacted by
   writers that don't build the index stay correct, just slower, until re-indexed.

## Notes per engine

These describe the natural extension point in each engine as we understand it;
verify against the version you target.

| engine | where it plugs in | notes |
|---|---|---|
| **Spark** | a DataSource V2 scan (filter + column pushdown, one partition per file), or the Python Data Source API / `mapInArrow`, which can call the reference implementation directly | Spark only pushes `ORDER BY … LIMIT` into a scan when no filters remain after it, which enforces rule 1 automatically. |
| **Trino** | a table function first; later, the Iceberg connector's filter + top-N pushdown | For top-N pushdown, the connector must accept the *whole* predicate (rule 1). |
| **DuckDB** | a table function in an extension, alongside the Iceberg extension | Pass `k` (and complex predicates) as function arguments. |
| **Daft** | a Python `DataSource` (this repository) | The filter arrives as an expression; the same source runs locally and on Ray. |
| **pyiceberg** | `plan_files()` + `search_file` in a thread pool (this repository) | No engine at all. |

## Writing

Engines don't need to write indexes to search them. Tables written by any
engine can be indexed in place: `python -m lakehouse_ann.write attach` builds
an index per data file into Puffin files without touching the data. Files
rewritten later, for example by compaction, simply have no matching index and
fall back to exact search (rule 3). Re-running `attach` indexes only those files.

## What a community standard would need to agree on

To make this portable across engines, only a few things need to be written down:

* the **index blob layout** (IVF centroids + posting lists of row positions;
  today `IFVI v1`, see `format.py`) as a **Puffin blob type**, plus the optional
  byte-runs blob for ranged reads;
* **how a reader finds a data file's index blob**. The natural place is the
  manifest, the way v3 deletion vectors point at their Puffin blob
  (`referenced_data_file`, `content_offset`, `content_size_in_bytes`).
  Until then, this repository uses a table property listing the Puffin files
  (see [design §1b](design.md#1b-where-the-index-lives-puffin-default-or-the-footer));
* the **distance metric** (L2 today; cosine via normalized vectors).

Puffin keeps data files untouched: an index can be added to an existing table,
rebuilt, or dropped without rewriting data, and it isn't tied to Parquet. One
practical detail for the proposal: some readers only accept the known
*statistics* blob types (pyiceberg 0.12 rejects others when loading table
metadata), which is why the index isn't registered as table statistics today.
The footer embedding (`ifs.vector.index.v1`) remains an optional co-located
alternative.

Everything else (k-means, probing strategy, cost model, caching) can stay
an implementation choice for each engine. Two engines with different search code
still return correct results from the same files.
