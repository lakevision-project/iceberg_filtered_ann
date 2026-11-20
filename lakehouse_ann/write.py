"""Write path: Parquet (or an existing Iceberg table) → indexed, registered Iceberg table.

    from pyiceberg.catalog import load_catalog
    from lakehouse_ann.write import write_indexed_table

    cat = load_catalog("local")                       # any pyiceberg catalog
    write_indexed_table(cat, "vec.docs", "staging/*.parquet", "embedding",
                        partition_by=["category"])            # single node, threads
    write_indexed_table(cat, "vec.docs", "s3://bucket/staging/*.parquet", "embedding",
                        partition_by=["customer_id:truncate[200]"], runner="ray")

Per input file, one unit of work (a thread, or a Ray task with `runner="ray"`):

  1. read    the file through the catalog's FileIO (local, S3, ...);
  2. split   by partition value, so every output file holds ONE value of each
             partition field — pyiceberg's add_files() infers the partition
             from the file's column statistics and needs that;
  3. index   lakehouse_ann.build.index_table: k-means on the file's own vectors,
             rows SORTED by cluster (each posting list is one contiguous run),
             IFVI blob spliced in front of the Parquet footer;
  4. verify  footer pointer, blob row/cluster counts, statistics on the
             partition columns;
  5. upload  under <table location>/data/<partition path>/ — idempotent: a file
             already there with the same row count and an index is skipped.

Then ONE add_files() commit registers every new file. Nothing is rewritten after
upload, so the footer indexes survive; to every other engine the table is plain
Iceberg over plain Parquet.

Supported partition transforms are the order-preserving ones add_files() can
infer from min/max statistics: identity (`col`) and truncate (`col:truncate[N]`).
"""
from __future__ import annotations

import argparse
import dataclasses
import glob as _glob
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_PART_RE = re.compile(r"^(?P<col>[A-Za-z_][\w]*)(?::(?P<tf>identity|truncate\[(?P<n>\d+)\]))?$")


@dataclasses.dataclass(frozen=True)
class PartitionField:
    column: str
    width: int | None = None          # None = identity, else truncate[width]

    @property
    def name(self) -> str:
        return self.column if self.width is None else f"{self.column}_trunc"

    @classmethod
    def parse(cls, spec: str) -> "PartitionField":
        m = _PART_RE.match(spec.strip())
        if not m:
            raise ValueError(f"bad partition spec {spec!r}: use 'col' or 'col:truncate[N]'")
        return cls(m["col"], int(m["n"]) if m["n"] else None)

    def apply(self, value):
        """The Iceberg partition value for one source value."""
        if value is None or self.width is None:
            return value
        if isinstance(value, str):
            return value[: self.width]
        return value - (value % self.width)       # Iceberg truncate for ints (floor mod)


@dataclasses.dataclass(frozen=True)
class IndexOptions:
    column: str
    n_clusters: int = 256
    row_group_size: int = 512
    compress_embedding: bool = True
    min_rows_per_cluster: int = 64     # small files/partitions get fewer clusters
    max_rows_per_file: int = 500_000   # larger inputs are split into several indexed files


def _split_by_partition(table: pa.Table, fields: list[PartitionField]):
    """Yield (partition values tuple, sub-table) with one value per field."""
    if not fields:
        yield (), table
        return
    keys = []
    for f in fields:
        col = table[f.column]
        uniq = pc.unique(col)
        mapped = [f.apply(v) for v in uniq.to_pylist()]
        if any(v is None for v in mapped):
            raise ValueError(f"partition column {f.column!r} has NULLs; add_files cannot "
                             "infer a partition from a file without min/max statistics")
        codes = {v: i for i, v in enumerate(dict.fromkeys(mapped))}
        idx = pc.index_in(col, value_set=uniq).to_numpy(zero_copy_only=False)
        keys.append((np.asarray([codes[v] for v in mapped])[idx], list(codes)))
    combo = np.stack([k[0] for k in keys], axis=1)
    groups, inverse = np.unique(combo, axis=0, return_inverse=True)
    for gi, g in enumerate(groups):
        rows = np.flatnonzero(inverse.ravel() == gi)
        yield tuple(keys[j][1][g[j]] for j in range(len(fields))), table.take(pa.array(rows))


def _partition_path(fields, values) -> str:
    return "/".join(f"{f.name}={v}" for f, v in zip(fields, values))


def index_one(src_path: str, src_io, dest_io, dest_prefix: str, fields: list[PartitionField],
              opts: IndexOptions) -> list[dict]:
    """Steps 1-5 for one input file. Returns one record per output file."""
    from .build import index_table
    from .footer import read_index, read_runs
    from .format import POINTER_KEY

    t0 = time.perf_counter()
    with src_io.new_input(src_path).open() as f:
        table = pq.ParquetFile(f).read()
    t_read = time.perf_counter() - t0
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out = []
    pieces = []
    for values, part in _split_by_partition(table, fields):
        n_chunks = max(1, -(-part.num_rows // opts.max_rows_per_file))
        step = -(-part.num_rows // n_chunks)
        for c in range(n_chunks):
            pieces.append((values, c if n_chunks > 1 else None, part.slice(c * step, step)))
    del table
    for values, chunk, part in pieces:
        sub = _partition_path(fields, values)
        tag = "" if not fields else "-" + re.sub(r"[^\w.-]", "_", "_".join(map(str, values)))
        tag += "" if chunk is None else f"-{chunk:03d}"
        dest = f"{dest_prefix}/data/{sub + '/' if sub else ''}{stem}{tag}-ivf.parquet"
        rec = {"src": src_path, "dest": dest, "rows": part.num_rows, "partition": values,
               "read_s": t_read}
        try:                                         # resume: already uploaded and indexed
            with dest_io.new_input(dest).open() as f:
                md = pq.ParquetFile(f).metadata
            if md.num_rows == part.num_rows and POINTER_KEY.encode() in (md.metadata or {}):
                out.append({**rec, "skipped": True})
                continue
        except (OSError, FileNotFoundError):
            pass
        k = max(1, min(opts.n_clusters, part.num_rows // opts.min_rows_per_cluster))
        with tempfile.TemporaryDirectory(prefix="ivf-") as d:
            local = os.path.join(d, "out.parquet")
            t0 = time.perf_counter()
            index_table(part, local, opts.column, k, row_group_size=opts.row_group_size,
                        compress_embedding=opts.compress_embedding)
            rec["index_s"] = time.perf_counter() - t0
            md = pq.read_metadata(local)
            with open(local, "rb") as fh:
                pf_src = pa.PythonFile(fh, mode="r")
                blob = read_index(pf_src, md)
                runs = read_runs(pf_src, md)
            assert md.num_rows == part.num_rows and blob.n_rows == part.num_rows
            assert blob.n_clusters == k
            assert (runs is None) == opts.compress_embedding
            names = md.schema.to_arrow_schema().names
            for f in fields:                         # add_files infers partitions from these
                st = md.row_group(0).column(names.index(f.column)).statistics
                assert st is not None and st.has_min_max, f"no statistics on {f.column}"
            t0 = time.perf_counter()
            with open(local, "rb") as fh, dest_io.new_output(dest).create(overwrite=True) as o:
                while chunk := fh.read(64 << 20):
                    o.write(chunk)
            rec.update(upload_s=time.perf_counter() - t0, clusters=k,
                       row_groups=md.num_row_groups, mb=os.path.getsize(local) / 1e6)
        out.append(rec)
    return out


def _expand(source, io) -> list[str]:
    paths = [source] if isinstance(source, str) else list(source)
    out = []
    for p in paths:
        if "*" not in p:
            out.append(p)
        elif "://" not in p or p.startswith("file://"):
            out.extend(sorted(_glob.glob(p.removeprefix("file://"))))
        else:
            import pyarrow.fs as pafs
            fs, root = pafs.FileSystem.from_uri(p.split("*", 1)[0].rsplit("/", 1)[0] + "/")
            scheme = p.split("://", 1)[0]
            rx = re.compile("^" + re.escape(p.split("://", 1)[1]).replace(r"\*", "[^/]*") + "$")
            for info in fs.get_file_info(pafs.FileSelector(root, recursive=True)):
                if rx.match(info.path):
                    out.append(f"{scheme}://{info.path}")
            out.sort()
    if not out:
        raise FileNotFoundError(f"no input files match {source!r}")
    return out


def _fields_from_spec(tbl) -> list[PartitionField]:
    """The table's own partition spec as PartitionFields (identity/truncate only)."""
    from pyiceberg.transforms import IdentityTransform, TruncateTransform
    out = []
    for pf in tbl.spec().fields:
        col = tbl.schema().find_field(pf.source_id).name
        if isinstance(pf.transform, IdentityTransform):
            out.append(PartitionField(col))
        elif isinstance(pf.transform, TruncateTransform):
            out.append(PartitionField(col, pf.transform.width))
        else:
            raise ValueError(f"partition transform {pf.transform} is not supported: add_files "
                             "can only infer identity and truncate partitions")
    return out


def _ensure_table(catalog, identifier, schema: pa.Schema, fields, properties):
    from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
    from pyiceberg.transforms import IdentityTransform, TruncateTransform
    try:
        return catalog.load_table(identifier), False
    except NoSuchTableError:
        pass
    try:
        tbl = catalog.create_table(identifier, schema=schema, properties=properties)
    except NoSuchNamespaceError:
        parts = identifier.split(".") if isinstance(identifier, str) else list(identifier)
        catalog.create_namespace(tuple(parts[:-1]))
        tbl = catalog.create_table(identifier, schema=schema, properties=properties)
    if fields:
        with tbl.update_spec() as spec:
            for f in fields:
                tf = IdentityTransform() if f.width is None else TruncateTransform(f.width)
                spec.add_field(f.column, tf, f.name)
        tbl = catalog.load_table(identifier)
    return tbl, True


def write_indexed_table(catalog, identifier: str, source, column: str, *,
                        partition_by=(), n_clusters: int = 256, row_group_size: int = 512,
                        compress_embedding: bool = True, max_rows_per_file: int = 500_000,
                        runner: str = "local",
                        max_workers: int = 4, ray_address: str | None = None,
                        ray_cpus_per_task: float = 2, ray_memory_gb: float = 6,
                        properties: dict | None = None, log=print):
    """Index `source` and register it as `identifier` (created if missing).

    source: a Parquet path, glob or list of them (anything the catalog's
            FileIO reads), or a pyiceberg Table to re-index as a new table.
    partition_by: ["col", "col:truncate[N]", ...] — applied only when the
            table is created; an existing table keeps its spec.
    runner: "local" (thread pool, `max_workers`) or "ray" (one task per input
            file on an existing or local Ray cluster).
    """
    from pyiceberg.table import Table

    fields = [PartitionField.parse(s) for s in partition_by]
    opts = IndexOptions(column=column, n_clusters=n_clusters, row_group_size=row_group_size,
                        compress_embedding=compress_embedding, max_rows_per_file=max_rows_per_file)
    if isinstance(source, Table):
        src_io = source.io
        inputs = sorted(t.file.file_path for t in source.scan().plan_files())
    else:
        from pyiceberg.io.pyarrow import PyArrowFileIO
        src_io = PyArrowFileIO(dict(catalog.properties))
        inputs = _expand(source, src_io)
    with src_io.new_input(inputs[0]).open() as f:
        schema = pq.read_schema(f).remove_metadata()
    if column not in schema.names:
        raise KeyError(f"embedding column {column!r} not in {schema.names}")

    props = {"lakehouse_ann.n_clusters": str(n_clusters), "lakehouse_ann.row_group_size": str(row_group_size),
             "write.parquet.compression-codec": "zstd", **(properties or {})}
    dest, created = _ensure_table(catalog, identifier, schema, fields, props)
    if not created:                      # an existing table keeps its own spec
        have = _fields_from_spec(dest)
        if fields and fields != have:
            raise ValueError(f"{identifier} is partitioned {have}, not {fields}")
        fields = have
    log(f"{identifier}: {'created' if created else 'exists'} at {dest.metadata.location}; "
        f"{len(inputs)} input file(s), runner={runner}")
    prefix = dest.metadata.location.rstrip("/")
    t_all = time.perf_counter()

    done = []
    def report(recs):
        for r in recs:
            done.append(r)
            what = "skip " if r.get("skipped") else "built"
            log(f"  [{len(done)}] {what} {r['rows']:>9,} rows {_partition_path(fields, r['partition'])} "
                f"read {r['read_s']:.0f}s index {r.get('index_s', 0):.0f}s "
                f"upload {r.get('upload_s', 0):.0f}s (wall {time.perf_counter() - t_all:.0f}s)")

    if runner == "ray":
        import ray
        if not ray.is_initialized():
            ray.init(address=ray_address)
        task = ray.remote(num_cpus=ray_cpus_per_task, memory=int(ray_memory_gb * 1024**3),
                          max_retries=2)(index_one)
        pending = [task.remote(p, src_io, dest.io, prefix, fields, opts) for p in inputs]
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            report(ray.get(ready[0]))
    elif runner == "local":
        with ThreadPoolExecutor(max_workers) as ex:
            for recs in ex.map(lambda p: index_one(p, src_io, dest.io, prefix, fields, opts), inputs):
                report(recs)
    else:
        raise ValueError(f"runner must be 'local' or 'ray', not {runner!r}")

    registered = {t.file.file_path for t in dest.scan().plan_files()}
    fresh = [r["dest"] for r in done if r["dest"] not in registered]
    if fresh:
        import logging                    # the embedding column has no stats, by design
        lg = logging.getLogger("pyiceberg.io.pyarrow")
        quiet = logging.Filter(); quiet.filter = lambda r: "statistics missing" not in r.getMessage()
        lg.addFilter(quiet)
        try:
            dest.add_files(fresh)
        finally:
            lg.removeFilter(quiet)
    dest = catalog.load_table(identifier)
    n_rows = sum(t.file.record_count for t in dest.scan().plan_files())
    log(f"{identifier}: +{len(fresh)} file(s) registered; {n_rows:,} rows @ snapshot "
        f"{dest.metadata.current_snapshot_id} ({time.perf_counter() - t_all:.0f}s)")
    return dest


# ── CLI ─────────────────────────────────────────────────────────────────────

def _runner(a):
    import daft
    if a.ray is not None:
        daft.set_runner_ray(address=a.ray or None)
        return "ray"
    daft.set_runner_native()
    return "local"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m lakehouse_ann.write",
        description="embed text → Parquet, then index + register as an Iceberg table. "
                    "Catalogs come from pyiceberg's config (~/.pyiceberg.yaml or "
                    "PYICEBERG_CATALOG__<NAME>__* env vars).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="compute embeddings for a text column, write Parquet")
    e.add_argument("inputs", nargs="+", help="Parquet/CSV/JSON paths or globs (anything Daft reads)")
    e.add_argument("--text-col", required=True)
    e.add_argument("--out", required=True, help="output directory/URI for Parquet")
    e.add_argument("--column", default="embedding", help="name of the embedding column")
    e.add_argument("--model", default="ibm-granite/granite-embedding-97m-multilingual-r2")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--gpus", type=float, default=None, help="GPUs per embedding worker")
    e.add_argument("--concurrency", type=int, default=None, help="embedding workers")
    e.add_argument("--rows-per-file", type=int, default=250_000)
    e.add_argument("--ray", nargs="?", const="", default=None, metavar="ADDRESS",
                   help="run on Ray (local cluster if no address)")

    i = sub.add_parser("index", help="index Parquet files (or a table) and register them")
    src = i.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", nargs="+", help="Parquet paths/globs")
    src.add_argument("--from-table", help="existing Iceberg table to re-index into --dest")
    i.add_argument("--dest", required=True, help="destination table, e.g. vec.docs")
    i.add_argument("--catalog", default="default", help="pyiceberg catalog name")
    i.add_argument("--column", default="embedding")
    i.add_argument("--partition-by", action="append", default=[],
                   help="col or col:truncate[N] (repeatable)")
    i.add_argument("--n-clusters", type=int, default=256)
    i.add_argument("--row-group-size", type=int, default=512)
    i.add_argument("--no-compress-embedding", action="store_true",
                   help="write the embedding PLAIN+uncompressed (adds IFVR byte runs)")
    i.add_argument("--max-rows-per-file", type=int, default=500_000)
    i.add_argument("--workers", type=int, default=4, help="threads for the local runner")
    i.add_argument("--ray", nargs="?", const="", default=None, metavar="ADDRESS",
                   help="one Ray task per input file (local cluster if no address)")
    a = ap.parse_args(argv)

    if a.cmd == "embed":
        from .embed import embed_files
        _runner(a)
        paths = embed_files(a.inputs, a.text_col, a.out, column=a.column, model=a.model,
                            batch_size=a.batch_size, num_gpus=a.gpus,
                            concurrency=a.concurrency, rows_per_file=a.rows_per_file)
        print(f"wrote {len(paths)} file(s) under {a.out}")
        return

    from pyiceberg.catalog import load_catalog
    cat = load_catalog(a.catalog)
    source = cat.load_table(a.from_table) if a.from_table else a.input
    runner = "local"
    if a.ray is not None:
        import ray
        ray.init(address=a.ray or None)
        runner = "ray"
    write_indexed_table(cat, a.dest, source, a.column, partition_by=a.partition_by,
                        n_clusters=a.n_clusters, row_group_size=a.row_group_size,
                        compress_embedding=not a.no_compress_embedding,
                        max_rows_per_file=a.max_rows_per_file, runner=runner,
                        max_workers=a.workers)


if __name__ == "__main__":
    sys.exit(main())
