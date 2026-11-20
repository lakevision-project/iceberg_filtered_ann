"""Put the index blobs inside a Parquet file, and get them back out.

Layout of an indexed file:

    PAR1 [data pages] [IFVI blob] [IFVR blob] [FileMetaData] [footer_len:4] PAR1
                                               KV  ifs.vector.index.v1 = {"offset", "length"}
                                               KV  ifs.vector.runs.v1  = {"offset", "length"}

A normal Parquet reader seeks to the end, reads FileMetaData, and never sees
the blobs. An index-aware reader looks up the KV pointers and does one ranged
read each. (A writer may also leave dead bytes, such as an earlier footer, in
front of a blob; readers only follow the pointers, so such files are valid too.)

Writing needs no Thrift surgery: pyarrow is deterministic, so writing the same
table twice — the second time with the pointers in the KV metadata — produces
byte-identical data pages and a footer that differs only by those entries. We
splice [data pages] + blobs + [footer of write 2], and assert the data pages
really are identical.
"""
from __future__ import annotations

import io
import json
import struct

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .format import (POINTER_KEY, RUNS_POINTER_KEY, IndexBlob, ValueRuns,
                     parse, parse_runs, serialize, serialize_runs)


def _embedding_keys(column: str) -> list[str]:
    """pyarrow keys per-column writer options by the parquet LEAF path, which
    for a list column `emb` is `emb.list.element` (compliant nested naming).
    Keys that match no column are ignored, so passing both spellings covers
    list, large_list and fixed_size_list alike."""
    return [column, f"{column}.list.element"]


def writer_options(table: pa.Table, column: str, row_group_size: int,
                    *, compress_embedding: bool = False) -> dict:
    """Parquet settings for the embedding column: uncompressed PLAIN, so each
    vector is a fixed-size byte range a reader can fetch on its own. Every
    other column keeps normal compression, dictionaries and statistics.

    With `compress_embedding` the column is written zstd like the rest.
    Compressed bytes cannot be needle-verified, so the file then gets an
    IFVI blob but no IFVR runs — readers fall back to row-group fetches,
    which stay cheap because the writer sorts rows by cluster and the row
    group size is chosen so a probe spans few of them."""
    others = [n for n in table.column_names if n != column]
    compression = {n: "zstd" for n in others}
    codec = "zstd" if compress_embedding else "none"
    for key in _embedding_keys(column):
        compression[key] = codec
    return dict(
        row_group_size=row_group_size,
        compression=compression,
        use_dictionary=others,
        column_encoding={column: "PLAIN"},
        write_statistics=others,
    )


def _write(table: pa.Table, opts: dict) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf, **opts)
    return buf.getvalue()


def _data_end(raw: bytes) -> int:
    assert raw[-4:] == b"PAR1"
    (footer_len,) = struct.unpack("<I", raw[-8:-4])
    return len(raw) - 8 - footer_len


def _leaf_index(md: pq.FileMetaData, column: str) -> int:
    rg = md.row_group(0)
    for j in range(rg.num_columns):
        p = rg.column(j).path_in_schema
        if p == column or p.startswith(column + "."):
            return j
    raise KeyError(column)


def locate_value_runs(raw: bytes, vectors: np.ndarray, column: str) -> ValueRuns | None:
    """Find where every vector's bytes sit in the written file.

    Walks each embedding column chunk: consecutive vectors are adjacent inside
    a page, and a page boundary shows up as a gap (page header + levels). Each
    run is verified byte-for-byte against the source vectors, so a wrong
    offset cannot be recorded — on any mismatch we return None and readers
    fall back to row-group reads.
    """
    md = pq.ParquetFile(io.BytesIO(raw)).metadata
    if md.num_row_groups == 0:
        return None
    j = _leaf_index(md, column)
    vb = vectors.shape[1] * 4
    vec_bytes = np.ascontiguousarray(vectors, dtype="<f4").view(np.uint8).reshape(len(vectors), vb)
    first_rows, offsets = [], []
    row = 0
    for g in range(md.num_row_groups):
        cc = md.row_group(g).column(j)
        if cc.compression != "UNCOMPRESSED" or "PLAIN" not in cc.encodings:
            return None
        start = cc.dictionary_page_offset or cc.data_page_offset
        end = start + cc.total_compressed_size
        pos = start
        for r in range(row, row + md.row_group(g).num_rows):
            needle = vec_bytes[r].tobytes()
            if raw[pos:pos + vb] != needle:            # page boundary: skip header + levels
                nxt = raw.find(needle, pos, end)
                if nxt < 0:
                    return None
                pos = nxt
                first_rows.append(r)
                offsets.append(pos)
            pos += vb
        row += md.row_group(g).num_rows
    runs = ValueRuns(dim=vectors.shape[1], first_row=np.asarray(first_rows, np.uint64),
                     byte_offset=np.asarray(offsets, np.uint64))
    # Defence in depth: re-derive every offset and compare against the bytes.
    offs = runs.offsets_of(np.arange(len(vectors), dtype=np.uint64))
    for r in range(0, len(vectors), max(1, len(vectors) // 997)):
        o = int(offs[r])
        if raw[o:o + vb] != vec_bytes[r].tobytes():
            return None
    return runs


def write_indexed_parquet(table: pa.Table, where: str, column: str, blob: IndexBlob,
                          vectors: np.ndarray, *, row_group_size: int = 8192,
                          filesystem: pafs.FileSystem | None = None,
                          compress_embedding: bool = False) -> int:
    """Write `table` to `where` with `blob` (and value runs, when the
    embedding column is uncompressed) embedded. Returns bytes written."""
    opts = writer_options(table, column, row_group_size, compress_embedding=compress_embedding)
    first = _write(table, opts)
    data_end = _data_end(first)
    blob_bytes = serialize(blob)
    runs = locate_value_runs(first, vectors, column)
    runs_bytes = serialize_runs(runs) if runs is not None else b""

    meta = dict(table.schema.metadata or {})
    meta[POINTER_KEY.encode()] = json.dumps(
        {"offset": str(data_end), "length": str(len(blob_bytes))}).encode()
    if runs_bytes:
        meta[RUNS_POINTER_KEY.encode()] = json.dumps(
            {"offset": str(data_end + len(blob_bytes)), "length": str(len(runs_bytes))}).encode()
    second = _write(table.replace_schema_metadata(meta), opts)
    if _data_end(second) != data_end or second[:data_end] != first[:data_end]:
        raise RuntimeError("pyarrow wrote different data pages on rewrite; refusing to splice")

    out = second[:data_end] + blob_bytes + runs_bytes + second[data_end:]
    fs, path = resolve(where, filesystem)
    with fs.open_output_stream(path) as f:
        f.write(out)
    return len(out)


def resolve(where: str, filesystem: pafs.FileSystem | None = None):
    if filesystem is not None:
        return filesystem, where
    if "://" in where:
        return pafs.FileSystem.from_uri(where)
    return pafs.LocalFileSystem(), where


def _pointer(metadata: pq.FileMetaData, key: str) -> tuple[int, int] | None:
    raw = (metadata.metadata or {}).get(key.encode())
    if raw is None:
        return None
    p = json.loads(raw)
    return int(p["offset"]), int(p["length"])


def _read_exact(source, offset: int, length: int) -> bytes:
    buf = source.read_at(length, offset)
    if len(buf) != length:
        raise IOError(f"short read: {len(buf)} of {length} bytes at {offset}")
    return buf if isinstance(buf, bytes) else buf.to_pybytes()


def read_index(source, metadata: pq.FileMetaData) -> IndexBlob | None:
    """Ranged-read and parse the IVF blob, or None if the file has no index.
    `source` is an open pyarrow RandomAccessFile."""
    ptr = _pointer(metadata, POINTER_KEY)
    return None if ptr is None else parse(_read_exact(source, *ptr))


def read_runs(source, metadata: pq.FileMetaData) -> ValueRuns | None:
    ptr = _pointer(metadata, RUNS_POINTER_KEY)
    return None if ptr is None else parse_runs(_read_exact(source, *ptr))
