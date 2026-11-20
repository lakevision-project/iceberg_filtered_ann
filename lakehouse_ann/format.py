"""IFVI v1 — the per-file IVF index blob.

A self-contained binary blob stored inside a Parquet file, between the data
pages and the footer (see footer.py). Any reader that implements this layout
can search the file; every other Parquet reader ignores the blob.

Layout (all integers little-endian):

    0   magic        b"IFVI"
    4   version      u8   (= 1)
    5   metric       u8   (0 = L2)
    6   flags        u16  (bit0: row ids are u64, else u32;
                           bit1: whole-file centroid + radius present)
    8   dim          u32
    12  n_clusters   u32
    16  n_rows       u64
    24  created_at   i64  (unix millis)
    32  seed         u64
    40  centroids    n_clusters * dim * f32, row-major
    ..  offsets      (n_clusters + 1) * u64, relative to posting-data start
    ..  posting data concatenated file-local row ordinals, ascending per cluster
    ..  file geom    IFF flags bit1: centroid dim * f32, radius f32
    ..  crc32        u32 over every preceding byte
"""
from __future__ import annotations

import dataclasses
import struct
import time
import zlib

import numpy as np

MAGIC = b"IFVI"
VERSION_V1 = 1
METRIC_L2 = 0
HEADER_SIZE = 40
FLAG_ROW_IDS_I64 = 1 << 0
FLAG_HAS_FILE_GEOM = 1 << 1

# Footer key-value entry that points at the blob: {"offset": "...", "length": "..."}.
POINTER_KEY = "ifs.vector.index.v1"

_HEADER = struct.Struct("<4sBBHIIQqQ")
assert _HEADER.size == HEADER_SIZE


class FormatError(Exception):
    """The blob is malformed or uses a version/metric we do not understand."""


@dataclasses.dataclass
class IndexBlob:
    dim: int
    n_rows: int
    centroids: np.ndarray            # (n_clusters, dim) float32
    offsets: np.ndarray              # (n_clusters + 1,) uint64, in ROWS (not bytes)
    postings: np.ndarray             # (n_rows,) uint64 — every cluster's ids, concatenated
    file_geom: tuple[np.ndarray, float] | None = None
    seed: int = 0
    created_at_ms: int = 0

    @property
    def n_clusters(self) -> int:
        return self.centroids.shape[0]

    def posting(self, c: int) -> np.ndarray:
        return self.postings[int(self.offsets[c]):int(self.offsets[c + 1])]

    def cluster_sizes(self) -> np.ndarray:
        return np.diff(self.offsets)


def serialize(blob: IndexBlob) -> bytes:
    wide = blob.n_rows > np.iinfo(np.uint32).max
    flags = (FLAG_ROW_IDS_I64 if wide else 0) | (FLAG_HAS_FILE_GEOM if blob.file_geom else 0)
    id_dtype = "<u8" if wide else "<u4"
    id_bytes = 8 if wide else 4
    created = blob.created_at_ms or int(time.time() * 1000)
    parts = [
        _HEADER.pack(MAGIC, VERSION_V1, METRIC_L2, flags, blob.dim, blob.n_clusters,
                     blob.n_rows, created, blob.seed),
        np.ascontiguousarray(blob.centroids, dtype="<f4").tobytes(),
        (blob.offsets.astype(np.uint64) * id_bytes).astype("<u8").tobytes(),
        blob.postings.astype(id_dtype).tobytes(),
    ]
    if blob.file_geom:
        c, r = blob.file_geom
        parts += [np.asarray(c, dtype="<f4").tobytes(), struct.pack("<f", r)]
    body = b"".join(parts)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def parse(buf: bytes) -> IndexBlob:
    if len(buf) < HEADER_SIZE + 4:
        raise FormatError(f"blob too short: {len(buf)} bytes")
    magic, version, metric, flags, dim, n_clusters, n_rows, created, seed = \
        _HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise FormatError(f"bad magic {magic!r}")
    if version != VERSION_V1:
        raise FormatError(f"unsupported version {version}")
    if metric != METRIC_L2:
        raise FormatError(f"unsupported metric {metric}")
    if flags & ~(FLAG_ROW_IDS_I64 | FLAG_HAS_FILE_GEOM):
        raise FormatError(f"unknown flag bits {flags:#x}")

    crc_at = len(buf) - 4
    (crc,) = struct.unpack_from("<I", buf, crc_at)
    if zlib.crc32(buf[:crc_at]) & 0xFFFFFFFF != crc:
        raise FormatError("CRC mismatch")

    wide = bool(flags & FLAG_ROW_IDS_I64)
    id_bytes = 8 if wide else 4
    pos = HEADER_SIZE
    centroids = np.frombuffer(buf, "<f4", n_clusters * dim, pos).reshape(n_clusters, dim)
    pos += n_clusters * dim * 4
    byte_offsets = np.frombuffer(buf, "<u8", n_clusters + 1, pos)
    pos += (n_clusters + 1) * 8
    if np.any(np.diff(byte_offsets.astype(np.int64)) < 0) or byte_offsets[-1] % id_bytes:
        raise FormatError("posting offsets not monotonic / not aligned")
    n_ids = int(byte_offsets[-1]) // id_bytes
    postings = np.frombuffer(buf, "<u8" if wide else "<u4", n_ids, pos).astype(np.uint64)
    pos += n_ids * id_bytes

    file_geom = None
    if flags & FLAG_HAS_FILE_GEOM:
        c = np.frombuffer(buf, "<f4", dim, pos)
        (r,) = struct.unpack_from("<f", buf, pos + dim * 4)
        pos += dim * 4 + 4
        # A non-finite bound would prune wrongly; dropping an optional hint is safe.
        if np.all(np.isfinite(c)) and np.isfinite(r) and r >= 0:
            file_geom = (c, float(r))
    if pos != crc_at:
        raise FormatError(f"length mismatch: parsed {pos} bytes, blob has {crc_at}")

    return IndexBlob(dim=dim, n_rows=n_rows, centroids=centroids,
                     offsets=byte_offsets // id_bytes, postings=postings,
                     file_geom=file_geom, seed=seed, created_at_ms=created)


# ── IFVR: where the embedding values physically sit ─────────────────────────
#
# Optional companion to IFVI, with its own pointer key: a reader that knows
# only IFVI ignores it and falls back to row-group reads. The embedding column is
# written PLAIN + uncompressed with no nulls, so inside every data page the
# vectors are one contiguous run of `dim * 4`-byte values. A "run" records the
# first row it holds and the absolute file offset of that row's first byte;
# row r then lives at  byte_offset[i] + (r - first_row[i]) * dim * 4.
#
#   0  magic b"IFVR" | 4 version u8 | 5 pad[3] | 8 dim u32 | 12 n_runs u32
#   16 first_row u64[n_runs] | .. byte_offset u64[n_runs] | .. crc32 u32

RUNS_POINTER_KEY = "ifs.vector.runs.v1"
RUNS_MAGIC = b"IFVR"
_RUNS_HEADER = struct.Struct("<4sB3xII")


@dataclasses.dataclass
class ValueRuns:
    dim: int
    first_row: np.ndarray      # (n_runs,) uint64, ascending, first_row[0] == 0
    byte_offset: np.ndarray    # (n_runs,) uint64, absolute offsets in the file

    def offsets_of(self, rows: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.first_row, rows, side="right") - 1
        return self.byte_offset[i] + (rows - self.first_row[i]) * np.uint64(self.dim * 4)


def serialize_runs(r: ValueRuns) -> bytes:
    body = (_RUNS_HEADER.pack(RUNS_MAGIC, 1, r.dim, len(r.first_row))
            + r.first_row.astype("<u8").tobytes() + r.byte_offset.astype("<u8").tobytes())
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def parse_runs(buf: bytes) -> ValueRuns:
    magic, version, dim, n = _RUNS_HEADER.unpack_from(buf, 0)
    if magic != RUNS_MAGIC or version != 1:
        raise FormatError("bad IFVR header")
    if len(buf) != _RUNS_HEADER.size + 16 * n + 4:
        raise FormatError("IFVR length mismatch")
    (crc,) = struct.unpack_from("<I", buf, len(buf) - 4)
    if zlib.crc32(buf[:-4]) & 0xFFFFFFFF != crc:
        raise FormatError("IFVR CRC mismatch")
    first = np.frombuffer(buf, "<u8", n, _RUNS_HEADER.size)
    off = np.frombuffer(buf, "<u8", n, _RUNS_HEADER.size + 8 * n)
    return ValueRuns(dim=dim, first_row=first.astype(np.uint64), byte_offset=off.astype(np.uint64))
