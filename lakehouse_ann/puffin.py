"""Per-file vector indexes stored in Iceberg Puffin files.

Instead of splicing the index into each Parquet data file (see footer.py), the
same IFVI blob can live in a Puffin file next to the data. Puffin is Iceberg's
container for statistics and indexes (v3 deletion vectors use it the same way:
one blob per data file). This lets an index be added to or rebuilt for an
existing table without rewriting a single data file, for tables written by
any engine.

Puffin file (spec: https://iceberg.apache.org/puffin-spec/):

    Magic Blob1 Blob2 ... Footer
    Footer = Magic FooterPayload FooterPayloadSize(int32 LE) Flags(4 bytes) Magic
    Magic  = b"PFA1";  FooterPayload = JSON {"blobs": [...], "properties": {...}}

Blobs written here (one pair per indexed data file):

    lakehouse-ann-ivf-v1       the IFVI v1 index (format.py)
    lakehouse-ann-ivf-runs-v1  optional IFVR byte runs into the data file, so
                               single vectors can be fetched with ranged reads

each with properties `referenced-data-file`, `data-file-size-in-bytes`,
`record-count`, `metric`. Because the index stores row positions within one
exact data file, a blob is only used when the planned file's path, size and
row count all match; anything else falls back to the footer index or to an
exact scan, so a stale registry can never return wrong rows.

Registry: today's Iceberg spec has no per-data-file index pointer, and
registering custom blob types as table statistics breaks readers that only
accept the known statistics blob types (e.g. pyiceberg 0.12). So the table
property `lakehouse_ann.index.<column>` holds a JSON list of Puffin paths;
every engine ignores unknown properties. A spec-level pointer (like deletion
vectors' manifest fields) would replace it.
"""
from __future__ import annotations

import dataclasses
import json
import struct
import threading

MAGIC = b"PFA1"
IVF_BLOB = "lakehouse-ann-ivf-v1"
RUNS_BLOB = "lakehouse-ann-ivf-runs-v1"
PROPERTY_PREFIX = "lakehouse_ann.index."
CREATED_BY = "lakehouse-ann"


@dataclasses.dataclass(frozen=True)
class IndexRef:
    """Where one data file's index lives inside a Puffin file."""
    puffin: str
    offset: int
    length: int
    runs_offset: int | None
    runs_length: int | None
    data_file: str
    data_file_size: int
    record_count: int

    def matches(self, data_file) -> bool:
        """True if this ref was built for exactly this pyiceberg DataFile."""
        return (self.data_file == data_file.file_path
                and self.data_file_size == data_file.file_size_in_bytes
                and self.record_count == data_file.record_count)


# ── writing ──────────────────────────────────────────────────────────────────

def build_puffin(entries, *, field_id: int, snapshot_id: int, sequence_number: int,
                 metric: str = "l2") -> bytes:
    """Puffin bytes holding one IVF blob (+ optional runs blob) per entry.

    entries: iterable of dicts with keys data_file, size, rows, ivf (bytes),
    runs (bytes or None).
    """
    buf = bytearray(MAGIC)
    blobs = []

    def add(kind, payload, e):
        blobs.append({
            "type": kind, "fields": [field_id], "snapshot-id": snapshot_id,
            "sequence-number": sequence_number, "offset": len(buf), "length": len(payload),
            "properties": {"referenced-data-file": e["data_file"],
                           "data-file-size-in-bytes": str(e["size"]),
                           "record-count": str(e["rows"]), "metric": metric},
        })
        buf.extend(payload)

    for e in entries:
        add(IVF_BLOB, e["ivf"], e)
        if e.get("runs"):
            add(RUNS_BLOB, e["runs"], e)
    payload = json.dumps({"blobs": blobs, "properties": {"created-by": CREATED_BY}},
                         separators=(",", ":")).encode()
    buf.extend(MAGIC + payload + struct.pack("<i", len(payload)) + b"\0\0\0\0" + MAGIC)
    return bytes(buf)


def write_puffin(io, path: str, entries, **kw) -> int:
    data = build_puffin(entries, **kw)
    with io.new_output(path).create(overwrite=True) as out:
        out.write(data)
    return len(data)


# ── reading ──────────────────────────────────────────────────────────────────

_FOOTERS: dict[str, list[dict]] = {}       # Puffin files are immutable: cache by path
_FOOTER_LOCK = threading.Lock()


def _read_at(f, offset: int, length: int) -> bytes:
    b = f.read_at(length, offset)
    if len(b) != length:
        raise IOError(f"short read: {len(b)} of {length} bytes at {offset}")
    return b if isinstance(b, bytes) else b.to_pybytes()


def read_footer(io, path: str) -> list[dict]:
    """The blob list of a Puffin file, with two ranged reads (tail, footer)."""
    with _FOOTER_LOCK:
        if path in _FOOTERS:
            return _FOOTERS[path]
    with io.new_input(path).open() as f:
        size = f.size()
        tail = _read_at(f, size - 12, 12)
        if tail[-4:] != MAGIC:
            raise ValueError(f"{path}: not a Puffin file")
        n = struct.unpack("<i", tail[:4])[0]
        if tail[4] & 1:
            raise ValueError(f"{path}: compressed Puffin footers are not supported")
        head = _read_at(f, size - 12 - n - 4, n + 4)
        if head[:4] != MAGIC:
            raise ValueError(f"{path}: bad Puffin footer")
        blobs = json.loads(head[4:])["blobs"]
    with _FOOTER_LOCK:
        _FOOTERS[path] = blobs
    return blobs


def registry_key(column: str) -> str:
    return PROPERTY_PREFIX + column


def registered_puffins(tbl, column: str) -> list[str]:
    raw = tbl.properties.get(registry_key(column))
    return json.loads(raw) if raw else []


def index_refs(tbl, column: str) -> dict[str, IndexRef]:
    """data file path -> IndexRef, from every registered Puffin file (later
    files win, so a rebuild simply supersedes the older blob)."""
    refs: dict[str, IndexRef] = {}
    for path in registered_puffins(tbl, column):
        found: dict[str, dict] = {}
        for b in read_footer(tbl.io, path):
            props = b.get("properties", {})
            df = props.get("referenced-data-file")
            if df and b["type"] in (IVF_BLOB, RUNS_BLOB):
                found.setdefault(df, {"props": props})[b["type"]] = b
        for df, d in found.items():
            if IVF_BLOB not in d:
                continue
            ivf, runs = d[IVF_BLOB], d.get(RUNS_BLOB)
            refs[df] = IndexRef(
                puffin=path, offset=ivf["offset"], length=ivf["length"],
                runs_offset=runs["offset"] if runs else None,
                runs_length=runs["length"] if runs else None,
                data_file=df, data_file_size=int(d["props"]["data-file-size-in-bytes"]),
                record_count=int(d["props"]["record-count"]))
    return refs


def ref_for(refs: dict[str, IndexRef], data_file) -> IndexRef | None:
    r = refs.get(data_file.file_path)
    return r if r is not None and r.matches(data_file) else None


def read_index_blobs(io, ref: IndexRef):
    """(IndexBlob, ValueRuns | None, bytes read, requests) for one ref."""
    from .format import parse, parse_runs
    with io.new_input(ref.puffin).open() as f:
        blob = parse(_read_at(f, ref.offset, ref.length))
        runs, n, reqs = None, ref.length, 1
        if ref.runs_offset is not None:
            runs = parse_runs(_read_at(f, ref.runs_offset, ref.runs_length))
            n, reqs = n + ref.runs_length, 2
    return blob, runs, n, reqs


def register(tbl, column: str, paths: list[str], tx=None):
    """Append Puffin `paths` to the table's registry (in `tx` if given)."""
    new = json.dumps(registered_puffins(tbl, column) + list(paths))
    if tx is not None:
        tx.set_properties({registry_key(column): new})
    else:
        with tbl.transaction() as t:
            t.set_properties({registry_key(column): new})
