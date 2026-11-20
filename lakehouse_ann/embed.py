"""Compute text embeddings with Daft — the first stage of the write path.

    import daft
    from lakehouse_ann.embed import embed_text

    df = daft.read_parquet("docs/*.parquet")
    df = embed_text(df, "text", model="ibm-granite/granite-embedding-97m-multilingual-r2")
    df.write_parquet("staging/")          # then: lakehouse_ann.write.write_indexed_table

The model runs inside a Daft class UDF, so it is loaded once per worker and
fed batches. The same code runs on Daft's native runner (one machine) or on
Ray (`daft.set_runner_ray()`), where `concurrency` workers each hold a model
copy and `num_gpus` places them on GPUs.

Vectors are L2-normalized by default: the index ranks by L2 distance, which
for unit vectors orders results exactly like cosine similarity.

Needs `sentence-transformers` (and torch) on every worker that embeds.
"""
from __future__ import annotations

DEFAULT_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"


def _require_st():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:            # pragma: no cover
        raise ImportError("embedding needs sentence-transformers: "
                          "pip install -r requirements-embed.txt") from e


def model_dim(model: str = DEFAULT_MODEL) -> int:
    _require_st()
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    get = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return get()


def embed_text(df, text_col: str, out_col: str = "embedding", model: str = DEFAULT_MODEL, *,
               dim: int | None = None, batch_size: int = 64, num_gpus: float | None = None,
               concurrency: int | None = None, normalize: bool = True,
               max_seq_length: int = 512):
    """Add `out_col` (FixedSizeList<float32, dim>) holding the embedding of `text_col`.

    `dim` is looked up from the model when not given (loads it once on the driver).
    """
    import daft
    from daft import DataType

    _require_st()
    dim = dim or model_dim(model)
    ret = DataType.fixed_size_list(DataType.float32(), dim)

    @daft.cls(gpus=num_gpus or 0, max_concurrency=concurrency, name_override="embed_text")
    class _Embedder:
        def __init__(self, name: str):
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(name)
            self.m.max_seq_length = max_seq_length

        @daft.method.batch(return_dtype=ret, batch_size=batch_size)
        def encode(self, texts: daft.Series):
            import numpy as np
            import pyarrow as pa
            v = self.m.encode([t or "" for t in texts.to_pylist()], batch_size=batch_size,
                              normalize_embeddings=normalize, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)
            return pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), dim)

    return df.with_column(out_col, _Embedder(model).encode(df[text_col]))


def embed_files(inputs, text_col: str, out: str, *, column: str = "embedding",
                model: str = DEFAULT_MODEL, rows_per_file: int = 250_000, **kw) -> list[str]:
    """Read `inputs` (Parquet/CSV/JSON paths or globs), embed `text_col`, write
    Parquet under `out`. Uses whichever Daft runner is set. On Ray the output is
    split into files of about `rows_per_file` rows; the native runner cannot
    repartition, and the indexer splits oversized files itself
    (`write_indexed_table(max_rows_per_file=...)`)."""
    import math

    import daft

    paths = [inputs] if isinstance(inputs, str) else list(inputs)
    readers = {".csv": daft.read_csv, ".json": daft.read_json, ".jsonl": daft.read_json}
    ext = "." + paths[0].rsplit(".", 1)[-1].lower() if "." in paths[0] else ".parquet"
    df = readers.get(ext, daft.read_parquet)(paths)
    n = df.count_rows()
    if daft.get_or_create_runner().name == "ray":
        df = df.into_partitions(max(1, math.ceil(n / rows_per_file)))
    df = embed_text(df, text_col, column, model, **kw)
    written = df.write_parquet(out, compression="zstd")
    return written.to_pydict()["path"]
