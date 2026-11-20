"""Shared setup for the examples: the local demo catalog and a query vector.

Build the catalog once (synthetic 1M x 128 vectors, ~2 min):

    python bench/make_data.py --out data
    python bench/iceberg_demo.py
"""
import os
import sys
import warnings

import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", message=".*statistics missing.*")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run without installing
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "iceberg")


def catalog():
    from pyiceberg.catalog.sql import SqlCatalog
    if not os.path.exists(os.path.join(ROOT, "catalog.db")):
        raise SystemExit("demo catalog missing: run bench/make_data.py then bench/iceberg_demo.py")
    return SqlCatalog("demo", uri=f"sqlite:///{ROOT}/catalog.db", warehouse=f"file://{ROOT}/warehouse")


def query_vector(tbl, seed=123):
    """A row of the table plus noise — a realistic 'near something' query."""
    from lakehouse_ann.build import embedding_matrix
    first = next(iter(tbl.scan().plan_files())).file.file_path
    v = embedding_matrix(pq.read_table(first, columns=["emb"])["emb"])[0]
    return v + np.random.default_rng(seed).standard_normal(v.shape[0]).astype(np.float32) * 0.5
