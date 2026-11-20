"""Write a small synthetic text corpus (Parquet) for trying the write path.

    python examples/sample_docs.py data/sample/docs.parquet --rows 20000

Columns: id, customer_id (0-999), category (5 values), text. Replace it with
your own Parquet/CSV/JSON: the write path only needs a text column (to embed)
or an embedding column (to index).
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TOPICS = {
    "tech": ["password reset", "GPU driver update", "database migration", "API rate limits", "laptop battery"],
    "finance": ["quarterly earnings", "interest rates", "bond yields", "currency markets", "retirement savings"],
    "news": ["city council vote", "election results", "weather warning", "transit strike", "new museum"],
    "science": ["vaccine trial", "exoplanet discovery", "climate model", "protein folding", "quantum sensor"],
    "sports": ["championship final", "transfer window", "marathon record", "injury update", "overtime win"],
}
FRAMES = ["How do I handle {t}?", "A short report on {t} this week.", "Questions people ask about {t}.",
          "What experts say about {t}.", "{t}: a beginner's guide.", "Latest developments in {t}."]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rows", type=int, default=20_000)
a = ap.parse_args()
rng = np.random.default_rng(0)
cats = rng.choice(list(TOPICS), a.rows)
text = [rng.choice(FRAMES).format(t=rng.choice(TOPICS[c])) + f" (note {i})" for i, c in enumerate(cats)]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
pq.write_table(pa.table({"id": np.arange(a.rows, dtype=np.int64),
                         "customer_id": rng.integers(0, 1000, a.rows).astype(np.int64),
                         "category": cats, "text": text}), a.out)
print(f"wrote {a.rows:,} rows to {a.out}")
