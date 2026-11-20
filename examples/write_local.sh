#!/usr/bin/env bash
# Write path on ONE machine: embed text -> Parquet -> index + register an Iceberg table.
# Uses a local SQLite catalog under data/write_demo; point --catalog at any pyiceberg
# catalog (and --out/--input at object storage) for a real deployment.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/write_demo
export PYICEBERG_CATALOG__LOCAL__TYPE=sql
export PYICEBERG_CATALOG__LOCAL__URI="sqlite:///$PWD/$D/catalog.db"
export PYICEBERG_CATALOG__LOCAL__WAREHOUSE="file://$PWD/$D/warehouse"
mkdir -p "$D/warehouse"
INPUT=${1:-$D/raw/docs.parquet}
[ -f "$INPUT" ] || python examples/sample_docs.py "$INPUT" --rows 20000

# 1. embed (Daft native runner; the model is loaded once and fed batches)
python -m lakehouse_ann.write embed "$INPUT" --text-col text --out "$D/staging"

# 2. index + register: one indexed file per (input file x partition value),
#    k-means per file, rows sorted by cluster, one add_files() commit
python -m lakehouse_ann.write index --catalog local --dest vec.sample \
    --input "$D/staging/*.parquet" --partition-by category --n-clusters 64

# 3. query it
python - <<'PY'
import daft, numpy as np
from pyiceberg.catalog import load_catalog
from lakehouse_ann import ann_sql
from lakehouse_ann.embed import embed_text
tbl = load_catalog("local").load_table("vec.sample")
q = np.asarray(embed_text(daft.from_pydict({"t": ["my password does not work"]}), "t")
               .to_pydict()["embedding"][0], np.float32)
ann_sql(tbl, "embedding", q, k=5, sql="""SELECT text, category, _distance FROM sample
        WHERE category = 'tech' ORDER BY _distance LIMIT 5""").show()
PY
