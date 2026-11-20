#!/usr/bin/env bash
# Write path DISTRIBUTED on Ray: same two commands, plus --ray.
#   examples/write_ray.sh                 # local Ray cluster on this machine
#   RAY_ADDRESS=auto examples/write_ray.sh  # from a node of a running cluster
# Embedding runs as a Daft class UDF on Ray workers (--gpus / --concurrency to
# place model copies); indexing runs one Ray task per staged file. On a real
# cluster, use a shared catalog and object-storage paths for --out / --input,
# since every worker must reach them.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/write_demo_ray
export PYICEBERG_CATALOG__LOCAL__TYPE=sql
export PYICEBERG_CATALOG__LOCAL__URI="sqlite:///$PWD/$D/catalog.db"
export PYICEBERG_CATALOG__LOCAL__WAREHOUSE="file://$PWD/$D/warehouse"
mkdir -p "$D/warehouse"
INPUT=${1:-$D/raw/docs.parquet}
[ -f "$INPUT" ] || python examples/sample_docs.py "$INPUT" --rows 20000
ADDR=${RAY_ADDRESS:-}

python -m lakehouse_ann.write embed "$INPUT" --text-col text --out "$D/staging" \
    --rows-per-file 5000 --ray $ADDR
python -m lakehouse_ann.write index --catalog local --dest vec.sample \
    --input "$D/staging/*.parquet" --partition-by "customer_id:truncate[250]" \
    --n-clusters 32 --ray $ADDR
python -c "
from pyiceberg.catalog import load_catalog
t = load_catalog('local').load_table('vec.sample')
fs = list(t.scan().plan_files())
print(f'vec.sample: {len(fs)} files, {sum(f.file.record_count for f in fs):,} rows, spec {t.spec()}')
print('customer_id < 250 plans', len(list(t.scan(row_filter=\"customer_id < 250\").plan_files())), 'files')"
