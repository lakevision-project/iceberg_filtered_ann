# One image for the Ray head, the workers and the driver. Daft pickles scan
# tasks on the driver and unpickles them on workers, so Python, Daft, pyiceberg
# and lakehouse_ann must be identical everywhere — build once, run everywhere.
#
#   docker build -f deploy/kubernetes/Dockerfile -t <registry>/lakehouse-ann:0.1.0 .
#   docker build -f deploy/kubernetes/Dockerfile --build-arg EXTRAS=ray,embed -t ... .   # + embedding
FROM python:3.12-slim
ARG EXTRAS=ray
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY lakehouse_ann /app/lakehouse_ann
RUN pip install --no-cache-dir "/app[${EXTRAS}]"
# arbitrary-UID friendly (OpenShift): writable HOME, group-0 permissions
ENV HOME=/tmp PYTHONUNBUFFERED=1
RUN chgrp -R 0 /app && chmod -R g=u /app
USER 1001
