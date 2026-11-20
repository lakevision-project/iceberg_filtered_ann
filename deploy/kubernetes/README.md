# Ray on Kubernetes for lakehouse-ann

A plain Ray cluster (one head, N workers) as standard Kubernetes resources: a
Service and two Deployments. No operator is required, and it works on OpenShift
(`KUBECTL=oc`). If you run [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/),
a `RayCluster` with the same image is equivalent.

```bash
export NS=lakehouse-ann                    # namespace (must exist)
deploy/kubernetes/deploy.sh up             # render, apply, wait, print `ray status`
deploy/kubernetes/deploy.sh run bench/ray_bench.py --catalog prod --table vec.docs \
    --queries /tmp/q.npy --gt              # copy a script to the head and run it there
deploy/kubernetes/deploy.sh code           # push local lakehouse_ann changes, restart pods
deploy/kubernetes/deploy.sh down           # remove everything it created
```

Settings (environment): `NS`, `IMAGE` (default `python:3.12-slim`), `WORKERS` (5),
`WORKER_CPUS` (8), `WORKER_MEM` (32Gi), `KUBECTL` (kubectl).

## Getting the software onto the pods

Daft pickles each scan task on the driver and unpickles it on a worker, so
Python, Daft, pyiceberg and `lakehouse_ann` must be **identical on every pod**.
There are two ways to do that:

* **Build an image (recommended).**

  ```bash
  docker build -f deploy/kubernetes/Dockerfile -t <registry>/lakehouse-ann:0.1.0 .
  docker push <registry>/lakehouse-ann:0.1.0
  IMAGE=<registry>/lakehouse-ann:0.1.0 deploy/kubernetes/deploy.sh up
  ```

  Add `--build-arg EXTRAS=ray,embed` to include sentence-transformers for the
  embedding stage. Use a CUDA base image if you embed on GPUs.

* **No registry.** Leave `IMAGE` at a stock Python image. An init container
  pip-installs the pinned stack into a shared volume, and the package code is
  mounted from a ConfigMap (`deploy.sh` creates it from your checkout, and
  `deploy.sh code` refreshes it). Pods start a few minutes slower, but you
  need no registry access.

## Credentials

The manifests contain none. Configure the catalog on the driver the usual
pyiceberg way (`~/.pyiceberg.yaml` or `PYICEBERG_CATALOG__<NAME>__*`). The
table's FileIO, with its storage settings, travels inside every task, so
workers read object storage with the driver's configuration. Keep the driver
config in a Kubernetes Secret mounted on whichever pod runs the driver, usually
the head.

## Sizing notes

* The head runs with `--num-cpus=0`: it hosts the driver, and all scan,
  index and embedding tasks land on workers.
* Workers spread one per node (`topologySpreadConstraints`) where possible.
* Query tasks are I/O-bound. Index tasks request 2 CPUs and 6 GiB each
  (`write_indexed_table(ray_cpus_per_task=…, ray_memory_gb=…)`), so an
  8-vCPU / 32-GiB worker runs about 4 at a time.
* `/dev/shm` is a memory-backed volume for Ray's object store.
* Run the driver in the same region as the object store. In our measurements
  that mattered as much as the runner ([benchmarks §5](../../docs/benchmarks.md#5-native-vs-ray-runner-5m-rows)).
