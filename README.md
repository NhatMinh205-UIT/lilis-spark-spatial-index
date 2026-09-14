# LiLIS: Reproducing a Distributed Learned Spatial Index on Apache Spark

Final-term project for **Dữ liệu lớn & Cơ sở dữ liệu phân tán** (IS405.Q22 & IS211.Q23), University of Information Technology, VNU-HCM — Group 6, 2026.

This project reproduces and experimentally evaluates **LiLIS** (arXiv:[2504.18883](https://arxiv.org/abs/2504.18883)), a lightweight distributed learned spatial index built on Apache Spark, proposed by the [SWUFE-DB-Group](https://swufe-db-group.github.io/learned-index-spark/). We deployed it on a **real 3-node Apache Spark Standalone cluster** (20 CPU cores over physical LAN, not simulated) and benchmarked it against traditional spatial indexing (Apache Sedona: R-tree, Quad-tree, KD-tree) on up to **100 million points**.

> This is a course reproduction of prior published research, not a novel system. The LiLIS algorithm design and the base Java implementation (partitioning strategies, learned-index build) originate from the paper authors' repository; our contribution was porting it to our own cluster and dependency versions, debugging real compatibility breaks, collecting and cleaning our own datasets, and running/analyzing a full benchmark suite. See [Team & contribution](#team--contribution) below for exactly who did what.

## Results summary

Measured on our own 3-node LAN cluster (not simulated), against 46.5M real NYC Yellow Taxi points:

| Query | LiLIS-K (learned index) | Sedona-N (no index, brute force) | Speedup |
|---|---:|---:|---:|
| Point Query | 780 ms | 7,032 ms | 9.0× |
| Range Query | 743 ms | 7,528 ms | 10.1× |
| kNN (k=10) | 566 ms | 5,643 ms | 10.0× |

Index build time: LiLIS-K built its index **1.6–2.5× faster** than the fastest Sedona baseline across all three datasets (CHI 7.7M / NYC 46.5M / SYN 100M rows). Full per-variant numbers are in [`lilis/results/`](lilis/results/) and [`sedona-baseline/results/`](sedona-baseline/results/).

We reproduced 3 of the paper's 5 core takeaways on our (smaller-scale, real-hardware) setup and found one extension the paper doesn't mention: on the CHI dataset, Adaptive-grid partitioning becomes competitive with KD-tree, unlike on NYC.

## Architecture

```
3-node Apache Spark Standalone cluster (LAN, WSL2/Ubuntu 22.04, Java 11, Spark 3.5.8)
 ├─ Master/Driver  — 4 cores / 4 GB
 ├─ Worker 1       — 8 cores / 8 GB
 └─ Worker 2       — 8 cores / 8 GB

Global partitioning (5 variants) → local learned index (Taut-String PLR spline, Y-sorted)
 Fixed-grid | Adaptive-grid | Quad-tree | KD-tree | R-tree
```

Getting all 3 machines to see each other over the physical LAN required switching WSL2 from NAT to **Mirrored Networking mode** (`.wslconfig`) — by default each WSL2 instance sits behind its own Hyper-V NAT and is unreachable from other machines on the network.

## Technical highlights worth knowing before you ask about this in an interview

- **Dependency pinning was not arbitrary.** `sedona-core` is pinned at `1.2.0-incubating` because that is the exact version whose API (`KDBTree`, `HalfOpenRectangle`, `QuadTreePartitioner`) the original authors' code targets — upgrading to 1.6.x breaks the API surface entirely. See the comments in [`lilis/pom.xml`](lilis/pom.xml).
- **Real compatibility break, fixed in code**: [`KDBTreePartitioner.java`](lilis/src/main/java/partitions/KDBTreePartitioner.java) documents fixing a class-rename between Sedona versions (`KDBTree` → `KDB`) that the original repo's code didn't account for.
- **Spark vs. bundled-dependency conflict**: an early build threw `ClassNotFoundException: org.locationtech.jts.geom.Envelope` because Spark's own JARs were being bundled and conflicting with the real runtime classpath — fixed by marking `spark-core`/`spark-sql` as `provided` scope while bundling `sedona-core` and `jts-core` into the fat JAR.
- **`AdaptiveGridPartitioner`** is fully custom (not calling into Sedona's built-in classes, unlike the KD-tree/Quad-tree/R-tree variants) — implements a custom Spark `Partitioner` and computes its grid step via a manual `aggregate()` combine/seq-op pass over the RDD.
- **`KDBTreePartitioner`** samples 1% of the data (paper spec) to estimate the spatial boundary before building the real KD-tree, then performs the actual physical shuffle via `flatMapToPair(...).partitionBy(...)`.
- Query execution ([`PointQuery.java`](lilis/src/main/java/query/PointQuery.java), [`RangeQuery.java`](lilis/src/main/java/query/RangeQuery.java)) is a two-stage filter: coarse-filter partitions by their bounding box (from local spline metadata) first, then fine-scan only the surviving partitions with `mapPartitions` — this is the actual mechanism behind the index's speedup over full scan.

## Team & contribution

2-person team. Per the project's own work-division table (also in the full academic report):

| Area | Nguyễn Nhật Minh | Nguyễn Văn Lê Hùng |
|---|---|---|
| Theory research | Spark/RDD architecture, shuffle mechanics; KD-tree/Quad-tree partitioning analysis | Learned-index theory (Spline/PLR); Space-filling curves (Z-order, Y-sort) |
| Data | Collected & merged NYC Yellow Taxi (4 CSVs); spatial distribution visualization | Collected/cleaned CHI (Chicago Crimes) and SYN (synthetic); Min-Max normalization |
| Infra | Master server + LAN config; fixed WSL2/Hyper-V NAT routing (Mirrored Networking) | Worker/executor setup; Apache Sedona baseline install & config |
| Implementation | Ported/debugged `SpatialPartition` (5 global-partitioning variants); Point/Range query benchmark scripts | Ported/debugged the Spline learned-index build; implemented kNN query (expanding-radius search) |
| Experiments | Ran LiLIS + Spark-Vanilla experiments; kNN/Range Query analysis vs. paper | Ran Sedona baseline experiments; index-build-cost analysis |
| Report | Chapters I, II, IV; presentation script & slides | Chapters III, V, VI; document formatting & data QA |

## Repository structure

```
lilis/               — LiLIS implementation (this repo's main code)
  src/main/java/      — partitioners, learned index, query engine
  src/test/java/      — parity tests (point/range/kNN query, spline build)
  scripts/            — benchmark shell scripts + Python result visualizers
  results/            — benchmark output (build times, query latency, correctness checks)
sedona-baseline/      — Apache Sedona baseline used for comparison
  src/main/java/baseline/
  results/
docs/figures/         — result charts referenced in the report
```

## Running it

Prerequisites: Java 11, Apache Spark 3.5.x (Standalone or local), Maven.

```bash
cd lilis
mvn clean package
# builds target/LearnIndexSpark-1.0-SNAPSHOT-jar-with-dependencies.jar
# entry point: idnexbuild.BuildAll

cd ../sedona-baseline
mvn clean package
# entry point: baseline.SedonaNBaseline
```

See [`lilis/scripts/run_benchmark.sh`](lilis/scripts/run_benchmark.sh) and [`run_query_benchmark.sh`](lilis/scripts/run_query_benchmark.sh) for the exact benchmark invocations used to produce the results in `results/` (15 runs/query, 2 warm-up runs, seed=42).

Note: this repo does not include the raw datasets (NYC Yellow Taxi ~7.9GB, CHI Crimes ~1.9GB) or the large per-run cluster logs — only the processed benchmark output. Datasets: [NYC Yellow Taxi (Kaggle)](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data), [Chicago Crimes](https://data.cityofchicago.org).

## Acknowledgment

Based on the LiLIS paper and reference implementation by the SWUFE-DB-Group: [swufe-db-group.github.io/learned-index-spark](https://swufe-db-group.github.io/learned-index-spark/) ([arXiv:2504.18883](https://arxiv.org/abs/2504.18883)).

## License

MIT — matching the original repository's license.
