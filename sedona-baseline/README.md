# Apache Sedona baseline

Baseline spatial-query implementation used to benchmark against the LiLIS learned index in [`../lilis`](../lilis). Five variants: `SedonaNBaseline` (no index, brute force), `SedonaQQBaseline`, `SedonaQKBaseline`, `SedonaRKBaseline`, `SedonaRQBaseline`, plus `SparkVanillaBaseline`.

This part of the project was primarily built by **Nguyễn Văn Lê Hùng** (Sedona setup/config, index-build-cost evaluation) — see the [root README's contribution table](../README.md#team--contribution) for the full breakdown.
