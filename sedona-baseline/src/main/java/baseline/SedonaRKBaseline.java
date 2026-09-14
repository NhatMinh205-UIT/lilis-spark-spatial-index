package baseline;

import org.apache.sedona.common.enums.FileDataSplitter;
import org.apache.sedona.core.enums.GridType;
import org.apache.sedona.core.enums.IndexType;
import org.apache.sedona.core.spatialOperator.KNNQuery;
import org.apache.sedona.core.spatialOperator.RangeQuery;
import org.apache.sedona.core.spatialRDD.PointRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * SedonaRKBaseline — Apache Sedona R-tree + KD-tree (Sedona-RK)
 *
 * Sedona-RK = R-tree index + KD-tree (KDB-tree) partitioner
 * Table II paper:
 *   Sedona-RK: R-tree index + KD-tree partitioner ← file này
 *   Sedona-RQ: R-tree index + Quad-tree partitioner
 *   Sedona-QK: Quad-tree index + KD-tree partitioner
 *   Sedona-QQ: Quad-tree index + Quad-tree partitioner
 *
 * R-tree index HỖ TRỢ kNN (khác Sedona-QQ/QK)
 * → Chạy đầy đủ Point + Range + kNN
 *
 * QUAN TRỌNG — Sedona API:
 *   buildIndex(RTREE, false) → indexedRawRDD
 *   RangeQuery(useIndex=true) → đọc từ indexedRawRDD
 *   KNNQuery(useIndex=true)   → đọc từ indexedRawRDD
 *
 * Paper reference:
 *   Figure 4: Sedona-RK Range NYC ~521,282ms (tương đương Sedona-RQ)
 *   Figure 9: Sedona-RK build ~100s
 *   Table IV: Sedona-RK kNN CHI=7,862ms | NYC=790,993ms | SYN=83,170ms
 *
 * Args:
 *   args[0] = datasetLabel  (nyc_full)
 *   args[1] = dataPath      (/home/pc/.../nyc_full_xy.csv)
 *   args[2] = TOTAL_N       (46475157)
 *   args[3] = MIN_X         (-74.9996)
 *   args[4] = MAX_X         (-72.1965)
 *   args[5] = MIN_Y         (40.0081)
 *   args[6] = MAX_Y         (41.9237)
 */
public class SedonaRKBaseline {

    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final int    KNN_K       = 10;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;  // selectivity ~0.001%
    static final double POINT_EPS   = 1e-6;    // tiny box cho point query

    static final GeometryFactory GF = new GeometryFactory();

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: SedonaRKBaseline " +
                "<label> <path> <TOTAL_N> <MIN_X> <MAX_X> <MIN_Y> <MAX_Y>");
            System.exit(1);
        }

        String label = args[0];
        String path  = args[1];
        double minX  = Double.parseDouble(args[3]);
        double maxX  = Double.parseDouble(args[4]);
        double minY  = Double.parseDouble(args[5]);
        double maxY  = Double.parseDouble(args[6]);

        System.out.println("=".repeat(65));
        System.out.println("Sedona-RK (R-tree index + KD-tree partitioner)");
        System.out.printf("Dataset: %s%n", label);
        System.out.printf("Runs: %d | Warmup: %d | kNN k: %d%n",
                RUNS, WARMUP, KNN_K);
        System.out.printf("BBox: X[%.4f,%.4f] Y[%.4f,%.4f]%n",
                minX, maxX, minY, maxY);
        System.out.println("NOTE: R-tree HỖ TRỢ kNN ✅");
        System.out.println("=".repeat(65));

        // ── Init Spark ───────────────────────────────────────────────
        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // ── Bước 1: Tạo query points ngẫu nhiên ─────────────────────
        Random rng = new Random(SAMPLE_SEED);
        List<Point>    qPoints     = new ArrayList<>();
        List<Envelope> qRanges     = new ArrayList<>();
        List<Envelope> qPointBoxes = new ArrayList<>();

        for (int i = 0; i < RUNS + WARMUP; i++) {
            double cx = minX + rng.nextDouble() * (maxX - minX);
            double cy = minY + rng.nextDouble() * (maxY - minY);
            qPoints.add(GF.createPoint(new Coordinate(cx, cy)));
            qRanges.add(new Envelope(
                    cx - RANGE_HALF, cx + RANGE_HALF,
                    cy - RANGE_HALF, cy + RANGE_HALF));
            qPointBoxes.add(new Envelope(
                    cx - POINT_EPS, cx + POINT_EPS,
                    cy - POINT_EPS, cy + POINT_EPS));
        }
        System.out.printf("[Bước 1] %d query points (seed=%d)%n",
                qPoints.size(), SAMPLE_SEED);

        // ── Bước 2: Load PointRDD ────────────────────────────────────
        System.out.println("[Bước 2] Loading PointRDD...");
        PointRDD pointRDD = new PointRDD(
                sc, path, 0, FileDataSplitter.CSV, false);
        System.out.printf("[Bước 2] Raw partitions: %d%n",
                pointRDD.rawSpatialRDD.getNumPartitions());

        // ── Bước 3: Build Sedona-RK Index ───────────────────────────
        System.out.println("[Bước 3] Building Sedona-RK index...");
        System.out.println("  3a. analyze()");
        System.out.println("  3b. spatialPartitioning(KDBTREE) ← K partitioner");
        System.out.println("  3c. buildIndex(RTREE, false)      ← R-tree index");
        System.out.println("  NOTE: false → indexedRawRDD (required by RangeQuery API)");

        long buildStart = System.currentTimeMillis();

        // 3a. Tính bounding box
        pointRDD.analyze();

        // 3b. KD-tree spatial partitioning (KDBTREE trong Sedona API)
        pointRDD.spatialPartitioning(GridType.KDBTREE);

        // 3c. Build R-tree index — false = build on rawSpatialRDD
        //     (RangeQuery với useIndex=true cần indexedRawRDD)
        pointRDD.buildIndex(IndexType.RTREE, false);

        // Persist + trigger để force build execution
        pointRDD.indexedRawRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long indexCount = pointRDD.indexedRawRDD.count();

        long buildTime = System.currentTimeMillis() - buildStart;
        System.out.printf("[Bước 3] R-tree INDEX BUILT: %d partitions | TIME: %,dms (%.1fs)%n",
                indexCount, buildTime, buildTime / 1000.0);

        // ── Bước 4: Warm-up ──────────────────────────────────────────
        System.out.printf("[Warm-up] %d runs...%n", WARMUP);
        for (int w = 0; w < WARMUP; w++) {
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qPointBoxes.get(w), false, true).count();
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qRanges.get(w), false, true).count();
            KNNQuery.SpatialKnnQuery(
                    pointRDD, qPoints.get(w), KNN_K, true);
        }
        System.out.println("[Warm-up] Done.");

        // ── Bước 5: Point Query ──────────────────────────────────────
        System.out.printf("[Benchmark] Point Query — %d runs...%n", RUNS);
        long[] pointTimes = new long[RUNS];
        for (int i = 0; i < RUNS; i++) {
            long t0 = System.currentTimeMillis();
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qPointBoxes.get(i + WARMUP), false, true).count();
            pointTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[Point %2d/%d] %dms%n",
                    i + 1, RUNS, pointTimes[i]);
        }

        // ── Bước 6: Range Query ──────────────────────────────────────
        System.out.printf("[Benchmark] Range Query — %d runs...%n", RUNS);
        long[] rangeTimes = new long[RUNS];
        for (int i = 0; i < RUNS; i++) {
            long t0 = System.currentTimeMillis();
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qRanges.get(i + WARMUP), false, true).count();
            rangeTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[Range %2d/%d] %dms%n",
                    i + 1, RUNS, rangeTimes[i]);
        }

        // ── Bước 7: kNN Query (R-tree hỗ trợ kNN) ───────────────────
        System.out.printf("[Benchmark] kNN Query (k=%d) — %d runs...%n",
                KNN_K, RUNS);
        long[] knnTimes = new long[RUNS];
        for (int i = 0; i < RUNS; i++) {
            long t0 = System.currentTimeMillis();
            KNNQuery.SpatialKnnQuery(
                    pointRDD, qPoints.get(i + WARMUP), KNN_K, true);
            knnTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[kNN   %2d/%d] %dms%n",
                    i + 1, RUNS, knnTimes[i]);
        }

        sc.close();

        // ── Bước 8: Ghi kết quả ──────────────────────────────────────
        new java.io.File("results").mkdirs();
        String outFile = "results/" + label + "_sedona_rk_cluster_3nodes.txt";
        writeResults(outFile, label, buildTime, pointTimes, rangeTimes, knnTimes);
        System.out.println("Saved: " + outFile);
    }

    static double avg(long[] a) {
        long s = 0; for (long v : a) s += v; return (double) s / a.length;
    }
    static long min(long[] a) {
        long m = a[0]; for (long v : a) if (v < m) m = v; return m;
    }
    static long max(long[] a) {
        long m = a[0]; for (long v : a) if (v > m) m = v; return m;
    }
    static double std(long[] a) {
        double mean = avg(a), sq = 0;
        for (long v : a) sq += (v - mean) * (v - mean);
        return Math.sqrt(sq / a.length);
    }

    static void writeResults(String file, String label, long buildTime,
            long[] pt, long[] rng, long[] knn) throws IOException {

        StringBuilder sb = new StringBuilder();

        sb.append("=".repeat(70)).append("\n");
        sb.append("SEDONA-RK BASELINE — ").append(label).append("\n");
        sb.append("Index:       R-tree  (IndexType.RTREE)\n");
        sb.append("Partitioner: KD-tree (GridType.KDBTREE)\n");
        sb.append("Cluster:     3 nodes LAN\n");
        sb.append("Runs=").append(RUNS)
          .append(" | Warmup=").append(WARMUP)
          .append(" | kNN k=").append(KNN_K).append("\n");
        sb.append("=".repeat(70)).append("\n\n");

        // Build time
        sb.append("## BUILD TIME (Figure 9)\n");
        sb.append("-".repeat(55)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "  Sedona-RK build:     %,dms (%.1fs)%n",
                buildTime, buildTime / 1000.0));
        sb.append("  Paper ref:           ~100,000ms (~100s)\n");
        sb.append("  LiLIS-K build NYC:   ~38,314ms  (~38s)\n");
        sb.append(String.format(Locale.ROOT,
                "  LiLIS vs Sedona-RK:  %.1fx faster to build%n%n",
                (double) buildTime / 38314.0));

        // Query time
        sb.append("## QUERY TIME — Cluster LAN 3 nodes\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s %10s %10s %10s%n",
                "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Point  Sedona-RK", avg(pt), min(pt), max(pt), std(pt)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Range  Sedona-RK", avg(rng), min(rng), max(rng), std(rng)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "kNN    Sedona-RK", avg(knn), min(knn), max(knn), std(knn)));

        // Speedup vs LiLIS-K
        sb.append("\n## SPEEDUP: LiLIS-K vs Sedona-RK (cluster 3 nodes)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("LiLIS-K NYC cluster (nyc_full_query_comparison.txt):\n");
        sb.append("  Point: 780ms | Range: 743ms | kNN: 566ms\n\n");
        sb.append(String.format(Locale.ROOT,
                "  Point : LiLIS 780ms  vs Sedona-RK %,.0fms → LiLIS %.1fx faster%n",
                avg(pt), avg(pt) / 780.0));
        sb.append(String.format(Locale.ROOT,
                "  Range : LiLIS 743ms  vs Sedona-RK %,.0fms → LiLIS %.1fx faster%n",
                avg(rng), avg(rng) / 743.0));
        sb.append(String.format(Locale.ROOT,
                "  kNN   : LiLIS 566ms  vs Sedona-RK %,.0fms → LiLIS %.1fx faster%n",
                avg(knn), avg(knn) / 566.0));

        // Table IV comparison
        sb.append("\n## TABLE IV COMPARISON (kNN queries)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Paper Table IV — kNN NYC:\n");
        sb.append("  LiLIS-K:   650ms | Sedona-RK: 790,993ms | Speedup: 1,217x\n");
        sb.append(String.format(Locale.ROOT,
                "  Nhóm:      566ms | Sedona-RK: %,.0fms     | Speedup: %.0fx%n",
                avg(knn), avg(knn) / 566.0));

        // Paper reference
        sb.append("\n## PAPER REFERENCE\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Figure 4 (NYC 300M): Sedona-RK ~10^5ms (log scale)\n");
        sb.append("Figure 9: Sedona-RK build ~100s | LiLIS-K ~50s → 2x faster\n");
        sb.append("Table IV: Sedona-RK kNN CHI=7,862ms | NYC=790,993ms | SYN=83,170ms\n");
        sb.append("Takeaway 1: LiLIS outperforms by 2-3 orders of magnitude\n");
        sb.append("Takeaway 5: LiLIS builds faster than Sedona (1.5-2x)\n");

        System.out.println("\n" + sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) {
            w.write(sb.toString());
        }
    }
}
