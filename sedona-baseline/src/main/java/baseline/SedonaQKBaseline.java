package baseline;

import org.apache.sedona.common.enums.FileDataSplitter;
import org.apache.sedona.core.enums.GridType;
import org.apache.sedona.core.enums.IndexType;
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
 * SedonaQKBaseline — Apache Sedona Quad-tree + KD-tree (Sedona-QK)
 *
 * Sedona-QK = Quad-tree index + KD-tree (KDBTREE) partitioner
 * Table II paper:
 *   Sedona-QK: Quad-tree index + KD-tree partitioner ← file này
 *   Sedona-QQ: Quad-tree index + Quad-tree partitioner
 *   Sedona-RQ: R-tree index  + Quad-tree partitioner
 *   Sedona-RK: R-tree index  + KD-tree partitioner
 *
 * QUAN TRỌNG:
 *   Quad-tree index KHÔNG hỗ trợ kNN → chỉ Point + Range
 *   buildIndex(QUADTREE, false) → indexedRawRDD
 *   RangeQuery(useIndex=true) → đọc từ indexedRawRDD
 *
 * Paper Figure 4: Sedona-QK ~10^4ms (log scale, NYC 300M)
 * Paper Figure 9: Sedona-QK build ~80s
 *
 * Args:
 *   args[0] = datasetLabel  (nyc_full)
 *   args[1] = dataPath
 *   args[2] = TOTAL_N
 *   args[3] = MIN_X
 *   args[4] = MAX_X
 *   args[5] = MIN_Y
 *   args[6] = MAX_Y
 */
public class SedonaQKBaseline {

    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;
    static final double POINT_EPS   = 1e-6;

    static final GeometryFactory GF = new GeometryFactory();

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: SedonaQKBaseline " +
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
        System.out.println("Sedona-QK (Quad-tree index + KD-tree partitioner)");
        System.out.printf("Dataset: %s%n", label);
        System.out.printf("Runs: %d | Warmup: %d%n", RUNS, WARMUP);
        System.out.printf("BBox: X[%.4f,%.4f] Y[%.4f,%.4f]%n",
                minX, maxX, minY, maxY);
        System.out.println("NOTE: kNN NOT supported by Quad-tree index → N/A");
        System.out.println("=".repeat(65));

        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // ── Bước 1: Query points ─────────────────────────────────────
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

        // ── Bước 3: Build Sedona-QK index ───────────────────────────
        System.out.println("[Bước 3] Building Sedona-QK index...");
        System.out.println("  3a. analyze()");
        System.out.println("  3b. spatialPartitioning(KDBTREE) ← K partitioner");
        System.out.println("  3c. buildIndex(QUADTREE, false)   ← Quad-tree index");

        long buildStart = System.currentTimeMillis();

        pointRDD.analyze();
        pointRDD.spatialPartitioning(GridType.KDBTREE);
        pointRDD.buildIndex(IndexType.QUADTREE, false);

        pointRDD.indexedRawRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long indexCount = pointRDD.indexedRawRDD.count();

        long buildTime = System.currentTimeMillis() - buildStart;
        System.out.printf("[Bước 3] INDEX BUILT: %d partitions | TIME: %,dms (%.1fs)%n",
                indexCount, buildTime, buildTime / 1000.0);

        // ── Bước 4: Warm-up ──────────────────────────────────────────
        System.out.printf("[Warm-up] %d runs...%n", WARMUP);
        for (int w = 0; w < WARMUP; w++) {
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qPointBoxes.get(w), false, true).count();
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qRanges.get(w), false, true).count();
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

        // ── Bước 7: kNN → N/A ────────────────────────────────────────
        System.out.println("[kNN] SKIPPED — Quad-tree index không support kNN");

        sc.close();

        // ── Bước 8: Ghi kết quả ──────────────────────────────────────
        new java.io.File("results").mkdirs();
        String outFile = "results/" + label + "_sedona_qk_cluster_3nodes.txt";
        writeResults(outFile, label, buildTime, pointTimes, rangeTimes);
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
            long[] pt, long[] rng) throws IOException {

        StringBuilder sb = new StringBuilder();

        sb.append("=".repeat(70)).append("\n");
        sb.append("SEDONA-QK BASELINE — ").append(label).append("\n");
        sb.append("Index:       Quad-tree (IndexType.QUADTREE)\n");
        sb.append("Partitioner: KD-tree   (GridType.KDBTREE)\n");
        sb.append("Cluster:     3 nodes LAN\n");
        sb.append("Runs=").append(RUNS)
          .append(" | Warmup=").append(WARMUP).append("\n");
        sb.append("NOTE: kNN NOT supported by Quad-tree index → N/A\n");
        sb.append("=".repeat(70)).append("\n\n");

        // Build time
        sb.append("## BUILD TIME (Figure 9)\n");
        sb.append("-".repeat(55)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "  Sedona-QK build:     %,dms (%.1fs)%n",
                buildTime, buildTime / 1000.0));
        sb.append("  Paper ref:           ~80,000ms (~80s)\n");
        sb.append("  LiLIS-K build NYC:   ~38,314ms (~38s)\n");
        sb.append(String.format(Locale.ROOT,
                "  LiLIS vs Sedona-QK:  %.1fx faster%n%n",
                (double) buildTime / 38314.0));

        // Query time
        sb.append("## QUERY TIME — Cluster LAN 3 nodes (Figure 4)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s %10s %10s %10s%n",
                "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Point  Sedona-QK", avg(pt), min(pt), max(pt), std(pt)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Range  Sedona-QK", avg(rng), min(rng), max(rng), std(rng)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s%n", "kNN    Sedona-QK", "N/A"));

        // Speedup vs LiLIS-K
        sb.append("\n## SPEEDUP: LiLIS-K vs Sedona-QK (NYC cluster)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("LiLIS-K NYC: Point=780ms | Range=743ms\n\n");
        sb.append(String.format(Locale.ROOT,
                "  Point: LiLIS 780ms vs Sedona-QK %,.0fms → %.1fx faster%n",
                avg(pt), avg(pt) / 780.0));
        sb.append(String.format(Locale.ROOT,
                "  Range: LiLIS 743ms vs Sedona-QK %,.0fms → %.1fx faster%n",
                avg(rng), avg(rng) / 743.0));
        sb.append("  kNN  : N/A\n");

        // Paper reference
        sb.append("\n## PAPER REFERENCE (Figure 4, Figure 9)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Figure 4: Sedona-QK ~10^4ms (log scale)\n");
        sb.append("Figure 9: Sedona-QK build ~80s\n");

        // So sánh nội bộ Sedona
        sb.append("\n## SO SÁNH NỘI BỘ SEDONA-QK vs QQ\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Sedona-QQ NYC: Point=9,286ms | Range=9,889ms\n");
        sb.append(String.format(Locale.ROOT,
                "Sedona-QK NYC: Point=%.0fms   | Range=%.0fms%n",
                avg(pt), avg(rng)));
        sb.append("Diff (QK vs QQ): KD-tree partitioner vs Quad-tree\n");

        System.out.println("\n" + sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) {
            w.write(sb.toString());
        }
    }
}
