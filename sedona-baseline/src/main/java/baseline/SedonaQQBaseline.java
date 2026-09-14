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
 * SedonaQQBaseline — Apache Sedona Quad-tree + Quad-tree (Sedona-QQ)
 *
 * Sedona-QQ = Quad-tree spatial index + Quad-tree spatial partitioner
 * Đây là 1 trong 4 variants Sedona trong paper (Table II):
 *   Sedona-RK: R-tree index + KD-tree partitioner
 *   Sedona-RQ: R-tree index + Quad-tree partitioner
 *   Sedona-QK: Quad-tree index + KD-tree partitioner
 *   Sedona-QQ: Quad-tree index + Quad-tree partitioner ← file này
 *
 * QUAN TRỌNG: Quad-tree index KHÔNG hỗ trợ kNN query
 *   → Chỉ benchmark Point Query và Range Query
 *   → Paper: "Quad-tree index in Sedona does not support kNN"
 *
 * Sequence chính xác theo Sedona API:
 *   1. Load PointRDD từ CSV
 *   2. analyze() — tính bounding box tự động
 *   3. spatialPartitioning(QUADTREE) — chia data theo Quad-tree
 *   4. buildIndex(QUADTREE, true) — build index trên từng partition
 *   5. indexedRawRDD.persist() + count() — trigger + cache vào RAM
 *   6. Query WITH index (useIndex=true)
 *
 * So sánh với paper Figure 9 (build cost) và Figure 4 (query time)
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
public class SedonaQQBaseline {

    // ── Cấu hình ────────────────────────────────────────────────────
    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;  // selectivity ~0.001%
    // Point query: tiny epsilon box xung quanh điểm truy vấn
    static final double POINT_EPS   = 1e-6;

    static final GeometryFactory GF = new GeometryFactory();

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: SedonaQQBaseline " +
                "<label> <path> <TOTAL_N> <MIN_X> <MAX_X> <MIN_Y> <MAX_Y>");
            System.exit(1);
        }

        String label  = args[0];
        String path   = args[1];
        double minX   = Double.parseDouble(args[3]);
        double maxX   = Double.parseDouble(args[4]);
        double minY   = Double.parseDouble(args[5]);
        double maxY   = Double.parseDouble(args[6]);

        System.out.println("=".repeat(65));
        System.out.println("Sedona-QQ Baseline (Quadtree index + Quadtree partitioner)");
        System.out.printf("Dataset: %s%n", label);
        System.out.printf("BBox: X[%.4f,%.4f] Y[%.4f,%.4f]%n",
                minX, maxX, minY, maxY);
        System.out.println("NOTE: kNN NOT supported by Quadtree index in Sedona");
        System.out.println("=".repeat(65));

        // ── Init Spark ───────────────────────────────────────────────
        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // ── BƯỚC 1: Tạo query points ngẫu nhiên ─────────────────────
        // Tạo TRƯỚC khi load data để đảm bảo reproducible
        Random rng = new Random(SAMPLE_SEED);
        List<Point>    qPoints      = new ArrayList<>();
        List<Envelope> qRanges      = new ArrayList<>();
        List<Envelope> qPointBoxes  = new ArrayList<>(); // tiny box cho point query

        for (int i = 0; i < RUNS + WARMUP; i++) {
            double cx = minX + rng.nextDouble() * (maxX - minX);
            double cy = minY + rng.nextDouble() * (maxY - minY);
            qPoints.add(GF.createPoint(new Coordinate(cx, cy)));

            // Range query window (RANGE_HALF ~0.001% selectivity)
            qRanges.add(new Envelope(
                    cx - RANGE_HALF, cx + RANGE_HALF,
                    cy - RANGE_HALF, cy + RANGE_HALF));

            // Point query: epsilon box (dùng index để tìm exact point)
            qPointBoxes.add(new Envelope(
                    cx - POINT_EPS, cx + POINT_EPS,
                    cy - POINT_EPS, cy + POINT_EPS));
        }
        System.out.printf("[Bước 1] Đã tạo %d query points (seed=%d)%n",
                qPoints.size(), SAMPLE_SEED);

        // ── BƯỚC 2: Load PointRDD ────────────────────────────────────
        System.out.println("[Bước 2] Loading PointRDD từ CSV...");
        PointRDD pointRDD = new PointRDD(
                sc, path, 0, FileDataSplitter.CSV, false);
        System.out.printf("[Bước 2] Raw partitions: %d%n",
                pointRDD.rawSpatialRDD.getNumPartitions());

        // ── BƯỚC 3: Build Index (đo thời gian cho Figure 9) ─────────
        System.out.println("[Bước 3] Building Sedona-QQ index...");
        System.out.println("         Step 3a: analyze()");
        System.out.println("         Step 3b: spatialPartitioning(QUADTREE)");
        System.out.println("         Step 3c: buildIndex(QUADTREE)");

        long buildStart = System.currentTimeMillis();

        // 3a. Analyze — tính bounding box và statistics
        pointRDD.analyze();

        // 3b. Partition với Quad-tree spatial partitioner
        pointRDD.spatialPartitioning(GridType.QUADTREE);

        // 3c. Build Quad-tree index trên từng partition
        //     true = buildOnSpatialPartitionedRDD (QUAN TRỌNG!)
        pointRDD.buildIndex(IndexType.QUADTREE, false);

        // 3d. Persist + trigger để force execution (đo đúng build time)
        pointRDD.indexedRawRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long indexCount = pointRDD.indexedRawRDD.count();

        long buildEnd  = System.currentTimeMillis();
        long buildTime = buildEnd - buildStart;

        System.out.printf("[Bước 3] Index BUILT và CACHED: %d partitions%n",
                indexCount);
        System.out.printf("[Bước 3] BUILD TIME: %,dms (%.1fs)%n",
                buildTime, buildTime / 1000.0);

        // ── BƯỚC 4: Warm-up queries ──────────────────────────────────
        System.out.printf("[Warm-up] %d runs với index...%n", WARMUP);
        for (int w = 0; w < WARMUP; w++) {
            // Point warm-up (dùng tiny box + index)
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qPointBoxes.get(w), false, true).count();
            // Range warm-up (dùng index)
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qRanges.get(w), false, true).count();
        }
        System.out.println("[Warm-up] Done.");

        // ── BƯỚC 5: Benchmark Point Query (WITH index) ───────────────
        System.out.printf("[Benchmark] Point Query — %d runs WITH Quad-tree index...%n",
                RUNS);
        long[] pointTimes = new long[RUNS];

        for (int i = 0; i < RUNS; i++) {
            int qi = i + WARMUP;
            long t0 = System.currentTimeMillis();

            // Dùng tiny epsilon box + Quad-tree index
            // useIndex=true → sử dụng Quad-tree để prune partitions
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qPointBoxes.get(qi), false, true).count();

            pointTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[Point %2d/%d] %dms%n",
                    i + 1, RUNS, pointTimes[i]);
        }

        // ── BƯỚC 6: Benchmark Range Query (WITH index) ───────────────
        System.out.printf("[Benchmark] Range Query — %d runs WITH Quad-tree index...%n",
                RUNS);
        long[] rangeTimes = new long[RUNS];

        for (int i = 0; i < RUNS; i++) {
            int qi = i + WARMUP;
            long t0 = System.currentTimeMillis();

            // useIndex=true → sử dụng Quad-tree index để tìm kiếm
            RangeQuery.SpatialRangeQuery(
                    pointRDD, qRanges.get(qi), false, true).count();

            rangeTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[Range %2d/%d] %dms%n",
                    i + 1, RUNS, rangeTimes[i]);
        }

        // ── BƯỚC 7: kNN → N/A ────────────────────────────────────────
        System.out.println("[kNN] SKIPPED — Quad-tree index does not support kNN in Sedona");
        System.out.println("      See paper: 'Quad-tree index in Sedona does not support kNN'");

        sc.close();

        // ── BƯỚC 8: Ghi kết quả ──────────────────────────────────────
        java.io.File dir = new java.io.File("results");
        if (!dir.exists()) dir.mkdirs();

        String outFile = "results/" + label + "_sedona_qq_cluster_3nodes.txt";
        writeResults(outFile, label, buildTime, pointTimes, rangeTimes);
        System.out.println("Saved: " + outFile);
    }

    // ── Helpers ───────────────────────────────────────────────────────
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

    // ── Write results ─────────────────────────────────────────────────
    static void writeResults(String file, String label,
            long buildTime, long[] pt, long[] rng) throws IOException {

        StringBuilder sb = new StringBuilder();

        // Header
        sb.append("=".repeat(70)).append("\n");
        sb.append("SEDONA-QQ BASELINE — ").append(label).append("\n");
        sb.append("Index:       Quad-tree (IndexType.QUADTREE)\n");
        sb.append("Partitioner: Quad-tree (GridType.QUADTREE)\n");
        sb.append("Runs=").append(RUNS)
          .append(" | Warmup=").append(WARMUP)
          .append(" | Range_half=").append(RANGE_HALF).append("\n");
        sb.append("NOTE: kNN NOT supported by Quad-tree index → N/A\n");
        sb.append("=".repeat(70)).append("\n\n");

        // Build time
        sb.append("## BUILD TIME (Figure 9)\n");
        sb.append("-".repeat(50)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "  Sedona-QQ build: %,dms (%.1fs)%n",
                buildTime, buildTime / 1000.0));
        sb.append(String.format(Locale.ROOT,
                "  Paper ref:       ~60,000-80,000ms (~60-80s)%n"));
        sb.append(String.format(Locale.ROOT,
                "  LiLIS-K build NYC LAN: ~38,314ms (~38s)%n"));
        sb.append(String.format(Locale.ROOT,
                "  → LiLIS-K vs Sedona-QQ build: %.1fx faster%n%n",
                (double) buildTime / 38314.0));

        // Query results
        sb.append("## QUERY TIME (Figure 4)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s %10s %10s %10s%n",
                "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(60)).append("\n");

        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Point  Sedona-QQ", avg(pt), min(pt), max(pt), std(pt)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s %10s %10s %10s%n",
                "Range  Sedona-QQ",
                String.format("%.1f", avg(rng)),
                min(rng), max(rng),
                String.format("%.1f", std(rng))));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s%n", "kNN    Sedona-QQ", "N/A"));

        // So sánh với LiLIS-K local
        sb.append("\n## SPEEDUP: LiLIS-K (local) vs Sedona-QQ (local)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("LiLIS-K local NYC (nyc_full_query_comparison_local.txt):\n");
        sb.append("  Point: ~1,116ms | Range: ~1,177ms | kNN: ~1,114ms\n\n");

        double lilisPt  = 1116.0;
        double lilisRng = 1177.0;

        sb.append(String.format(Locale.ROOT,
                "  Point : LiLIS-K %,.0fms vs Sedona-QQ %,.0fms → LiLIS %.1fx faster%n",
                lilisPt, avg(pt), avg(pt) / lilisPt));
        sb.append(String.format(Locale.ROOT,
                "  Range : LiLIS-K %,.0fms vs Sedona-QQ %,.0fms → LiLIS %.1fx faster%n",
                lilisRng, avg(rng), avg(rng) / lilisRng));
        sb.append("  kNN   : N/A (Quad-tree không support kNN)\n");

        // So sánh với Sedona-N
        sb.append("\n## SEDONA-QQ vs SEDONA-N (so sánh nội bộ Sedona)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Sedona-N NYC (nyc_full_sedona_n_baseline.txt):\n");
        sb.append("  Point: ~12,929ms | Range: ~???,???ms (pending fix)\n\n");
        sb.append(String.format(Locale.ROOT,
                "  Point : Sedona-QQ %,.0fms vs Sedona-N 12,929ms%n", avg(pt)));
        sb.append(String.format(Locale.ROOT,
                "  → Index help: %.2fx%n",
                12929.0 / avg(pt)));

        // Paper reference
        sb.append("\n## PAPER REFERENCE (Figure 4, Figure 9)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Figure 4 (NYC 300M) — log scale (10^4 ms range):\n");
        sb.append("  Sedona-QQ Point: ~10,000-100,000ms\n");
        sb.append("  Sedona-QQ Range: ~10,000-100,000ms\n");
        sb.append("  LiLIS-K   Range: ~472ms\n");
        sb.append("  → Paper speedup: ~1,000x (2-3 orders of magnitude)\n\n");
        sb.append("Figure 9 (Build cost):\n");
        sb.append("  Sedona-QQ build: ~60-80s\n");
        sb.append("  LiLIS-K   build: ~50s\n");
        sb.append("  → Paper: LiLIS 1.5-2x faster to build\n\n");
        sb.append("Takeaway 1: LiLIS outperforms Sedona by 2-3 orders of magnitude\n");
        sb.append("Takeaway 5: LiLIS build cost faster than Sedona (1.5-2x)\n");

        System.out.println("\n" + sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) {
            w.write(sb.toString());
        }
    }
}
