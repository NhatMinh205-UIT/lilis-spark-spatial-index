package baseline;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * SparkVanillaBaseline — Apache Spark thuần (không có Sedona, không có index)
 *
 * Spark vanilla = brute-force RDD filter trên toàn bộ dataset
 * Đây là baseline cuối cùng trong Figure 4 của paper:
 *   LiLIS-K > Sedona variants > Sedona-N > Spark
 *
 * Chỉ support Point và Range query (filter trên RDD)
 * kNN: không implement (cần sort toàn bộ data = quá chậm)
 *
 * Point Query:  filter điểm nằm trong epsilon box
 * Range Query:  filter điểm nằm trong range window
 *
 * Paper Figure 4: Spark ~10^1 ms throughput (jobs/min)
 *   → Spark chậm hơn Sedona-N, chậm hơn LiLIS rất nhiều
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
public class SparkVanillaBaseline {

    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;
    static final double POINT_EPS   = 1e-6;

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: SparkVanillaBaseline " +
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
        System.out.println("Spark Vanilla Baseline (NO index, NO Sedona)");
        System.out.printf("Dataset: %s%n", label);
        System.out.printf("Runs: %d | Warmup: %d%n", RUNS, WARMUP);
        System.out.printf("BBox: X[%.4f,%.4f] Y[%.4f,%.4f]%n",
                minX, maxX, minY, maxY);
        System.out.println("Mode: brute-force RDD filter on raw CSV");
        System.out.println("=".repeat(65));

        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // ── Bước 1: Load và cache raw RDD ───────────────────────────
        System.out.println("[Bước 1] Loading và caching raw RDD...");

        JavaRDD<double[]> pointRDD = sc.textFile(path)
                .map(line -> {
                    String[] parts = line.trim().split(",");
                    return new double[]{
                            Double.parseDouble(parts[0]),
                            Double.parseDouble(parts[1])
                    };
                })
                .persist(StorageLevel.MEMORY_AND_DISK_SER());

        long totalCount = pointRDD.count();
        System.out.printf("[Bước 1] Loaded %,d points%n", totalCount);

        // ── Bước 2: Tạo query points ─────────────────────────────────
        Random rng = new Random(SAMPLE_SEED);
        List<double[]> qPoints = new ArrayList<>();

        for (int i = 0; i < RUNS + WARMUP; i++) {
            double cx = minX + rng.nextDouble() * (maxX - minX);
            double cy = minY + rng.nextDouble() * (maxY - minY);
            qPoints.add(new double[]{cx, cy});
        }
        System.out.printf("[Bước 2] %d query points (seed=%d)%n",
                qPoints.size(), SAMPLE_SEED);

        // ── Bước 3: Warm-up ──────────────────────────────────────────
        System.out.printf("[Warm-up] %d runs...%n", WARMUP);
        for (int w = 0; w < WARMUP; w++) {
            final double[] q = qPoints.get(w);
            // Point warm-up
            pointRDD.filter(p ->
                    p[0] >= q[0] - POINT_EPS && p[0] <= q[0] + POINT_EPS &&
                    p[1] >= q[1] - POINT_EPS && p[1] <= q[1] + POINT_EPS
            ).count();
            // Range warm-up
            pointRDD.filter(p ->
                    p[0] >= q[0] - RANGE_HALF && p[0] <= q[0] + RANGE_HALF &&
                    p[1] >= q[1] - RANGE_HALF && p[1] <= q[1] + RANGE_HALF
            ).count();
        }
        System.out.println("[Warm-up] Done.");

        // ── Bước 4: Point Query ──────────────────────────────────────
        System.out.printf("[Benchmark] Point Query — %d runs...%n", RUNS);
        long[] pointTimes = new long[RUNS];

        for (int i = 0; i < RUNS; i++) {
            final double[] q = qPoints.get(i + WARMUP);
            long t0 = System.currentTimeMillis();

            pointRDD.filter(p ->
                    p[0] >= q[0] - POINT_EPS && p[0] <= q[0] + POINT_EPS &&
                    p[1] >= q[1] - POINT_EPS && p[1] <= q[1] + POINT_EPS
            ).count();

            pointTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[Point %2d/%d] %dms%n",
                    i + 1, RUNS, pointTimes[i]);
        }

        // ── Bước 5: Range Query ──────────────────────────────────────
        System.out.printf("[Benchmark] Range Query — %d runs...%n", RUNS);
        long[] rangeTimes = new long[RUNS];

        for (int i = 0; i < RUNS; i++) {
            final double[] q = qPoints.get(i + WARMUP);
            long t0 = System.currentTimeMillis();

            pointRDD.filter(p ->
                    p[0] >= q[0] - RANGE_HALF && p[0] <= q[0] + RANGE_HALF &&
                    p[1] >= q[1] - RANGE_HALF && p[1] <= q[1] + RANGE_HALF
            ).count();

            rangeTimes[i] = System.currentTimeMillis() - t0;
            System.out.printf("[Range %2d/%d] %dms%n",
                    i + 1, RUNS, rangeTimes[i]);
        }

        // ── Bước 6: kNN → skip ───────────────────────────────────────
        System.out.println("[kNN] SKIPPED — quá chậm cho brute-force sort");

        sc.close();

        // ── Bước 7: Ghi kết quả ──────────────────────────────────────
        new java.io.File("results").mkdirs();
        String outFile = "results/" + label + "_spark_vanilla_cluster_3nodes.txt";
        writeResults(outFile, label, pointTimes, rangeTimes);
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

    static void writeResults(String file, String label,
            long[] pt, long[] rng) throws IOException {

        StringBuilder sb = new StringBuilder();

        sb.append("=".repeat(70)).append("\n");
        sb.append("SPARK VANILLA BASELINE — ").append(label).append("\n");
        sb.append("Mode:    Brute-force RDD filter (NO index, NO Sedona)\n");
        sb.append("Cluster: 3 nodes LAN\n");
        sb.append("Runs=").append(RUNS)
          .append(" | Warmup=").append(WARMUP).append("\n");
        sb.append("=".repeat(70)).append("\n\n");

        // Query time
        sb.append("## QUERY TIME — Cluster LAN 3 nodes (Figure 4)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s %10s %10s %10s%n",
                "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Point  Spark", avg(pt), min(pt), max(pt), std(pt)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Range  Spark", avg(rng), min(rng), max(rng), std(rng)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10s%n", "kNN    Spark", "N/A"));

        // Speedup vs LiLIS-K
        sb.append("\n## SPEEDUP: LiLIS-K vs Spark Vanilla (NYC cluster)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("LiLIS-K NYC: Point=780ms | Range=743ms\n\n");
        sb.append(String.format(Locale.ROOT,
                "  Point: LiLIS 780ms vs Spark %,.0fms → %.1fx faster%n",
                avg(pt), avg(pt) / 780.0));
        sb.append(String.format(Locale.ROOT,
                "  Range: LiLIS 743ms vs Spark %,.0fms → %.1fx faster%n",
                avg(rng), avg(rng) / 743.0));

        // So sánh với Sedona-N
        sb.append("\n## SO SÁNH: Spark Vanilla vs Sedona-N\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Sedona-N NYC: Point=7,032ms | Range=7,528ms\n");
        sb.append(String.format(Locale.ROOT,
                "  Point: Spark %,.0fms vs Sedona-N 7,032ms%n", avg(pt)));
        sb.append(String.format(Locale.ROOT,
                "  Range: Spark %,.0fms vs Sedona-N 7,528ms%n", avg(rng)));
        sb.append("  → Paper: Spark ≈ Sedona-N (similar brute-force)\n");

        // Paper reference
        sb.append("\n## PAPER REFERENCE (Figure 4)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Figure 4: Spark throughput ~10^1 jobs/min (lowest)\n");
        sb.append("          Sedona-N throughput ~10^1 jobs/min\n");
        sb.append("          LiLIS-K throughput ~10^3 jobs/min (highest)\n");
        sb.append("Takeaway 1: LiLIS outperforms Spark by 2-3 orders of magnitude\n");

        System.out.println("\n" + sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) {
            w.write(sb.toString());
        }
    }
}
