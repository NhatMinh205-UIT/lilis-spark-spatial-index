package idnexbuild;

import datatypes.Point;
import datatypes.Rectangle;
import index.BuildIndex;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.storage.StorageLevel;
import partitions.SpatialPartition;
import pointrdd.PointRDDUtils;
import query.KNNQuery;
import query.PointQuery;
import query.RangeQuery;
import spline.Spline;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Locale;

/**
 * QueryBenchmarkComparison — Phase 4 (Cluster LAN, Parameter Sweep)
 *
 * Thay đổi so với version cũ:
 *   1. Fix hardcode: TOTAL_N + TOTAL_AREA nhận từ args[]
 *   2. kNN sweep:   k ∈ {1, 10, 50, 100}  → reproduce Figure 8 paper
 *   3. Range sweep: selectivity ∈ 5 mức   → reproduce Figure 7 paper
 *   4. Output CSV: dễ import vào Python/Excel để vẽ biểu đồ
 *
 * Args (bắt buộc, đúng thứ tự):
 *   args[0] = datasetLabel  (vd: chi_full, nyc_full, syn_full)
 *   args[1] = dataPath      (vd: /home/pc/.../chi_full_xy.csv)
 *   args[2] = TOTAL_N       (số điểm thực tế, vd: 7699698)
 *   args[3] = MIN_X         (vd: -87.9401)
 *   args[4] = MAX_X         (vd: -87.5248)
 *   args[5] = MIN_Y         (vd: 41.6445)
 *   args[6] = MAX_Y         (vd: 42.0231)
 *
 * Memory strategy — sequential, không contention:
 *   Bước 0: Load pointRDD → takeSample(50) → compute bounds
 *   Hiệp 1: Build splineRDD → chạy TẤT CẢ LiLIS queries → unpersist
 *   Hiệp 2: Persist pointRDD → chạy TẤT CẢ No-index queries → unpersist
 */
public class QueryBenchmarkComparison {

    // ── Cấu hình runs ─────────────────────────────────────────────────
    static final int LILIS_RUNS   = 15;
    static final int NO_IDX_RUNS  = 10;
    static final int WARMUP       = 2;
    static final long SAMPLE_SEED = 42L;

    // ── kNN sweep: k ∈ {1, 10, 50, 100} — sát paper Figure 8 ─────────
    static final int[] K_ARRAY = {1, 10, 50, 100};

    // ── Selectivity sweep: 0.00001% → 0.1% — sát paper Figure 7 ──────
    // Selectivity = window_area / total_area
    // RANGE_HALF  = sqrt(selectivity * total_area) / 2 (tính trong main)
    static final double[] SELECTIVITY_ARRAY = {
        1e-7,   // 0.00001%
        1e-6,   // 0.0001%
        1e-5,   // 0.001%   ← default paper
        1e-4,   // 0.01%
        1e-3    // 0.1%
    };
    static final String[] SELECTIVITY_LABELS = {
        "0.00001%", "0.0001%", "0.001%", "0.01%", "0.1%"
    };

    // ── Biến toàn cục (set từ args) ───────────────────────────────────
    static long   TOTAL_N;
    static double TOTAL_AREA;

    // ══════════════════════════════════════════════════════════════════
    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: QueryBenchmarkComparison " +
                "<label> <path> <TOTAL_N> <MIN_X> <MAX_X> <MIN_Y> <MAX_Y>");
            System.exit(1);
        }

        String datasetLabel = args[0];
        String dataPath     = args[1];
        TOTAL_N             = Long.parseLong(args[2]);
        double minX         = Double.parseDouble(args[3]);
        double maxX         = Double.parseDouble(args[4]);
        double minY         = Double.parseDouble(args[5]);
        double maxY         = Double.parseDouble(args[6]);
        TOTAL_AREA          = (maxX - minX) * (maxY - minY);

        // Tính RANGE_HALF cho từng mức selectivity
        double[] rangeHalfArray = new double[SELECTIVITY_ARRAY.length];
        for (int i = 0; i < SELECTIVITY_ARRAY.length; i++) {
            rangeHalfArray[i] = Math.sqrt(SELECTIVITY_ARRAY[i] * TOTAL_AREA) / 2.0;
        }

        System.out.printf("Dataset:    %s%n", datasetLabel);
        System.out.printf("TOTAL_N:    %,d%n", TOTAL_N);
        System.out.printf("TOTAL_AREA: %.6f sq-deg%n", TOTAL_AREA);
        System.out.printf("BBox:       X[%.4f, %.4f] Y[%.4f, %.4f]%n",
                minX, maxX, minY, maxY);

        // ── BƯỚC 0: Load + sample ─────────────────────────────────────
        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        System.out.println("\n[Bước 0] Loading + sampling 50 query points...");
        JavaRDD<Point> pointRDD = PointRDDUtils.CreatePointRDD(sc, dataPath, 0);
        List<Point> queryPoints = pointRDD.takeSample(false, LILIS_RUNS, SAMPLE_SEED);
        System.out.println("[Bước 0] Sampled " + queryPoints.size() + " points.");

        // Tạo range rectangles cho từng selectivity × query point
        // rangeRects[sel_idx][point_idx]
        Rectangle[][] rangeRects = new Rectangle[rangeHalfArray.length][queryPoints.size()];
        for (int s = 0; s < rangeHalfArray.length; s++) {
            double h = rangeHalfArray[s];
            for (int i = 0; i < queryPoints.size(); i++) {
                double cx = queryPoints.get(i).getX();
                double cy = queryPoints.get(i).getY();
                rangeRects[s][i] = new Rectangle(
                    new Point(cx - h, cy - h),
                    new Point(cx + h, cy + h)
                );
            }
        }

        // ══════════════════════════════════════════════════════════════
        // HIỆP 1 — LiLIS-K (splineRDD trong RAM, pointRDD bị GC)
        // ══════════════════════════════════════════════════════════════
        System.out.println("\n" + "=".repeat(65));
        System.out.println("HIỆP 1 — LiLIS-K Benchmark");
        System.out.println("=".repeat(65));

        JavaRDD<Point> partitionRDD = SpatialPartition.KDBTreePartitioner(pointRDD);
        JavaRDD<Spline> splineRDD   = BuildIndex.indexBuild(partitionRDD);
        splineRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long splineCount = splineRDD.count();
        System.out.println("[Hiệp 1] Index CACHED: " + splineCount + " partitions");

        // Warm-up
        System.out.println("[Hiệp 1] Warm-up " + WARMUP + " runs...");
        for (int w = 0; w < WARMUP; w++) {
            Point wp = queryPoints.get(w);
            PointQuery.SpatialPointQuery(splineRDD, wp).count();
            RangeQuery.SpatialRangeQuery(splineRDD, rangeRects[2][w]).count(); // default sel
            KNNQuery.SpatialKNNQuery(splineRDD, 10, wp, TOTAL_AREA, TOTAL_N);
        }

        // 1A. Point Query (default)
        System.out.println("[Hiệp 1] Point Query (50 runs)...");
        long[] pointLiLIS = new long[LILIS_RUNS];
        for (int i = 0; i < LILIS_RUNS; i++) {
            long t0 = System.currentTimeMillis();
            PointQuery.SpatialPointQuery(splineRDD, queryPoints.get(i)).count();
            pointLiLIS[i] = System.currentTimeMillis() - t0;
        }

        // 1B. Range Query sweep (5 selectivity levels)
        System.out.println("[Hiệp 1] Range Query sweep (5 selectivity levels × 50 runs)...");
        long[][] rangeLiLIS = new long[rangeHalfArray.length][LILIS_RUNS];
        for (int s = 0; s < rangeHalfArray.length; s++) {
            System.out.printf("  sel=%s (rangeHalf=%.6f)...%n",
                    SELECTIVITY_LABELS[s], rangeHalfArray[s]);
            for (int i = 0; i < LILIS_RUNS; i++) {
                long t0 = System.currentTimeMillis();
                RangeQuery.SpatialRangeQuery(splineRDD, rangeRects[s][i]).count();
                rangeLiLIS[s][i] = System.currentTimeMillis() - t0;
            }
        }

        // 1C. kNN Query sweep (k ∈ {1, 10, 50, 100})
        System.out.println("[Hiệp 1] kNN Query sweep (k={1,10,50,100} × 50 runs)...");
        long[][] knnLiLIS = new long[K_ARRAY.length][LILIS_RUNS];
        for (int ki = 0; ki < K_ARRAY.length; ki++) {
            int k = K_ARRAY[ki];
            System.out.printf("  k=%d...%n", k);
            for (int i = 0; i < LILIS_RUNS; i++) {
                long t0 = System.currentTimeMillis();
                KNNQuery.SpatialKNNQuery(splineRDD, k, queryPoints.get(i),
                        TOTAL_AREA, TOTAL_N);
                knnLiLIS[ki][i] = System.currentTimeMillis() - t0;
            }
        }

        splineRDD.unpersist(true);
        System.out.println("[Hiệp 1] DONE. splineRDD released.");

        // ══════════════════════════════════════════════════════════════
        // HIỆP 2 — No-Index (pointRDD trong RAM)
        // ══════════════════════════════════════════════════════════════
        System.out.println("\n" + "=".repeat(65));
        System.out.println("HIỆP 2 — No-Index Benchmark");
        System.out.println("=".repeat(65));

        pointRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long rawCount = pointRDD.count();
        System.out.println("[Hiệp 2] pointRDD CACHED: " + rawCount + " points");

        // Warm-up
        System.out.println("[Hiệp 2] Warm-up " + WARMUP + " runs...");
        for (int w = 0; w < WARMUP; w++) {
            final double wx = queryPoints.get(w).getX();
            final double wy = queryPoints.get(w).getY();
            pointRDD.filter(new Function<Point, Boolean>() {
                public Boolean call(Point p) { return p.getX() == wx && p.getY() == wy; }
            }).count();
            RangeQuery.SpatialRangeQueryWithOutIndex(pointRDD, rangeRects[2][w]).count();
            KNNQuery.SpatialKNNQuerywithoutIndex(pointRDD, 10,
                    queryPoints.get(w), TOTAL_AREA, TOTAL_N);
        }

        // 2A. Point Query (default)
        System.out.println("[Hiệp 2] Point Query (10 runs)...");
        long[] pointNoIdx = new long[NO_IDX_RUNS];
        for (int i = 0; i < NO_IDX_RUNS; i++) {
            final double qx = queryPoints.get(i).getX();
            final double qy = queryPoints.get(i).getY();
            long t0 = System.currentTimeMillis();
            pointRDD.filter(new Function<Point, Boolean>() {
                public Boolean call(Point p) { return p.getX() == qx && p.getY() == qy; }
            }).count();
            pointNoIdx[i] = System.currentTimeMillis() - t0;
        }

        // 2B. Range Query sweep
        System.out.println("[Hiệp 2] Range Query sweep (5 selectivity levels × 10 runs)...");
        long[][] rangeNoIdx = new long[rangeHalfArray.length][NO_IDX_RUNS];
        for (int s = 0; s < rangeHalfArray.length; s++) {
            System.out.printf("  sel=%s...%n", SELECTIVITY_LABELS[s]);
            for (int i = 0; i < NO_IDX_RUNS; i++) {
                long t0 = System.currentTimeMillis();
                RangeQuery.SpatialRangeQueryWithOutIndex(pointRDD, rangeRects[s][i]).count();
                rangeNoIdx[s][i] = System.currentTimeMillis() - t0;
            }
        }

        // 2C. kNN Query sweep
        System.out.println("[Hiệp 2] kNN Query sweep (k={1,10,50,100} × 10 runs)...");
        long[][] knnNoIdx = new long[K_ARRAY.length][NO_IDX_RUNS];
        for (int ki = 0; ki < K_ARRAY.length; ki++) {
            int k = K_ARRAY[ki];
            System.out.printf("  k=%d...%n", k);
            for (int i = 0; i < NO_IDX_RUNS; i++) {
                long t0 = System.currentTimeMillis();
                KNNQuery.SpatialKNNQuerywithoutIndex(pointRDD, k,
                        queryPoints.get(i), TOTAL_AREA, TOTAL_N);
                knnNoIdx[ki][i] = System.currentTimeMillis() - t0;
            }
        }

        pointRDD.unpersist(true);
        sc.close();

        // ── Ghi kết quả ───────────────────────────────────────────────
        String outputFile = "results/" + datasetLabel + "_query_comparison.txt";
        writeResults(outputFile, datasetLabel,
                pointLiLIS, pointNoIdx,
                rangeLiLIS, rangeNoIdx,
                knnLiLIS,   knnNoIdx,
                rangeHalfArray);

        System.out.println("\nSaved: " + outputFile);
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

    // ── Write output ──────────────────────────────────────────────────
    static void writeResults(String file, String label,
            long[] pointL, long[] pointN,
            long[][] rangeL, long[][] rangeN,
            long[][] knnL,   long[][] knnN,
            double[] rangeHalfArr) throws IOException {

        StringBuilder sb = new StringBuilder();

        // ── Header ────────────────────────────────────────────────────
        sb.append("=".repeat(70)).append("\n");
        sb.append("QUERY BENCHMARK — ").append(label).append("\n");
        sb.append("LiLIS-K runs=").append(LILIS_RUNS)
          .append(" | No-index runs=").append(NO_IDX_RUNS)
          .append(" | Warmup=").append(WARMUP).append("\n");
        sb.append("TOTAL_N=").append(String.format("%,d", TOTAL_N))
          .append(" | TOTAL_AREA=").append(String.format("%.4f", TOTAL_AREA))
          .append(" sq-deg\n");
        sb.append("Query points: 50 random via takeSample(seed=42)\n");
        sb.append("=".repeat(70)).append("\n\n");

        // ── SECTION 1: Default summary ─────────────────────────────────
        sb.append("## SECTION 1 — Default Summary (k=10, sel=0.001%)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append(String.format(Locale.ROOT, "%-25s %10s %10s %10s %10s%n",
                "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(65)).append("\n");

        appendRow(sb, "Point  LiLIS-K",  pointL);
        appendRow(sb, "Point  No-index", pointN);
        sb.append("\n");
        appendRow(sb, "Range  LiLIS-K (def)", rangeL[2]);  // index 2 = 0.001%
        appendRow(sb, "Range  No-index (def)",rangeN[2]);
        sb.append("\n");
        appendRow(sb, "kNN    LiLIS-K (k=10)", knnL[1]);   // index 1 = k=10
        appendRow(sb, "kNN    No-index (k=10)",knnN[1]);

        sb.append("\n");
        sb.append("SPEEDUP (No-index / LiLIS-K):\n");
        sb.append(String.format(Locale.ROOT, "  Point : %.1fx%n", avg(pointN) / avg(pointL)));
        sb.append(String.format(Locale.ROOT, "  Range : %.1fx%n", avg(rangeN[2]) / avg(rangeL[2])));
        sb.append(String.format(Locale.ROOT, "  kNN   : %.1fx%n", avg(knnN[1])   / avg(knnL[1])));

        // ── SECTION 2: kNN sweep (CSV format) ─────────────────────────
        sb.append("\n\n## SECTION 2 — kNN Sweep (Figure 8)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("k,method,avg_ms,min_ms,max_ms,std_ms\n");
        for (int ki = 0; ki < K_ARRAY.length; ki++) {
            sb.append(String.format(Locale.ROOT, "%d,LiLIS-K,%.1f,%d,%d,%.1f%n",
                    K_ARRAY[ki], avg(knnL[ki]), min(knnL[ki]), max(knnL[ki]), std(knnL[ki])));
            sb.append(String.format(Locale.ROOT, "%d,No-index,%.1f,%d,%d,%.1f%n",
                    K_ARRAY[ki], avg(knnN[ki]), min(knnN[ki]), max(knnN[ki]), std(knnN[ki])));
        }

        // ── SECTION 3: Range selectivity sweep (CSV format) ───────────
        sb.append("\n\n## SECTION 3 — Range Selectivity Sweep (Figure 7)\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("selectivity,range_half,method,avg_ms,min_ms,max_ms,std_ms\n");
        for (int s = 0; s < SELECTIVITY_LABELS.length; s++) {
            sb.append(String.format(Locale.ROOT,
                    "%s,%.6f,LiLIS-K,%.1f,%d,%d,%.1f%n",
                    SELECTIVITY_LABELS[s], rangeHalfArr[s],
                    avg(rangeL[s]), min(rangeL[s]), max(rangeL[s]), std(rangeL[s])));
            sb.append(String.format(Locale.ROOT,
                    "%s,%.6f,No-index,%.1f,%d,%d,%.1f%n",
                    SELECTIVITY_LABELS[s], rangeHalfArr[s],
                    avg(rangeN[s]), min(rangeN[s]), max(rangeN[s]), std(rangeN[s])));
        }

        // ── SECTION 4: Paper reference ─────────────────────────────────
        sb.append("\n\n## SECTION 4 — Paper Reference\n");
        sb.append("-".repeat(70)).append("\n");
        sb.append("Takeaway 1: LiLIS nhanh hơn 2-3 orders of magnitude vs Sedona\n");
        sb.append("Table III (NYC, LiLIS-K):\n");
        sb.append("  Point :  82.59 ms | Range : 468.64 ms | kNN : 650.20 ms\n");
        sb.append("Figure 8 (kNN varying k): stable across k=1→100 for LiLIS-K\n");
        sb.append("Figure 7 (Range selectivity): uniform < skewed for LiLIS\n");

        System.out.println(sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) {
            w.write(sb.toString());
        }
    }

    static void appendRow(StringBuilder sb, String lbl, long[] arr) {
        sb.append(String.format(Locale.ROOT,
                "%-25s %10.1f %10d %10d %10.1f%n",
                lbl, avg(arr), min(arr), max(arr), std(arr)));
    }
}