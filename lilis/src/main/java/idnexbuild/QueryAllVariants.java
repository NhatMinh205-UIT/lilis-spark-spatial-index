package idnexbuild;

import datatypes.Point;
import datatypes.Rectangle;
import index.BuildIndex;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
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
 * QueryAllVariants — Reproduce Table III (RQ2: Varying Partitioners)
 *
 * Chạy Point + Range + kNN cho TẤT CẢ 5 variants:
 *   LiLIS-F (Fixed-grid)
 *   LiLIS-A (Adaptive-grid)
 *   LiLIS-Q (Quad-tree)
 *   LiLIS-K (KD-tree)      ← default paper
 *   LiLIS-R (R-tree)
 *
 * Paper Table III (NYC, ms):
 *   LiLIS-F: Point=218  Range=704   kNN=1107
 *   LiLIS-A: Point=199  Range=521   kNN=1767
 *   LiLIS-Q: Point=141  Range=341   kNN=774
 *   LiLIS-K: Point=83   Range=469   kNN=650
 *   LiLIS-R: Point=77   Range=472   kNN=619
 *
 * Args:
 *   args[0] = datasetLabel (nyc_full)
 *   args[1] = dataPath
 *   args[2] = TOTAL_N
 *   args[3] = MIN_X
 *   args[4] = MAX_X
 *   args[5] = MIN_Y
 *   args[6] = MAX_Y
 */
public class QueryAllVariants {

    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final int    KNN_K       = 10;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;

    static long   TOTAL_N;
    static double TOTAL_AREA;

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: QueryAllVariants " +
                "<label> <path> <TOTAL_N> <MIN_X> <MAX_X> <MIN_Y> <MAX_Y>");
            System.exit(1);
        }

        String label = args[0];
        String path  = args[1];
        TOTAL_N      = Long.parseLong(args[2]);
        double minX  = Double.parseDouble(args[3]);
        double maxX  = Double.parseDouble(args[4]);
        double minY  = Double.parseDouble(args[5]);
        double maxY  = Double.parseDouble(args[6]);
        TOTAL_AREA   = (maxX - minX) * (maxY - minY);

        System.out.println("=".repeat(65));
        System.out.println("QueryAllVariants — Reproduce Table III (RQ2)");
        System.out.printf("Dataset: %s | N=%,d%n", label, TOTAL_N);
        System.out.println("Variants: F, A, Q, K, R");
        System.out.println("=".repeat(65));

        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load data một lần
        JavaRDD<Point> pointRDD = PointRDDUtils.CreatePointRDD(sc, path, 0);

        // Sample query points
        List<Point> queryPoints = pointRDD.takeSample(false, RUNS + WARMUP, SAMPLE_SEED);
        Rectangle[] queryRanges = new Rectangle[queryPoints.size()];
        for (int i = 0; i < queryPoints.size(); i++) {
            double cx = queryPoints.get(i).getX();
            double cy = queryPoints.get(i).getY();
            queryRanges[i] = new Rectangle(
                new datatypes.Point(cx - RANGE_HALF, cy - RANGE_HALF),
                new datatypes.Point(cx + RANGE_HALF, cy + RANGE_HALF)
            );
        }
        System.out.printf("[Init] %d query points sampled%n", queryPoints.size());

        // Output file
        String outFile = "results/" + label + "_query_all_variants.txt";
        StringBuilder sb = new StringBuilder();
        sb.append("=".repeat(70)).append("\n");
        sb.append("QUERY ALL VARIANTS — ").append(label).append(" (Table III)\n");
        sb.append("Runs=").append(RUNS).append(" | Warmup=").append(WARMUP)
          .append(" | kNN k=").append(KNN_K).append("\n");
        sb.append("Cluster: 3 nodes LAN\n");
        sb.append("=".repeat(70)).append("\n\n");
        sb.append(String.format(Locale.ROOT,
                "%-12s %12s %12s %12s%n",
                "Variant", "Point(ms)", "Range(ms)", "kNN(ms)"));
        sb.append("-".repeat(50)).append("\n");

        // Chạy từng variant
        String[] variantNames = {"F", "A", "Q", "K", "R"};

        for (String variant : variantNames) {
            System.out.printf("%n>>> LiLIS-%s <<<%n", variant);

            // Build index theo variant
            JavaRDD<Point> partitioned;
            switch (variant) {
                case "F": partitioned = SpatialPartition.FixGridPartitioner(pointRDD); break;
                case "A": partitioned = SpatialPartition.AdaptiveGridPartitioner(pointRDD); break;
                case "Q": partitioned = SpatialPartition.QuadtreePartitioner(pointRDD); break;
                case "K": partitioned = SpatialPartition.KDBTreePartitioner(pointRDD); break;
                case "R": partitioned = SpatialPartition.RtreePartitoner(pointRDD); break;
                default:  throw new Exception("Unknown variant: " + variant);
            }

            JavaRDD<Spline> splineRDD = BuildIndex.indexBuild(partitioned);
            splineRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
            long cnt = splineRDD.count();
            System.out.printf("[LiLIS-%s] Index built: %d partitions%n", variant, cnt);

            // Warm-up
            for (int w = 0; w < WARMUP; w++) {
                PointQuery.SpatialPointQuery(splineRDD, queryPoints.get(w)).count();
                RangeQuery.SpatialRangeQuery(splineRDD, queryRanges[w]).count();
                KNNQuery.SpatialKNNQuery(splineRDD, KNN_K, queryPoints.get(w), TOTAL_AREA, TOTAL_N);
            }

            // Benchmark
            long[] pointT = new long[RUNS];
            long[] rangeT = new long[RUNS];
            long[] knnT   = new long[RUNS];

            for (int i = 0; i < RUNS; i++) {
                int qi = i + WARMUP;
                long t0;

                t0 = System.currentTimeMillis();
                PointQuery.SpatialPointQuery(splineRDD, queryPoints.get(qi)).count();
                pointT[i] = System.currentTimeMillis() - t0;

                t0 = System.currentTimeMillis();
                RangeQuery.SpatialRangeQuery(splineRDD, queryRanges[qi]).count();
                rangeT[i] = System.currentTimeMillis() - t0;

                t0 = System.currentTimeMillis();
                KNNQuery.SpatialKNNQuery(splineRDD, KNN_K, queryPoints.get(qi), TOTAL_AREA, TOTAL_N);
                knnT[i] = System.currentTimeMillis() - t0;

                System.out.printf("[LiLIS-%s] Run %2d/%d | P=%dms R=%dms K=%dms%n",
                    variant, i+1, RUNS, pointT[i], rangeT[i], knnT[i]);
            }

            double pAvg = avg(pointT), rAvg = avg(rangeT), kAvg = avg(knnT);
            System.out.printf("[LiLIS-%s] AVG: Point=%.0fms Range=%.0fms kNN=%.0fms%n",
                variant, pAvg, rAvg, kAvg);

            sb.append(String.format(Locale.ROOT,
                "%-12s %12.1f %12.1f %12.1f%n",
                "LiLIS-" + variant, pAvg, rAvg, kAvg));

            splineRDD.unpersist(true);
        }

        // Paper reference
        sb.append("\n").append("=".repeat(70)).append("\n");
        sb.append("PAPER Table III (NYC 300M):\n");
        sb.append("-".repeat(50)).append("\n");
        sb.append(String.format(Locale.ROOT, "%-12s %12s %12s %12s%n",
                "Variant", "Point(ms)", "Range(ms)", "kNN(ms)"));
        sb.append("-".repeat(50)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-12s %12s %12s %12s%n", "LiLIS-F", "218", "704", "1107"));
        sb.append(String.format(Locale.ROOT,
                "%-12s %12s %12s %12s%n", "LiLIS-A", "199", "521", "1767"));
        sb.append(String.format(Locale.ROOT,
                "%-12s %12s %12s %12s%n", "LiLIS-Q", "141", "341", "774"));
        sb.append(String.format(Locale.ROOT,
                "%-12s %12s %12s %12s%n", "LiLIS-K", "83",  "469", "650"));
        sb.append(String.format(Locale.ROOT,
                "%-12s %12s %12s %12s%n", "LiLIS-R", "77",  "472", "619"));
        sb.append("\nTakeaway 2: Tree-based (K,R) outperforms Grid-based (F,A)\n");

        sc.close();

        System.out.println("\n" + sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(outFile))) {
            w.write(sb.toString());
        }
        System.out.println("Saved: " + outFile);
    }

    static double avg(long[] a) {
        long s = 0; for (long v : a) s += v; return (double) s / a.length;
    }
}
