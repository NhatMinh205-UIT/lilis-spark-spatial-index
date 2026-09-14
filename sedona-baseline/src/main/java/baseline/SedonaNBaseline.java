package baseline;

import org.apache.sedona.core.spatialOperator.KNNQuery;
import org.apache.sedona.core.spatialOperator.RangeQuery;
import org.apache.sedona.core.spatialRDD.PointRDD;
import org.apache.sedona.common.enums.FileDataSplitter;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
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
 * SedonaNBaseline — Apache Sedona No-Index (Sedona-N)
 * Package: baseline (project riêng, không conflict với LiLIS)
 *
 * Sedona-N = brute-force scan, KHÔNG có spatial index
 * So sánh với LiLIS-K → chứng minh Takeaway 1 paper
 *
 * Args: label path TOTAL_N MIN_X MAX_X MIN_Y MAX_Y
 */
public class SedonaNBaseline {

    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final int    KNN_K       = 10;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;

    static final GeometryFactory GF = new GeometryFactory();

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: SedonaNBaseline " +
                "<label> <path> <TOTAL_N> <MIN_X> <MAX_X> <MIN_Y> <MAX_Y>");
            System.exit(1);
        }

        String label = args[0];
        String path  = args[1];
        double minX  = Double.parseDouble(args[3]);
        double maxX  = Double.parseDouble(args[4]);
        double minY  = Double.parseDouble(args[5]);
        double maxY  = Double.parseDouble(args[6]);

        System.out.printf("=== Sedona-N Baseline === %n");
        System.out.printf("Dataset: %s%n", label);
        System.out.printf("BBox: X[%.4f,%.4f] Y[%.4f,%.4f]%n",
                minX, maxX, minY, maxY);

        SparkConf conf = new SparkConf().setAppName("SparkApp");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load data với Sedona PointRDD (col 0 = lon, col 1 = lat)
        System.out.println("[Bước 0] Loading Sedona PointRDD...");
        PointRDD pointRDD = new PointRDD(
                sc, path, 0, FileDataSplitter.CSV, false);

        // BẮT BUỘC 1: Phân tích dữ liệu để lấy Bounding Box
        // Nếu không có hàm này, RangeQuery sẽ bỏ qua quét do tưởng tập dữ liệu trống
        System.out.println("[Bước 0.1] Analyzing RDD bounding box...");
        pointRDD.analyze();
        System.out.println("Tổng số điểm: " + pointRDD.approximateTotalCount);

        // BẮT BUỘC 2: Nạp dữ liệu vào RAM (Cache)
        // Đảm bảo Sedona-N chạy trên RAM thay vì đọc lại file CSV ở mỗi vòng lặp
        System.out.println("[Bước 0.2] Caching data to memory...");
        pointRDD.rawSpatialRDD.cache();
        pointRDD.rawSpatialRDD.count(); // Action ép Spark thực thi cache

        // KHÔNG gọi buildIndex() → Sedona-N
        System.out.println("[Bước 0] NO index built → Sedona-N mode");

        // Tạo query points ngẫu nhiên
        Random rng = new Random(SAMPLE_SEED);
        List<Point>    qPoints = new ArrayList<>();
        List<Envelope> qRanges = new ArrayList<>();

        for (int i = 0; i < RUNS + WARMUP; i++) {
            double cx = minX + rng.nextDouble() * (maxX - minX);
            double cy = minY + rng.nextDouble() * (maxY - minY);
            qPoints.add(GF.createPoint(new Coordinate(cx, cy)));
            qRanges.add(new Envelope(
                    cx - RANGE_HALF, cx + RANGE_HALF,
                    cy - RANGE_HALF, cy + RANGE_HALF));
        }

        // Warm-up
        System.out.println("[Warm-up] " + WARMUP + " runs...");
        for (int w = 0; w < WARMUP; w++) {
            RangeQuery.SpatialRangeQuery(pointRDD, qRanges.get(w), false, false).count();
            KNNQuery.SpatialKnnQuery(pointRDD, qPoints.get(w), KNN_K, false);
        }

        // Benchmark
        long[] pointT = new long[RUNS];
        long[] rangeT = new long[RUNS];
        long[] knnT   = new long[RUNS];

        System.out.println("[Benchmark] " + RUNS + " timed runs...");
        for (int i = 0; i < RUNS; i++) {
            int    qi = i + WARMUP;
            Point    qp = qPoints.get(qi);
            Envelope qr = qRanges.get(qi);
            long t0, t1;

            // Point Query — brute force filter
            final double qx = qp.getX(), qy = qp.getY();
            t0 = System.currentTimeMillis();
            pointRDD.rawSpatialRDD.filter(g ->
                Math.abs(g.getCoordinate().x - qx) < 1e-9 &&
                Math.abs(g.getCoordinate().y - qy) < 1e-9
            ).count();
            t1 = System.currentTimeMillis();
            pointT[i] = t1 - t0;

            // Range Query — Sedona-N (useIndex=false)
            t0 = System.currentTimeMillis();
            RangeQuery.SpatialRangeQuery(pointRDD, qr, false, false).count();
            t1 = System.currentTimeMillis();
            rangeT[i] = t1 - t0;

            // kNN Query — Sedona-N (useApprox=false)
            t0 = System.currentTimeMillis();
            KNNQuery.SpatialKnnQuery(pointRDD, qp, KNN_K, false);
            t1 = System.currentTimeMillis();
            knnT[i] = t1 - t0;

            System.out.printf("[Run %2d/%d] point=%dms range=%dms knn=%dms%n",
                    i+1, RUNS, pointT[i], rangeT[i], knnT[i]);
        }

        sc.close();

        // Ghi kết quả
        java.io.File dir = new java.io.File("results");
        if (!dir.exists()) dir.mkdirs();
        String outFile = "results/" + label + "_sedona_n_cluster_3nodes.txt";
        writeResults(outFile, label, pointT, rangeT, knnT);
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
            long[] pt, long[] rng, long[] knn) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("=".repeat(70)).append("\n");
        sb.append("SEDONA-N BASELINE — ").append(label).append("\n");
        sb.append("Runs=").append(RUNS)
          .append(" | Warmup=").append(WARMUP)
          .append(" | kNN k=").append(KNN_K).append("\n");
        sb.append("Mode: Sedona-N — brute-force scan, NO spatial index\n");
        sb.append("=".repeat(70)).append("\n\n");

        sb.append(String.format(Locale.ROOT,
                "%-22s %10s %10s %10s %10s%n",
                "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Point  Sedona-N", avg(pt), min(pt), max(pt), std(pt)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "Range  Sedona-N", avg(rng), min(rng), max(rng), std(rng)));
        sb.append(String.format(Locale.ROOT,
                "%-22s %10.1f %10d %10d %10.1f%n",
                "kNN    Sedona-N", avg(knn), min(knn), max(knn), std(knn)));

        sb.append("\n").append("=".repeat(70)).append("\n");
        sb.append("LiLIS-K vs Sedona-N (").append(label).append("):\n");
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format(Locale.ROOT,
                "  Point  LiLIS-K: ~780ms  | Sedona-N: %.0fms → %.0fx faster%n",
                avg(pt), avg(pt) / 780.0));
        sb.append(String.format(Locale.ROOT,
                "  Range  LiLIS-K: ~743ms  | Sedona-N: %.0fms → %.0fx faster%n",
                avg(rng), avg(rng) / 743.0));
        sb.append(String.format(Locale.ROOT,
                "  kNN    LiLIS-K: ~566ms  | Sedona-N: %.0fms → %.0fx faster%n",
                avg(knn), avg(knn) / 566.0));
        sb.append("\nPaper (NYC 300M): Sedona-N Range ~521,282ms vs LiLIS 472ms = 1,100x\n");

        System.out.println("\n" + sb);
        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) {
            w.write(sb.toString());
        }
    }
}