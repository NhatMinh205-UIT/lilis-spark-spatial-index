package baseline;
import org.apache.sedona.common.enums.FileDataSplitter; // Đã fix sang common 
import org.apache.sedona.core.enums.GridType;           // Giữ nguyên ở core
import org.apache.sedona.core.enums.IndexType;          // Giữ nguyên ở core
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
 * SedonaRQBaseline — Apache Sedona R-tree + Quad-tree (Sedona-RQ)
 *
 * Sedona-RQ = R-tree spatial index (hỗ trợ kNN) + Quad-tree spatial partitioner
 * * Khác với Sedona-QQ, R-Tree CÓ HỖ TRỢ kNN, vì vậy bài test này sẽ đo lường
 * đầy đủ cả 3 loại truy vấn: Point, Range và kNN.
 */
public class SedonaRQBaseline {

    // ── Cấu hình ────────────────────────────────────────────────────
    static final int    RUNS        = 15;
    static final int    WARMUP      = 2;
    static final int    KNN_K       = 10;
    static final long   SAMPLE_SEED = 42L;
    static final double RANGE_HALF  = 0.0065;  // selectivity ~0.001%
    static final double POINT_EPS   = 1e-6;    // Epsilon box cho Point query

    static final GeometryFactory GF = new GeometryFactory();

    public static void main(String[] args) throws Exception {

        if (args.length < 7) {
            System.err.println("Usage: SedonaRQBaseline " +
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
        System.out.println("Sedona-RQ Baseline (R-Tree Index + Quad-tree Partitioner)");
        System.out.printf("Dataset: %s%n", label);
        System.out.println("=".repeat(65));

        // ── Init Spark ───────────────────────────────────────────────
        SparkConf conf = new SparkConf().setAppName("SparkApp_SedonaRQ");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // ── BƯỚC 1: Tạo query points ngẫu nhiên ─────────────────────
        Random rng = new Random(SAMPLE_SEED);
        List<Point>    qPoints      = new ArrayList<>();
        List<Envelope> qRanges      = new ArrayList<>();
        List<Envelope> qPointBoxes  = new ArrayList<>();

        for (int i = 0; i < RUNS + WARMUP; i++) {
            double cx = minX + rng.nextDouble() * (maxX - minX);
            double cy = minY + rng.nextDouble() * (maxY - minY);
            qPoints.add(GF.createPoint(new Coordinate(cx, cy)));
            qRanges.add(new Envelope(cx - RANGE_HALF, cx + RANGE_HALF, cy - RANGE_HALF, cy + RANGE_HALF));
            qPointBoxes.add(new Envelope(cx - POINT_EPS, cx + POINT_EPS, cy - POINT_EPS, cy + POINT_EPS));
        }

        // ── BƯỚC 2: Load PointRDD ────────────────────────────────────
        System.out.println("[Bước 2] Loading PointRDD từ CSV...");
        PointRDD pointRDD = new PointRDD(sc, path, 0, FileDataSplitter.CSV, false);

        // ── BƯỚC 3: Build Index (Đã fix lỗi NullPointerException) ────
        System.out.println("[Bước 3] Building Sedona-RQ index...");
        long buildStart = System.currentTimeMillis();

        pointRDD.analyze();
        
        // Cấu hình RQ: Partitioner là Quad-Tree
        pointRDD.spatialPartitioning(GridType.QUADTREE);
        
        // Cấu hình RQ: Index là R-Tree. 
        // Bắt buộc dùng tham số `false` để xây trên indexedRawRDD phục vụ Single Query
        pointRDD.buildIndex(IndexType.RTREE, false);

        // Đẩy vào RAM và kích hoạt chạy
        pointRDD.indexedRawRDD.persist(StorageLevel.MEMORY_AND_DISK_SER());
        long indexCount = pointRDD.indexedRawRDD.count();

        long buildEnd  = System.currentTimeMillis();
        long buildTime = buildEnd - buildStart;

        System.out.printf("[Bước 3] Index RTREE BUILT: %d partitions. Time: %,dms%n", indexCount, buildTime);

        // ── BƯỚC 4: Warm-up queries ──────────────────────────────────
        System.out.println("[Warm-up] Đang chạy " + WARMUP + " lần với R-Tree...");
        for (int w = 0; w < WARMUP; w++) {
            RangeQuery.SpatialRangeQuery(pointRDD, qPointBoxes.get(w), false, true).count();
            RangeQuery.SpatialRangeQuery(pointRDD, qRanges.get(w), false, true).count();
            KNNQuery.SpatialKnnQuery(pointRDD, qPoints.get(w), KNN_K, true);
        }

        // ── BƯỚC 5: Benchmark ────────────────────────────────────────
        long[] pointT = new long[RUNS];
        long[] rangeT = new long[RUNS];
        long[] knnT   = new long[RUNS];

        System.out.println("[Benchmark] Bắt đầu đo lường 10 lần chạy chính thức...");
        for (int i = 0; i < RUNS; i++) {
            int qi = i + WARMUP;
            long t0, t1;

            // 1. Point Query (Dùng R-Tree Index)
            t0 = System.currentTimeMillis();
            RangeQuery.SpatialRangeQuery(pointRDD, qPointBoxes.get(qi), false, true).count();
            t1 = System.currentTimeMillis();
            pointT[i] = t1 - t0;

            // 2. Range Query (Dùng R-Tree Index)
            t0 = System.currentTimeMillis();
            RangeQuery.SpatialRangeQuery(pointRDD, qRanges.get(qi), false, true).count();
            t1 = System.currentTimeMillis();
            rangeT[i] = t1 - t0;

            // 3. kNN Query (Dùng R-Tree Index - Bí mật sức mạnh của Sedona-RQ)
            t0 = System.currentTimeMillis();
            KNNQuery.SpatialKnnQuery(pointRDD, qPoints.get(qi), KNN_K, true);
            t1 = System.currentTimeMillis();
            knnT[i] = t1 - t0;

            System.out.printf("[Run %2d/%d] point=%dms range=%dms knn=%dms%n",
                    i + 1, RUNS, pointT[i], rangeT[i], knnT[i]);
        }

        sc.close();

        // ── BƯỚC 6: Ghi kết quả ──────────────────────────────────────
        java.io.File dir = new java.io.File("results");
        if (!dir.exists()) dir.mkdirs();

        String outFile = "results/" + label + "_sedona_rq_cluster_3nodes.txt";
        writeResults(outFile, label, buildTime, pointT, rangeT, knnT);
        System.out.println("Saved: " + outFile);
    }

    // ── Helpers ───────────────────────────────────────────────────────
    static double avg(long[] a) { long s = 0; for (long v : a) s += v; return (double) s / a.length; }
    static long min(long[] a) { long m = a[0]; for (long v : a) if (v < m) m = v; return m; }
    static long max(long[] a) { long m = a[0]; for (long v : a) if (v > m) m = v; return m; }
    static double std(long[] a) {
        double mean = avg(a), sq = 0;
        for (long v : a) sq += (v - mean) * (v - mean);
        return Math.sqrt(sq / a.length);
    }

    // ── Write results ─────────────────────────────────────────────────
    static void writeResults(String file, String label,
            long buildTime, long[] pt, long[] rng, long[] knn) throws IOException {

        StringBuilder sb = new StringBuilder();
        sb.append("=".repeat(70)).append("\n");
        sb.append("SEDONA-RQ BASELINE — ").append(label).append("\n");
        sb.append("Index:       R-tree (IndexType.RTREE)\n");
        sb.append("Partitioner: Quad-tree (GridType.QUADTREE)\n");
        sb.append("Runs=").append(RUNS).append(" | Warmup=").append(WARMUP).append("\n");
        sb.append("=".repeat(70)).append("\n\n");

        sb.append("## BUILD TIME\n");
        sb.append(String.format(Locale.ROOT, "  Sedona-RQ build: %,dms (%.1fs)%n%n", buildTime, buildTime / 1000.0));

        sb.append("## QUERY TIME\n");
        sb.append(String.format(Locale.ROOT, "%-22s %10s %10s %10s %10s%n", "Query", "Avg(ms)", "Min", "Max", "Std"));
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format(Locale.ROOT, "%-22s %10.1f %10d %10d %10.1f%n", "Point  Sedona-RQ", avg(pt), min(pt), max(pt), std(pt)));
        sb.append(String.format(Locale.ROOT, "%-22s %10.1f %10d %10d %10.1f%n", "Range  Sedona-RQ", avg(rng), min(rng), max(rng), std(rng)));
        sb.append(String.format(Locale.ROOT, "%-22s %10.1f %10d %10d %10.1f%n", "kNN    Sedona-RQ", avg(knn), min(knn), max(knn), std(knn)));

        sb.append("\n## SPEEDUP CHECK (Lưu ý: Chỉ đối chiếu cùng môi trường Local/LAN)\n");
        sb.append("So sánh tốc độ Query với LiLIS-K để xem R-Tree có đấu lại được Learned Index ở bài test kNN hay không.\n");

        try (BufferedWriter w = new BufferedWriter(new FileWriter(file))) { w.write(sb.toString()); }
    }
}