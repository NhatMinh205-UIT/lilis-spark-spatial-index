#!/bin/bash
# ============================================================
# run_query_benchmark.sh
# Chạy Query Benchmark tuần tự: CHI → NYC → SYN
# Reproduce paper RQ1 (speedup) + RQ4 (kNN sweep + selectivity)
# ============================================================

# ── Config cluster ──────────────────────────────────────────
MASTER="spark://192.168.0.202:7077"
JAR="/home/pc/workspace/learned-index-spark/target/LearnIndexSpark-1.0-SNAPSHOT-jar-with-dependencies.jar"
CLASS="idnexbuild.QueryBenchmarkComparison"

SPARK_SUBMIT="spark-submit \
  --master $MASTER \
  --class $CLASS \
  --driver-memory 4g \
  --executor-memory 8g \
  --executor-cores 3 \
  --conf spark.driver.host=192.168.0.202 \
  --conf spark.default.parallelism=64 \
  --conf spark.locality.wait=0s \
  --conf spark.network.timeout=120s \
  --conf spark.executor.heartbeatInterval=30s \
  --conf spark.shuffle.io.maxRetries=3 \
  --conf spark.shuffle.io.retryWait=10s \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.rdd.compress=true"

# ── Bounding boxes (chạy lệnh dưới để verify trước khi submit) ──
# python3 -c "
# import subprocess
# for f,n in [('chi_full','chi'),('nyc_full','nyc'),('syn_full','syn')]:
#   r=subprocess.run(['awk','-F,','NR>0{if(NR==1||(\$1<minx)){minx=\$1};...}',
#                    f'data/{f}_xy.csv'],capture_output=True,text=True)
# "

DATA_DIR="/home/pc/workspace/learned-index-spark/data"
RESULT_DIR="/home/pc/workspace/learned-index-spark/results"
LOG_DIR="$RESULT_DIR/cluster_logs"
mkdir -p "$LOG_DIR"

# ── Tham số từng dataset ────────────────────────────────────
# Format: LABEL PATH TOTAL_N MIN_X MAX_X MIN_Y MAX_Y

declare -A DATASETS

# CHI: Crime events Chicago
# Verify: awk -F, 'NR>0{if(NR==1||$1<mnx)mnx=$1;if($1>mxx)mxx=$1;if($2<mny)mny=$2;if($2>mxy)mxy=$2}END{print mnx,mxx,mny,mxy}' data/chi_full_xy.csv
CHI_N=7699698
CHI_MINX=-87.9401
CHI_MAXX=-87.5248
CHI_MINY=41.6445
CHI_MAXY=42.0231

# NYC: Taxi rides New York
NYC_N=46475157
NYC_MINX=-74.9996
NYC_MAXX=-72.1965
NYC_MINY=40.0081
NYC_MAXY=41.9237

# SYN: Uniform random points
# Verify bounding box từ data thực tế trước khi chạy
SYN_N=100000000
SYN_MINX=0.0
SYN_MAXX=1.0
SYN_MINY=0.0
SYN_MAXY=1.0

# ── Hàm chạy 1 dataset ──────────────────────────────────────
run_dataset() {
    local LABEL=$1
    local PATH=$2
    local N=$3
    local MINX=$4
    local MAXX=$5
    local MINY=$6
    local MAXY=$7

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  QUERY BENCHMARK: $LABEL"
    echo "║  N=$N | BBox: X[$MINX,$MAXX] Y[$MINY,$MAXY]"
    echo "╚══════════════════════════════════════════════════╝"
    echo "Bắt đầu lúc: $(date '+%Y-%m-%d %H:%M:%S')"

    # Backup kết quả cũ nếu có
    if [ -f "$RESULT_DIR/${LABEL}_query_comparison.txt" ]; then
        cp "$RESULT_DIR/${LABEL}_query_comparison.txt" \
           "$RESULT_DIR/${LABEL}_query_comparison_backup_$(date '+%Y%m%d_%H%M%S').txt"
        echo "Backed up old results."
    fi

    # Chạy spark-submit
    $SPARK_SUBMIT \
        $JAR \
        "$LABEL" \
        "$PATH" \
        "$N" \
        "$MINX" "$MAXX" \
        "$MINY" "$MAXY" \
        2>&1 | tee "$LOG_DIR/${LABEL}_query_cluster.log"

    local EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ $LABEL DONE lúc $(date '+%H:%M:%S')"
        echo "   Kết quả: $RESULT_DIR/${LABEL}_query_comparison.txt"
    else
        echo ""
        echo "❌ $LABEL FAILED (exit code: $EXIT_CODE)"
        echo "   Log: $LOG_DIR/${LABEL}_query_cluster.log"
    fi

    return $EXIT_CODE
}

# ── Main ────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║        QUERY BENCHMARK — 3 DATASETS                 ║"
echo "║        Cluster: 192.168.0.202 (Master)              ║"
echo "║                 192.168.0.59  (Huy - Worker)        ║"
echo "║                 192.168.0.169 (Hùng - Worker)       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Thời gian ước tính:"
echo "  CHI (~7.7M):  30-45 phút"
echo "  NYC (~46.5M): 60-90 phút"
echo "  SYN (~100M):  90-150 phút"
echo "  Tổng:         ~3-5 giờ"
echo ""

# Bước 0: Verify bounding boxes trước khi chạy
echo "=== VERIFY BOUNDING BOXES ==="
echo "Đang tính bounding box từ data (mất 1-2 phút)..."

for DATASET in chi_full nyc_full syn_full; do
    CSV="$DATA_DIR/${DATASET}_xy.csv"
    if [ -f "$CSV" ]; then
        BOUNDS=$(awk -F, 'NR==1{mnx=$1;mxx=$1;mny=$2;mxy=$2}
                          NR>1{if($1<mnx)mnx=$1; if($1>mxx)mxx=$1;
                               if($2<mny)mny=$2; if($2>mxy)mxy=$2}
                          END{printf "minX=%.4f maxX=%.4f minY=%.4f maxY=%.4f",
                              mnx,mxx,mny,mxy}' "$CSV")
        echo "  $DATASET: $BOUNDS"
    else
        echo "  $DATASET: FILE NOT FOUND at $CSV"
    fi
done

echo ""
echo "⚠️  So sánh với tham số hardcode ở trên và sửa nếu cần!"
echo "Nhấn Enter để tiếp tục hoặc Ctrl+C để dừng..."
read -r

# Chạy tuần tự CHI → NYC → SYN
START_TIME=$(date '+%s')

# --- ĐÃ BỎ QUA CHI ---
# run_dataset "chi_full" "$DATA_DIR/chi_full_xy.csv" $CHI_N $CHI_MINX $CHI_MAXX $CHI_MINY $CHI_MAXY
# echo "Nghỉ 30 giây giữa các dataset..."
# sleep 30

echo "Đang chạy Full Parameter Sweep cho tập NYC..."
run_dataset "nyc_full" "$DATA_DIR/nyc_full_xy.csv" \
    $NYC_N $NYC_MINX $NYC_MAXX $NYC_MINY $NYC_MAXY

echo "Nghỉ 30 giây giữa các dataset..."
sleep 30

# --- ĐÃ BỎ QUA SYN ---
# run_dataset "syn_full" "$DATA_DIR/syn_full_xy.csv" $SYN_N $SYN_MINX $SYN_MAXX $SYN_MINY $SYN_MAXY
# Tổng kết
END_TIME=$(date '+%s')
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              TẤT CẢ ĐÃ XONG!                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Tổng thời gian: ${ELAPSED} phút"
echo ""
echo "Files kết quả:"
ls -lh "$RESULT_DIR"/*_query_comparison.txt 2>/dev/null
echo ""
echo "Chạy lệnh sau để xem tổng hợp:"
echo "  grep -A5 'SPEEDUP' $RESULT_DIR/*_query_comparison.txt"
