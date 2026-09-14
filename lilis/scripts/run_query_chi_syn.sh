#!/bin/bash
# run_query_chi_syn.sh — Chạy query benchmark CHI → SYN tự động

MASTER="spark://192.168.0.202:7077"
JAR="/home/pc/workspace/learned-index-spark/target/LearnIndexSpark-1.0-SNAPSHOT-jar-with-dependencies.jar"
DATA="/home/pc/workspace/learned-index-spark/data"
LOG_DIR="/home/pc/workspace/learned-index-spark/results/cluster_logs"
mkdir -p $LOG_DIR

SPARK_ARGS="
  --master $MASTER
  --class idnexbuild.QueryBenchmarkComparison
  --driver-memory 4g
  --executor-memory 8g
  --executor-cores 3
  --conf spark.driver.host=192.168.0.202
  --conf spark.default.parallelism=64
  --conf spark.locality.wait=0s
  --conf spark.network.timeout=120s
  --conf spark.executor.heartbeatInterval=30s
  --conf spark.shuffle.io.maxRetries=3
  --conf spark.shuffle.io.retryWait=10s
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
  --conf spark.rdd.compress=true"

echo "========================================"
echo "CHI bắt đầu lúc: $(date '+%H:%M:%S')"
echo "========================================"

spark-submit $SPARK_ARGS $JAR \
  chi_full $DATA/chi_full_xy.csv \
  7699698 -91.6866 -87.5245 36.6194 42.0229 \
  2>&1 | tee $LOG_DIR/chi_full_query_cluster.log

CHI_EXIT=${PIPESTATUS[0]}
echo ""
if [ $CHI_EXIT -eq 0 ]; then
    echo "✅ CHI xong lúc: $(date '+%H:%M:%S')"
else
    echo "❌ CHI FAILED (exit=$CHI_EXIT) — vẫn tiếp tục SYN"
fi

echo "Nghỉ 30 giây..."
sleep 30

echo "========================================"
echo "SYN bắt đầu lúc: $(date '+%H:%M:%S')"
echo "========================================"

spark-submit $SPARK_ARGS $JAR \
  syn_full $DATA/syn_full_xy.csv \
  100000000 0.0 1.0 0.0 1.0 \
  2>&1 | tee $LOG_DIR/syn_full_query_cluster.log

SYN_EXIT=${PIPESTATUS[0]}
echo ""
if [ $SYN_EXIT -eq 0 ]; then
    echo "✅ SYN xong lúc: $(date '+%H:%M:%S')"
else
    echo "❌ SYN FAILED (exit=$SYN_EXIT)"
fi

echo ""
echo "========================================"
echo "TẤT CẢ XONG lúc: $(date '+%H:%M:%S')"
echo "========================================"
echo "Kết quả:"
ls -lh /home/pc/workspace/learned-index-spark/results/*chi*query* \
        /home/pc/workspace/learned-index-spark/results/*syn*query* 2>/dev/null
