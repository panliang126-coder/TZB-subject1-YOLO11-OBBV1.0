#!/bin/bash
# Mosaic 实验自动连跑: C→D→E→F
# C 当前在 8 卡续训中 (epoch 159→500), 完成后自动跑 D/E/F
# 所有实验统一: 8×A100, batch=96, 500 epochs
set -e

PROJECT_DIR="/data/work1/panliang/2026/00_TZB_game/YOLOv11"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/runs/.mosaic_logs"
mkdir -p "$LOG_DIR"

# ============================================================
# 实验列表 (按顺序): "name:close_mosaic"
# ============================================================
EXPERIMENTS=(
    "exp_mosaic_D_close300:300"
    "exp_mosaic_E_close500:500"
    "exp_mosaic_F_never:0"
)

run_test_eval() {
    local NAME="$1"
    local BEST_PT="$PROJECT_DIR/runs/$NAME/weights/best.pt"
    if [ -f "$BEST_PT" ]; then
        echo "  📊 Test set 评估: $NAME ..."
        python -c "
from ultralytics import YOLO
m = YOLO('$BEST_PT')
m.val(data='$PROJECT_DIR/dataset_yolo/fold_0/data.yaml', split='test')
" 2>&1 | tee -a "$LOG_DIR/${NAME}.log"
        echo "  ✅ Test 评估完成: $NAME"
    else
        echo "  ⚠️ 未找到 $BEST_PT, 跳过 test 评估"
    fi
}

wait_for_epoch() {
    local NAME="$1"
    local TARGET="$2"
    local RESULTS="$PROJECT_DIR/runs/$NAME/results.csv"
    while true; do
        if [ -f "$RESULTS" ]; then
            local CUR=$(tail -1 "$RESULTS" 2>/dev/null | cut -d, -f1)
            if [ "$CUR" -ge "$TARGET" ] 2>/dev/null; then
                echo "  ✅ $NAME 已达到 epoch $CUR (目标 $TARGET)"
                return 0
            fi
            echo "  ⏳ $NAME: epoch $CUR / $TARGET ... $(date '+%H:%M:%S')"
        else
            echo "  ⏳ $NAME: 等待 results.csv ... $(date '+%H:%M:%S')"
        fi
        sleep 300  # 每 5 分钟检查一次
    done
}

# ============================================================
# Step 0: 等待 C 完成
# ============================================================
echo "================================================================"
echo "  Mosaic 自动连跑: C→D→E→F (4×A100: GPU 0-3, batch=48)"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "  ⏳ Step 0: 等待 C (exp_mosaic_C_close100-2) 完成 500 epoch..."

RESULTS_C="$PROJECT_DIR/runs/exp_mosaic_C_close100-2/results.csv"
while true; do
    if [ -f "$RESULTS_C" ]; then
        CUR_C=$(tail -1 "$RESULTS_C" 2>/dev/null | cut -d, -f1)
        if [ "$CUR_C" -ge 500 ] 2>/dev/null; then
            echo "  ✅ C 已完成! epoch=$CUR_C"
            break
        fi
        echo "  ⏳ C: epoch $CUR_C / 500 ... $(date '+%H:%M:%S')"
    else
        echo "  ⏳ C: 等待 results.csv ... $(date '+%H:%M:%S')"
    fi
    sleep 300
done

# C test 评估
echo ""
echo "  📊 C Test 评估..."
run_test_eval "exp_mosaic_C_close100-2"

# ============================================================
# Step 1-3: 串行跑 D→E→F
# ============================================================
for entry in "${EXPERIMENTS[@]}"; do
    NAME="${entry%%:*}"
    CM="${entry##*:}"

    LOG_FILE="$LOG_DIR/${NAME}.log"
    echo ""
    echo "================================================================"
    echo "  ▶ 开始实验: $NAME (close_mosaic=$CM)"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  日志: $LOG_FILE"
    echo "================================================================"

    # 清理僵尸进程
    echo "  🧹 清理残留进程..."
    ps aux | grep -E "multiprocessing|forkserver|resource_tracker|torch.distributed" | grep $(whoami) | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 5

    # 训练 (8 卡, batch=96)
    echo "  🚀 启动训练 (8×A100: GPU 0-7, batch=96)..."
    python train.py \
        --cfg configs/exp_mosaic_base.yaml \
        --fold 0 \
        --name "$NAME" \
        --close-mosaic "$CM" \
        --batch 48 \
        --device 0,1,2,3 \
        2>&1 | tee "$LOG_FILE"

    # 验证训练完成
    BEST_PT="$PROJECT_DIR/runs/$NAME/weights/best.pt"
    if [ -f "$BEST_PT" ]; then
        echo "  ✅ 训练完成: $NAME"
        run_test_eval "$NAME"
    else
        echo "  ❌ 训练失败: 未找到 $BEST_PT"
        echo "  继续下一组..."
    fi
done

# ============================================================
# 最终: 生成对比报告
# ============================================================
echo ""
echo "================================================================"
echo "  🎉 全部实验完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "  📊 生成 Mosaic 对比报告..."
python tools/benchmark_mosaic.py 2>&1 | tee "$LOG_DIR/benchmark_report.log"
echo ""
echo "================================================================"
echo "  ✅ 连跑完成! 查看报告: $LOG_DIR/benchmark_report.log"
echo "================================================================"
