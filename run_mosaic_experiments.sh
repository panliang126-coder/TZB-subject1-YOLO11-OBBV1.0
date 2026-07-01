#!/bin/bash
# Mosaic 调度策略对比实验 — 串行执行脚本 (4×A100)
# 基于 exp1_focal_v10 配置，唯一变量：close_mosaic
#
# GPU: 0,1,2,3 (前4卡) — 后4卡保留不动
# batch: 48 (12 img/GPU, 与8卡batch=96的per-GPU负载一致)
#
# 实验矩阵:
#   方案A: close_mosaic=15   ✅ 已完成 (8卡, 500ep)
#   方案B: close_mosaic=30   (= exp1_focal_v10, 作为参考)
#   方案C: close_mosaic=100  (中期关闭)
#   方案D: close_mosaic=300  (晚期关闭)
#   方案E: close_mosaic=500  (最后500ep关闭, 等价永不)
#   方案F: close_mosaic=0    (永不关闭, 官方推荐)

set -e

PROJECT_DIR="/data/work1/panliang/2026/00_TZB_game/YOLOv11"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/runs/.mosaic_logs"
mkdir -p "$LOG_DIR"

# 实验列表: "name:close_mosaic"
EXPERIMENTS=(
    "exp_mosaic_C_close100:100"
    "exp_mosaic_D_close300:300"
    "exp_mosaic_E_close500:500"
    "exp_mosaic_F_never:0"
)

echo "================================================================"
echo "  Mosaic 调度策略对比实验 (4×A100: GPU 0-3)"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  共 ${#EXPERIMENTS[@]} 组实验, batch=48"
echo "================================================================"

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

    # 清理僵尸进程 (避免 DDP pickle 截断)
    echo "  🧹 清理残留进程..."
    ps aux | grep -E "multiprocessing|forkserver|resource_tracker|torch.distributed" | grep $(whoami) | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 3

    # 训练
    echo "  🚀 启动训练 (GPU 0-3, batch=48)..."
    python train.py \
        --cfg configs/exp_mosaic_base.yaml \
        --fold 0 \
        --name "$NAME" \
        --close-mosaic "$CM" \
        2>&1 | tee "$LOG_FILE"

    # 检查训练是否成功
    BEST_PT="$PROJECT_DIR/runs/$NAME/weights/best.pt"
    if [ -f "$BEST_PT" ]; then
        echo "  ✅ 训练完成, best.pt 已保存"

        # Test set 评估
        echo "  📊 Test set 评估..."
        python -c "
from ultralytics import YOLO
m = YOLO('$BEST_PT')
m.val(data='$PROJECT_DIR/dataset_yolo/fold_0/data.yaml', split='test')
" 2>&1 | tee -a "$LOG_FILE"
        echo "  ✅ Test 评估完成"
    else
        echo "  ❌ 训练失败: 未找到 $BEST_PT"
        exit 1
    fi

    echo "  ✅ 实验 $NAME 完成: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "================================================================"
echo "  🎉 全部实验完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
