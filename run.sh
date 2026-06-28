#!/bin/bash
# YOLO11-OBB 遥感车辆检测 — 一键训练脚本
# 用法: bash run.sh <模式> [fold] [可选参数]

set -e

cd "$(dirname "$0")"
PROJECT=$(pwd)

# ── 默认配置 ──
FOLD="${2:-0}"
DEVICE="${DEVICE:-0,1,2,3,4,5,6,7}"
GPUS=$(echo "$DEVICE" | tr ',' '\n' | wc -l)

# ── 模式 ──
case "$1" in

    # ======== 单机8卡 ========

    baseline)
        # ★ 标准 YOLO11-OBB, 3层检测, BCE Loss
        DESC="Baseline"
        python train.py \
            --fold "$FOLD" --baseline \
            --batch $((GPUS * 8)) --epochs 200 \
            --name "baseline_fold${FOLD}" \
            --device "$DEVICE"
        ;;

    p2)
        # ★★★★★ P2检测层 (stride 4) — 针对15px小目标
        DESC="P2"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --batch $((GPUS * 16)) --epochs 500 --imgsz 640 \
            --cos-lr --close-mosaic 10 \
            --name "p2_fold${FOLD}" --device "$DEVICE"
        ;;

    enhanced)
        # ★ 增强版: P2 + 数据增强 (读取 configs/enhanced.yaml)
        DESC="Enhanced"
        python train.py \
            --fold "$FOLD" --enhanced \
            --device "$DEVICE" \
            --name "enhanced_fold${FOLD}"
        ;;

    final)
        # ★★★★★ 冲榜配置: 大模型 + 大图 + 全增强
        DESC="Final-Submission"
        MODEL="yolo11x-obb-p2.yaml"
        python train.py \
            --fold "$FOLD" \
            --model "$MODEL" \
            --use-focal --focal-gamma 2.0 \
            --cls-pw 0.5 \
            --cos-lr --epochs 500 \
            --imgsz 1024 --batch $((GPUS * 4)) \
            --multi-scale 0.1 --mosaic9 0.2 --mixup 0.1 \
            --close-mosaic 20 \
            --device "$DEVICE" \
            --name "final_fold${FOLD}_x_p2"
        ;;

    # ======== Ablation 实验 ========

    ab-p2)
        # P2 vs Baseline 对比
        DESC="Ablation-P2"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --batch $((GPUS * 8)) --epochs 200 --imgsz 640 \
            --name "abl_p2_fold${FOLD}" --device "$DEVICE"
        ;;

    ab-focal)
        # P2 + Focal Loss
        DESC="Ablation-P2+Focal"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --use-focal --focal-gamma 2.0 \
            --batch $((GPUS * 8)) --epochs 200 --imgsz 640 \
            --name "abl_p2_focal_fold${FOLD}" --device "$DEVICE"
        ;;

    ab-clspw)
        # P2 + 类别权重
        DESC="Ablation-P2+ClsPW"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --cls-pw 0.5 \
            --batch $((GPUS * 8)) --epochs 200 --imgsz 640 \
            --name "abl_p2_clspw_fold${FOLD}" --device "$DEVICE"
        ;;

    ab-imgsz)
        # 大尺寸输入
        DESC="Ablation-P2+ImgSz800"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --use-focal --cls-pw 0.5 --imgsz 800 --batch $((GPUS * 6)) \
            --epochs 200 --name "abl_p2_imgsz800_fold${FOLD}" --device "$DEVICE"
        ;;

    ab-mosaic9)
        # 3x3 Mosaic
        DESC="Ablation-P2+Mosaic9"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --use-focal --cls-pw 0.5 --mosaic9 0.2 \
            --batch $((GPUS * 8)) --epochs 200 --imgsz 640 \
            --name "abl_p2_m9_fold${FOLD}" --device "$DEVICE"
        ;;

    ab-coslr)
        # Cosine LR + 更长训练
        DESC="Ablation-P2+CosLR300"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --use-focal --cls-pw 0.5 --cos-lr --epochs 300 \
            --batch $((GPUS * 8)) --imgsz 640 \
            --name "abl_p2_coslr_fold${FOLD}" --device "$DEVICE"
        ;;

    ab-kld)
        # KLD Angle Loss
        DESC="Ablation-P2+KLD"
        python train.py \
            --fold "$FOLD" --model yolo11n-obb-p2.yaml \
            --use-focal --cls-pw 0.5 --use-kld-angle \
            --batch $((GPUS * 8)) --epochs 200 --imgsz 640 \
            --name "abl_p2_kld_fold${FOLD}" --device "$DEVICE"
        ;;

    # ======== 全 5 折交叉验证 ========

    cv-baseline)
        for f in 0 1 2 3 4; do
            echo "=== Fold $f ==="
            bash "$0" baseline "$f"
        done
        ;;

    cv-enhanced)
        for f in 0 1 2 3 4; do
            echo "=== Fold $f ==="
            bash "$0" enhanced "$f"
        done
        ;;

    # ======== 全部 Ablation 一键运行 ========

    all-ab)
        for exp in ab-p2 ab-focal ab-clspw ab-imgsz ab-mosaic9 ab-coslr ab-kld; do
            echo "===== $exp ====="
            bash "$0" "$exp" "$FOLD"
        done
        ;;

    # ======== 推理测试 ========

    val)
        # 验证已有模型
        WEIGHTS="${2:-runs/obb/baseline_fold0/weights/best.pt}"
        python -c "
import sys; sys.path.insert(0, '$PROJECT/ultralytics_src')
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
model.val(data='dataset_yolo/fold_0/data.yaml', imgsz=800, device='$DEVICE')
"
        ;;

    predict)
        # 推理单张图
        WEIGHTS="${2}"
        IMG="${3}"
        python -c "
import sys; sys.path.insert(0, '$PROJECT/ultralytics_src')
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
results = model.predict('$IMG', imgsz=800)
results[0].save('predict_output.jpg')
print('✅ 推理结果保存至 predict_output.jpg')
"
        ;;

    *)
        echo ""
        echo "YOLO11-OBB 遥感车辆检测 — 训练脚本"
        echo "========================================"
        echo ""
        echo "用法: bash run.sh <模式> [fold] [参数]"
        echo ""
        echo "【训练模式】"
        echo "  baseline       标准 YOLO11-OBB (3层)"
        echo "  p2             P2检测层 ★★★★★"
        echo "  enhanced       P2 + Focal + cls_pw + 增强数据 ★★★★★"
        echo "  final          冲榜配置: x模型 + 1024 + 500epoch ★★★★★"
        echo ""
        echo "【Ablation 实验】"
        echo "  ab-p2          P2 单点验证"
        echo "  ab-focal       P2 + Focal Loss"
        echo "  ab-clspw       P2 + 类别权重"
        echo "  ab-imgsz       P2 + 大尺寸输入(800)"
        echo "  ab-mosaic9     P2 + 3×3 Mosaic"
        echo "  ab-coslr       P2 + Cosine LR(300ep)"
        echo "  ab-kld         P2 + KLD Angle Loss"
        echo ""
        echo "【交叉验证】"
        echo "  cv-baseline    5折 Baseline"
        echo "  cv-enhanced    5折 Enhanced"
        echo "  all-ab         一键运行所有 Ablation"
        echo ""
        echo "【推理】"
        echo "  val [weights]  验证模型"
        echo "  predict <weights> <img>  推理单图"
        echo ""
        echo "示例:"
        echo "  bash run.sh baseline 0"
        echo "  bash run.sh enhanced 0"
        echo "  bash run.sh final 0"
        echo "  bash run.sh ab-p2 0"
        echo "  bash run.sh val runs/obb/baseline_fold0/weights/best.pt"
        echo ""
        echo "GPU 设置: DEVICE=0,1,2,3 bash run.sh enhanced 0"
        ;;
esac

echo ""
echo "✅ $DESC 完成!"
