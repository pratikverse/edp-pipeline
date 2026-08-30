#!/bin/bash
cd /d/Projects/edp
source activate edp 2>/dev/null
echo "[queue] waiting for realdata training / GPU..."
until [ -f data/models/realdata_symbol_detector.pt ] || grep -qiE "error:|traceback" outputs/realdata_train.log; do sleep 30; done
sleep 15
echo "[queue] starting pid detector training"
python scripts/train_yolo.py --data data/synth_pid/data.yaml --epochs 40 --batch 8 --name pid_synthetic_yolo > outputs/pid_train.log 2>&1
cp outputs/yolo_runs/pid_synthetic_yolo/weights/best.pt data/models/pid_synthetic_yolo.pt 2>/dev/null && echo "[queue] pid_synthetic_yolo.pt written"
