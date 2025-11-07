#!/bin/bash

# Setup logging
source scripts/logging_utils.sh
setup_log "evaluation"

echo "=========================================="
echo "MoxinVLA Evaluation - LIBERO Object"
echo "Started: $(date)"
echo "=========================================="

# Environment setup

export CUDA_VISIBLE_DEVICES="0"
# export PYTHONPATH=/home/user1/arashwork/moxinvla-oft/LIBERO:$PYTHONPATH

# Run evaluation
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "YOUR PRETRAINED CHECKPOINT" \
  --task_suite_name libero_object \
  --num_images_in_input 2 \
  --use_film False \
  --use_proprio True \
  --center_crop True 2>&1 | tee "$LOG_FILE"

# Check status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed. Check log: $LOG_FILE"
    exit 1
fi 