#!/bin/bash

# Setup logging
source scripts/logging_utils.sh
setup_log "evaluation"

echo "=========================================="
echo "MoxinVLA Evaluation - LIBERO Object"
echo "Started: $(date)"
echo "=========================================="

# Environment setup
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="e5cbd387d8e0c181e93c7e4ec56e965c5115e94c"
export WANDB_CONFIG_DIR="/home/user1/.config/wandb_arash"
export WANDB_CACHE_DIR="/home/user1/.cache/wandb_arash"
export PRISMATIC_DATA_ROOT=/home/user1/aras_prism_training/OXE/
export PYTHONPATH=/home/user1/arashwork/moxinvla-oft
export CUDA_VISIBLE_DEVICES="3"
export PYTHONPATH=/home/user1/arashwork/moxinvla-oft/LIBERO:$PYTHONPATH

# Run evaluation
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./logs/libero_finetune_logs_checkpoints/moxin-7b-oxe-250k-image_aug-libero-object_no_film_proprio_1_img_3e-4/moxin-7b-224px+mx-oxe-magic-soup+n1+b32+x7-250k--image_aug+libero_object_no_noops+b16+lr-0.0003+lora-r32--2_imgs--proprio--90000_chkpt/ \
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