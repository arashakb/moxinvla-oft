#!/bin/bash

# Setup logging
source scripts/logging_utils.sh
setup_log "training"

echo "=========================================="
echo "MoxinVLA Fine-tuning - LIBERO Spatial"
echo "Started: $(date)"
echo "=========================================="

# Environment setup
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="e5cbd387d8e0c181e93c7e4ec56e965c5115e94c"
export WANDB_CONFIG_DIR="/home/user1/.config/wandb_arash"
export WANDB_CACHE_DIR="/home/user1/.cache/wandb_arash"
export PYTHONPATH=/home/user1/arashwork/moxinvla-oft
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Run training
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path moxin-hf-convert \
  --data_root_dir modified_libero_rlds_data \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir logs/checkpoints/moxin-libero-spatial-2img-3e-4 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 3e-4 \
  --use_film False \
  --use_proprio True \
  --num_images_in_input 1 \
  --resume False \
  --image_aug True \
  --save_latest_checkpoint_only False \
  --wandb_project OpenVLA-OFT-Moxin_7B_VLM_finetune_LIBERO_SPATIAL_8H100_testtt \
  --wandb_entity arash-akbari-stu-northeastern-university \
  --save_freq 10000 \
  --max_steps 50000 2>&1 | tee "$LOG_FILE"

# Check status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed. Check log: $LOG_FILE"
    exit 1
fi 