#!/bin/bash

# Setup logging
source scripts/logging_utils.sh
setup_log "training"

echo "=========================================="
echo "MoxinVLA Fine-tuning - LIBERO Spatial"
echo "Started: $(date)"
echo "=========================================="

# Environment setup
# export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="YOUR WANDB API KEY"
# export PYTHONPATH="YOUR PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Run training
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "YOUR VLA PATH (HF Converted Pretrained checkpoint)" \
  --data_root_dir modified_libero_rlds_data \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir "YOUR LOG AND CHECKPOINT DIR" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 3e-4 \
  --use_film False \
  --use_proprio True \
  --num_images_in_input 1 \
  --resume False \
  --image_aug True \
  --save_latest_checkpoint_only False \
  --wandb_project "YOUR WANDB ENTITY" \
  --wandb_entity "YOUR WANDB PROJECT" \
  --save_freq 10000 \
  --max_steps 50000 2>&1 | tee "$LOG_FILE"

# Check status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed. Check log: $LOG_FILE"
    exit 1
fi 