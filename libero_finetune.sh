#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs/training_logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/training_logs"
STDOUT_LOG="${LOG_DIR}/vla_training_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/vla_training_${TIMESTAMP}.err"


echo "Starting VLA training at ${TIMESTAMP}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"




export NCCL_P2P_DISABLE=1
export WANDB_API_KEY="e5cbd387d8e0c181e93c7e4ec56e965c5115e94c"
export WANDB_CONFIG_DIR="/home/user1/.config/wandb_arash"
export WANDB_CACHE_DIR="/home/user1/.cache/wandb_arash"

export PYTHONPATH=/home/user1/arashwork/testing_oft/openvla-oft

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUS=8

# Run the training command
torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune.py \
  --vla_path "moxin-hf-convert" \
  --data_root_dir modified_libero_rlds_data \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir logs/libero_finetune_logs_checkpoints/moxin-7b-vlm-libero-spatial_no_film_proprio_2_img_3e-4 \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 3e-4 \
  --use_film False \
  --use_proprio True \
  --num_images_in_input 2 \
  --resume False \
  --image_aug True \
  --save_latest_checkpoint_only False \
  --wandb_project OpenVLA-OFT-Moxin_7B_VLM_finetune_LIBERO_SPATIAL_8H100 \
  --wandb_entity "arash-akbari-stu-northeastern-university" \
  --save_freq 10000 \
  --max_steps 50000 > output_libero_spatial_2_img_3e-4.txt 2>&1
    # 1>"${STDOUT_LOG}" \
    # 2>"${STDERR_LOG}"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    echo "Check the error log at: ${STDERR_LOG}"
fi 