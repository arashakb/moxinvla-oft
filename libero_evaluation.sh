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

export PRISMATIC_DATA_ROOT=/home/user1/aras_prism_training/OXE/
export PYTHONPATH=/home/user1/arash_prism_training/openvla-oft
export CUDA_VISIBLE_DEVICES="3"
NUM_GPUS=1

# Run the training command
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./logs/libero_finetune_logs_checkpoints/moxin-7b-oxe-250k-image_aug-libero-object_no_film_proprio_1_img_3e-4/moxin-7b-224px+mx-oxe-magic-soup+n1+b32+x7-250k--image_aug+libero_object_no_noops+b16+lr-0.0003+lora-r32--2_imgs--proprio--70000_chkpt/ \
  --task_suite_name libero_object \
  --num_images_in_input 2 \
  --use_film False \
  --use_proprio True \
  --center_crop True > eval_libero_object_moxin_oxe_250k_1img_3e-4_70kft.txt 2>&1
    # 1>"${STDOUT_LOG}" \
    # 2>"${STDERR_LOG}"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    echo "Check the error log at: ${STDERR_LOG}"
fi 