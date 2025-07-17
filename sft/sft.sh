# export PATH=/horizon-bucket/saturn_v_dev/di.feng/env/r1-v/bin:$PATH
# cd tudindtu/sft
# sleep 1000000
# accelerate launch train_sft.py \
#     --model_name_or_path "/horizon-bucket/saturn_v_dev/di.feng/repo/Qwen2-VL-2B-Instruct" \
#     --data_path "./processed_data.json" \
#     --image_folder "/horizon-bucket/saturn_v_dev/01_users/hao.gao/detection/bdd100k/bdd100k/" \
#     --output_dir "/horizon-bucket/saturn_v_dev/di.feng/repo/tufindtu/output" \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 2e-5 \
#     --save_strategy "epoch" \
#     --logging_steps 10 \
#     --bf16 True \
#     --deepspeed ds_config_zero2.json \
#     --use_lora False \
#     --report_to none
# FILE: run_torchrun.sh

export PATH=/horizon-bucket/saturn_v_dev/di.feng/env/traffic_env/bin:$PATH
cd tufindtu/sft

# Set the number of GPUs to use for training.
# Defaults to 8 if the environment variable is not set.
# You can run `export NPROC_PER_NODE=4` before this script to use 4 GPUs.
# NPROC_PER_NODE=${NPROC_PER_NODE:-8}
# sleep 1000000
# CUDA_VISIBLE_DEVICES=1,2,3 
# Use torchrun instead of accelerate launch
# --nproc_per_node specifies how many processes (GPUs) to use.
export NCCL_P2P_DISABLE=1   # 禁用点对点通信
export NCCL_IB_DISABLE=1    # 禁用 InfiniBand
torchrun --nproc_per_node=8 ./train_sft.py \
    --model_name_or_path "/horizon-bucket/saturn_v_dev/di.feng/repo/Qwen2-VL-2B-Instruct" \
    --data_path "./processed_data.json" \
    --image_folder "/horizon-bucket/saturn_v_dev/01_users/hao.gao/detection/bdd100k/bdd100k/" \
    --output_dir "/horizon-bucket/saturn_v_dev/di.feng/repo/tufindtu/output_sft/" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --save_strategy "epoch" \
    --logging_steps 10 \
    --bf16 True \
    --deepspeed ./ds_config_zero2.json \
    --use_lora False \
    --report_to none