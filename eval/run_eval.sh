#!/bin/bash
cd tufindtu/eval
export PATH=/horizon-bucket/saturn_v_dev/di.feng/env/traffic_env/bin:$PATH
# Default values
MODEL_DIR="/horizon-bucket/saturn_v_dev/di.feng/repo/tufindtu/output_sft"
# MODEL_DIR="/horizon-bucket/saturn_v_dev/di.feng/repo/Qwen2-VL-2B-Instruct"
BASE_MODEL_DIR="/horizon-bucket/saturn_v_dev/di.feng/repo/Qwen2-VL-2B-Instruct"
EVAL_DATA_PATH="./processed_data_otherprompts.json"
IMAGE_FOLDER="/horizon-bucket/saturn_v_dev/01_users/hao.gao/detection/bdd100k/bdd100k/"
# OUTPUT_PATH="/horizon-bucket/saturn_v_dev/di.feng/repo/tufindtu/eval_output_base/eval_predictions.json"
OUTPUT_PATH="./eval/new_prompt_results.json"
GPUS=5
BATCH_SIZE=1
MAX_NEW_TOKENS=4096

# Create output directory if it does not exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Execute the evaluation script
python -u eval.py \
  --model_dir "$MODEL_DIR" \
  --base_model_dir "$BASE_MODEL_DIR" \
  --eval_data_path "$EVAL_DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --output_path "$OUTPUT_PATH" \
  --gpus "$GPUS" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

