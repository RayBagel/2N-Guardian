#!/usr/bin/env bash
set -euo pipefail

# 2N-Guardian GRPO launch script (Qwen3.5-2B + LoRA)
# Usage:
#   export SILICONFLOW_API_KEY=xxxx   # if your reward path needs API later
#   bash v2_mvp/rl/ms_swift/run_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ms_swift is v2_mvp/rl/ms_swift -> project root is two levels up (v2_mvp)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# workspace root is parent of v2_mvp (contains src/ and v2_mvp/)
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="$PROJECT_ROOT/rl/ms_swift/grpo_config.yaml"
DATASET_INFO="$PROJECT_ROOT/rl/ms_swift/dataset_info.json"
REWARD_PATH="$PROJECT_ROOT/rl/reward_manager.py"
REWARD_ADAPTER_PATH="$PROJECT_ROOT/rl/ms_swift/reward_swift_adapter.py"

# Ensure src/ is discoverable when v2_mvp and src are siblings under a common root.
export PYTHONPATH="$WORKSPACE_ROOT:${PYTHONPATH:-}"

if ! command -v swift >/dev/null 2>&1; then
  echo "[ERROR] swift command not found. Install ms-swift first: pip install ms-swift[llm] -U"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] missing config: $CONFIG_PATH"
  exit 1
fi
if [ ! -f "$DATASET_INFO" ]; then
  echo "[ERROR] missing dataset_info: $DATASET_INFO"
  exit 1
fi
if [ ! -f "$REWARD_PATH" ]; then
  echo "[ERROR] missing reward manager backend: $REWARD_PATH"
  exit 1
fi
if [ ! -f "$REWARD_ADAPTER_PATH" ]; then
  echo "[ERROR] missing reward adapter: $REWARD_ADAPTER_PATH"
  exit 1
fi

if command -v realpath >/dev/null 2>&1; then
  echo "[INFO] dataset_info path: $(realpath "$DATASET_INFO")"
else
  echo "[INFO] dataset_info path: $DATASET_INFO"
fi
echo "[HINT] dataset_info.json is used with --custom_dataset_info (JSON), not --custom_register_path (.py)."

HELP_TEXT="$(swift rlhf -h 2>/dev/null || true)"
REWARD_ARG_NAME=""
if echo "$HELP_TEXT" | grep -q -- "--reward_funcs"; then
  REWARD_ARG_NAME="--reward_funcs"
elif echo "$HELP_TEXT" | grep -q -- "--reward_func"; then
  REWARD_ARG_NAME="--reward_func"
elif echo "$HELP_TEXT" | grep -q -- "--custom_reward_path"; then
  REWARD_ARG_NAME="--custom_reward_path"
elif echo "$HELP_TEXT" | grep -q -- "--reward_path"; then
  REWARD_ARG_NAME="--reward_path"
fi

if [ -z "$REWARD_ARG_NAME" ]; then
  echo "[ERROR] Could not detect reward argument from 'swift rlhf -h'."
  echo "[HINT] Please inspect help output manually and set reward arg in this script."
  exit 1
fi

echo "[INFO] Detected reward argument: $REWARD_ARG_NAME"

DATASET_ARG_NAME=""
if echo "$HELP_TEXT" | grep -q -- "--custom_dataset_info"; then
  DATASET_ARG_NAME="--custom_dataset_info"
elif echo "$HELP_TEXT" | grep -q -- "--dataset_info"; then
  DATASET_ARG_NAME="--dataset_info"
elif echo "$HELP_TEXT" | grep -q -- "--custom_register_path"; then
  DATASET_ARG_NAME="--custom_register_path"
fi

if [ -z "$DATASET_ARG_NAME" ]; then
  echo "[ERROR] Could not detect dataset info argument from 'swift rlhf -h'."
  echo "[HINT] Please inspect help output manually and set dataset arg in this script."
  exit 1
fi

echo "[INFO] Detected dataset argument: $DATASET_ARG_NAME"

# Optional args: quantization / KL / temperature / gradient checkpointing
KL_VALUE="${KL_VALUE:-0.02}"
TEMP_VALUE="${TEMP_VALUE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

QUANT_ARG=""
if echo "$HELP_TEXT" | grep -q -- "--quantization_bit"; then
  QUANT_ARG="--quantization_bit 4"
elif echo "$HELP_TEXT" | grep -q -- "--quantization_bits"; then
  QUANT_ARG="--quantization_bits 4"
elif echo "$HELP_TEXT" | grep -q -- "--load_in_4bit"; then
  QUANT_ARG="--load_in_4bit true"
fi

GC_ARG=""
if echo "$HELP_TEXT" | grep -q -- "--gradient_checkpointing"; then
  GC_ARG="--gradient_checkpointing true"
elif echo "$HELP_TEXT" | grep -q -- "--grad_checkpoint"; then
  GC_ARG="--grad_checkpoint true"
fi

KL_ARG=""
if echo "$HELP_TEXT" | grep -q -- "--kl_coef"; then
  KL_ARG="--kl_coef $KL_VALUE"
elif echo "$HELP_TEXT" | grep -q -- "--kl_beta"; then
  KL_ARG="--kl_beta $KL_VALUE"
elif echo "$HELP_TEXT" | grep -q -- "--kl_coeff"; then
  KL_ARG="--kl_coeff $KL_VALUE"
fi

TEMP_ARG=""
if echo "$HELP_TEXT" | grep -q -- "--temperature"; then
  TEMP_ARG="--temperature $TEMP_VALUE"
elif echo "$HELP_TEXT" | grep -q -- "--sampling_temperature"; then
  TEMP_ARG="--sampling_temperature $TEMP_VALUE"
elif echo "$HELP_TEXT" | grep -q -- "--gen_temperature"; then
  TEMP_ARG="--gen_temperature $TEMP_VALUE"
fi

echo "[INFO] Quant arg: ${QUANT_ARG:-<none>}"
echo "[INFO] Grad checkpoint arg: ${GC_ARG:-<none>}"
echo "[INFO] KL arg: ${KL_ARG:-<none>}"
echo "[INFO] Temp arg: ${TEMP_ARG:-<none>}"

# NOTE:
# Keep --model consistent with grpo_config.yaml
# If your swift version does not support --reward_funcs path directly,
# adapt to the corresponding custom reward registration argument.
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen3.5-2B \
  --dataset 2n-guardian-grpo \
  "$DATASET_ARG_NAME" "$DATASET_INFO" \
  "$REWARD_ARG_NAME" "$REWARD_ADAPTER_PATH" \
  $QUANT_ARG \
  $GC_ARG \
  $KL_ARG \
  $TEMP_ARG \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-7 \
  --num_train_epochs 1 \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --warmup_ratio 0.03 \
  --logging_steps 1 \
  --save_steps 50 \
  --eval_steps 50 \
  --output_dir "$PROJECT_ROOT/rl/output/qwen3_5_2b_grpo_lora"
