#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64000M
#SBATCH --array=0-15
#SBATCH --job-name=kairos_taqa
#SBATCH --output=slurm_logs/%A/stdout_%a.out
#SBATCH --error=slurm_logs/%A/stderr_%a.out


MODELS=(
    "kyutai/Sequential_Helium_6B,"
    "kyutai/Sequential_Helium_6B,sequential_2024"
    "kyutai/Sequential_Helium_6B,sequential_2023"
    "kyutai/Sequential_Helium_6B,sequential_2022"
    "kyutai/Sequential_Helium_6B,sequential_2021"
    "kyutai/Sequential_Helium_6B,sequential_2020"
    "kyutai/Sequential_Helium_6B,shuffle_eq_2020"
    "kyutai/Sequential_Helium_6B,shuffle_eq_2024"
    "kyutai/Sequential_Helium_6B,shuffle_eq_2025"
    "google/gemma-3-4b-pt,"
    "google/gemma-3-12b-pt,"
    "meta-llama/Llama-3.1-8B,"
    "Qwen/Qwen3-14B,"
    "Qwen/Qwen3-4B,"
    "Qwen/Qwen3-8B,"
    "allenai/Olmo-3-1025-7B,"
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Split by comma into two variables
IFS=',' read -r MODEL_NAME SUBFOLDER <<< "$MODEL"

echo "Model: $MODEL_NAME"
echo "Folder: $SUBFOLDER"

MODEL_EXP_NAME="$(basename "$MODEL_NAME")"

uv run python kairos/evaluate.py \
    --model $MODEL_NAME \
    --subfolder "${SUBFOLDER}" \
    -k 5 \
    --tasks taqa \
    --insensitive "true,false" \
    --time-strat ",2018,2019,2020,2021,2022,2023" \
    --run-name taqa_${MODEL_EXP_NAME}
    # --verbose-log \  # Add if you want to save each eval for investigation purposes
