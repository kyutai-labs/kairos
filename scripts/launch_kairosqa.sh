#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64000M
#SBATCH --array=0-191
#SBATCH --job-name=kairos_kairosqa
#SBATCH --output=slurm_logs/%A/stdout_%a.out
#SBATCH --error=slurm_logs/%A/stderr_%a.out

# Array layout: 16 models x 12 years = 192 tasks.
# task_id = model_idx * 12 + year_idx
# To run model 0 across all years: sbatch --array=0-11
# To run model 5 only:             sbatch --array=60-71
# To run a single (model, year):   sbatch --array=$((MODEL*12+YEAR))

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

EVAL_YEARS=(2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025)
N_YEARS=${#EVAL_YEARS[@]}

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / N_YEARS))
YEAR_IDX=$((SLURM_ARRAY_TASK_ID % N_YEARS))

MODEL=${MODELS[$MODEL_IDX]}
EVAL_YEAR=${EVAL_YEARS[$YEAR_IDX]}

IFS=',' read -r MODEL_NAME SUBFOLDER <<< "$MODEL"

echo "Model: $MODEL_NAME"
echo "Folder: $SUBFOLDER"
echo "Year:  $EVAL_YEAR"
K_SHOT='5'

MODEL_EXP_NAME="$(basename "$MODEL_NAME")"
RUN_NAME="kairosqa_${MODEL_EXP_NAME}"

echo "Evaluating KairosQA multiple-choice ($EVAL_YEAR)"
uv run python kairos/evaluate.py \
    --model $MODEL_NAME \
    --subfolder "${SUBFOLDER}" \
    -k $K_SHOT \
    --tasks temporal_${EVAL_YEAR} \
    --run-name $RUN_NAME
    # --verbose-log \  # Add if you want to save each eval for investigation purposes

echo "Evaluating KairosQA cloze ($EVAL_YEAR)"
uv run python kairos/evaluate.py \
    --model $MODEL_NAME \
    --subfolder "${SUBFOLDER}" \
    -k $K_SHOT \
    --cloze \
    --tasks temporal_${EVAL_YEAR} \
    --run-name $RUN_NAME
    # --verbose-log \  # Add if you want to save each eval for investigation purposes

echo "Evaluating KairosQA generative ($EVAL_YEAR)"
uv run python kairos/evaluate.py \
    --model $MODEL_NAME \
    --subfolder "${SUBFOLDER}" \
    -k $K_SHOT \
    --generate-task \
    --tasks temporal_${EVAL_YEAR} \
    --run-name $RUN_NAME
    # --verbose-log \  # Add if you want to save each eval for investigation purposes
