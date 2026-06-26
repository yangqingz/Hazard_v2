#!/bin/bash
# A bash script to run multiple HAZARD submit jobs

# Fail fast if something goes wrong
set -e

# List of experiments
# Each line: output_dir env_name agent port max_test_episode lm_id
EXPERIMENTS=(
  "outputs/fire/gemini-3.5-flash_with_inference fire llm 1071 25 gemini-3.5-flash google True"
  "outputs/flood/gemini-3.5-flash_with_inference flood llm 1071 25 gemini-3.5-flash google True"
  "outputs/fire/gemini-3.5-flash fire llm 1071 25 gemini-3.5-flash google False"
  "outputs/flood/gemini-3.5-flash flood llm 1071 25 gemini-3.5-flash google False"
  # "outputs/flood/fine_tune_gpt3.5 flood llm 1071 25 ft:gpt-3.5-turbo-0125:usyd::CEyvkoyQ openai"
  # "outputs/fire/rrara_plan_with_img_1 fire llmv4 1071 25 ft:gpt-3.5-turbo-0125:usyd::CGcHaZKl openai"
  # "outputs/fire/plan_with_inference_img  fire llmv4 1071 25 gpt-5 openai"
  # "outputs/flood/rrara_test_rule flood rule 1071 25 ft:gpt-3.5-turbo-0125:usyd::CEyvkoyQ openai"
  # "outputs/fire/gpt-4__with_inference_20260624 fire llm 1071 25 gpt-4 openai"
  # "outputs/wind/random_with_inference wind random 1071 25 claude-sonnet-3-7 anthropic"
  # "outputs/wind/llama2-7b-hf_with_inference wind llm 1071 25 llama2-7b-hf huggingface"
  # "outputs/fire/plan_fine_tune2_with_inference fire llmv4 1072 25 ft:gpt-3.5-turbo-0125:usyd::CDiV1ta4"
  #"outputs/fire/plan_fine_tune3_with_inference fire llmv4 1071 25 ft:gpt-3.5-turbo-0125:usyd::CEyvkoyQ"

)
for exp in "${EXPERIMENTS[@]}"; do
  set -- $exp
  OUT=$1
  ENV=$2
  AGENT=$3
  PORT=$4
  EPISODES=$5
  LM=$6
  LS=$7
  INFERENCE=$8

  echo "Running $ENV with output=$OUT on port=$PORT with inference=$INFERENCE"
  python starter_code.py \
    --output_dir "$OUT" \
    --env_name "$ENV" \
    --agent "$AGENT" \
    --port "$PORT" \
    --max_test_episode "$EPISODES" \
    --lm_id "$LM" \
    --api_key_file api_key.txt \
    --lm_source "$LS" \
    --inference "$INFERENCE"
done
