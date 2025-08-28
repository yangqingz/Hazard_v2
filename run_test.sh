#!/bin/bash
# A bash script to run multiple HAZARD submit jobs

# Fail fast if something goes wrong
set -e

# List of experiments
# Each line: output_dir env_name agent port max_test_episode lm_id
EXPERIMENTS=(
  "outputs/wind/llm-gpt3.5_v1.11.16 wind llm 1071 25 gpt-3.5-turbo"
  "outputs/fire/llm-gpt4_prompt fire llm 1071 25 gpt-4"
  "outputs/flood/mcts_v1.11.16 flood mcts 1071 25 gpt-3.5-turbo"
  "outputs/flood/mctsv2_v1.11.16 flood mctsv2 1071 25 gpt-3.5-turbo"
  "outputs/flood/rule_v1.11.16 flood rule 1071 25 gpt-3.5-turbo"
  "outputs/flood/greedy_v1.11.16 flood greedy 1071 25 gpt-3.5-turbo"
  "outputs/wind/mcts_v1.11.16 wind mcts 1071 25 gpt-3.5-turbo"
)

for exp in "${EXPERIMENTS[@]}"; do
  set -- $exp
  OUT=$1
  ENV=$2
  AGENT=$3
  PORT=$4
  EPISODES=$5
  LM=$6

  echo "Running $ENV with output=$OUT on port=$PORT"
  python starter_code.py \
    --output_dir "$OUT" \
    --env_name "$ENV" \
    --agent "$AGENT" \
    --port "$PORT" \
    --max_test_episode "$EPISODES" \
    --lm_id "$LM" \
    --api_key_file api_key.txt
done
