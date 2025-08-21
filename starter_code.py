from HAZARD import submit

# Run on full test set
submit(output_dir="outputs/mctsv2", env_name="fire", agent="mctsv2", port=1071, max_test_episode=25, 
       run_on_test=True,
       lm_id = "gpt-4",
       api_key_file="api_key.txt")
       # perceptional: bool = False,  # turn on this for the perceptional version of HAZARD
       #  effect_on_agents: bool = False,  # turn on this to let hazard affect agents
       #  run_on_test: bool = False,  # turn off to run on test set
