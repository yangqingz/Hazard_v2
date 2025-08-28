from HAZARD import submit

import argparse
from HAZARD import submit

def main():
    parser = argparse.ArgumentParser(description="Run HAZARD experiments")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to store results")
    parser.add_argument("--env_name", type=str, choices=["fire", "flood", "wind"], required=True,
                        help="Environment to run")
    parser.add_argument("--agent", type=str, default="llm",
                        help="Agent type (default: llm)")
    parser.add_argument("--port", type=int, default=1071,
                        help="Port to use (default: 1071)")
    parser.add_argument("--max_test_episode", type=int, default=25,
                        help="Max number of test episodes")
    parser.add_argument("--run_on_test", default=True)
    parser.add_argument("--lm_id", type=str, default="gpt-3.5-turbo",
                        help="Language model id")
    parser.add_argument("--api_key_file", type=str, default="api_key.txt",
                        help="Path to API key file")

    args = parser.parse_args()

    # Pass arguments into submit
    submit(output_dir=args.output_dir,
           env_name=args.env_name,
           agent=args.agent,
           port=args.port,
           max_test_episode=args.max_test_episode,
           run_on_test=True,
           lm_id=args.lm_id,
           api_key_file=args.api_key_file)
           # You could also pass args.perceptional and args.effect_on_agents if submit supports them.

if __name__ == "__main__":
    main()