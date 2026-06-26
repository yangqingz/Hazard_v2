import os
import json
from collections import defaultdict

def analyze_actions_file(filepath):
    # Dictionary to store total time and count for each action type
    action_stats = {}
    print(f"Analyzing file: {filepath}")
    with open(filepath, 'r') as f:
        total_times = 0
        total_steps = 0
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse the line
            try:
                parts = line.strip().split()
                # Find the action tuple string between 'action' and 'elapsed time' or end of line
                if 'elapsed' in parts and 'time' in parts:
                    elapsed_time = float(parts[-1])
                    total_times += elapsed_time
                    total_steps += 1
        
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error details: {e}")
                continue
        print(f"Total steps in file: {total_steps}, Total time: {total_times:.2f}s")
    average_time = total_times / total_steps if total_steps > 0 else 0
    action_stats['total'] = {'total_time': total_times, 'count': total_steps}
    action_stats['average'] = {'avg_time': average_time}
    return action_stats

def analyze_directory(directory_path):
    # Combined stats across all files
    combined_stats = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
    file_count = 0
    
    # Variables to calculate average steps per episode
    total_steps = 0
    total_episodes = 0
    
    # Variables for avg_steps from eval_result.json
    total_avg_steps = 0
    eval_file_count = 0
    
    # Ensure directory path is absolute
    directory_path = os.path.abspath(directory_path)
    print(f"Searching in directory: {directory_path}")
    
    total_average_time = 0.0
    # Find all matching actions.txt files
    for root, dirs, files in os.walk(directory_path):
        if 'actions.txt' in files and 'mm_kitchen_3a-' in root:
            file_count += 1
            filepath = os.path.join(root, 'actions.txt')
            
            if not os.path.exists(filepath):
                print(f"Warning: File does not exist: {filepath}")
                continue
            
            # Get stats for this file
            file_stats = analyze_actions_file(filepath)
            
            if not file_stats:
                print(f"Warning: No valid actions found in {filepath}")
                continue
            
            total_average_time += file_stats['average']['avg_time']
            
            # Add steps for this file
            total_steps += file_stats['total']['count']
            total_episodes += 1  # Each file represents one episode
        
        # Check for eval_result.json in the same directory
        eval_filepath = os.path.join(root, 'eval_result.json')
        if os.path.exists(eval_filepath):
            try:
                with open(eval_filepath, 'r') as eval_file:
                    eval_data = json.load(eval_file)
                    if 'avg_steps' in eval_data:
                        total_avg_steps += eval_data['avg_steps']
                        print(f"Found avg_steps: {eval_data['avg_steps']} in {eval_filepath}")
                        eval_file_count += 1
            except Exception as e:
                print(f"Error reading {eval_filepath}: {e}")
    
    # Print combined averages
    print(f"\nCombined averages across {file_count} action files:")
    print("-" * 50)
    if file_count > 0:
        overall_avg_time = total_average_time / file_count
        avg_steps_per_episode = total_steps / total_episodes if total_episodes > 0 else 0
        print(f"{'Overall Time':15} {overall_avg_time:10.2f}s")
        print(f"{'Avg Steps/Episode':15} {avg_steps_per_episode:10.2f}")
    
    # Print avg_steps from eval_result.json files
    print(f"\nCombined avg_steps across {eval_file_count} eval_result.json files:")
    print("-" * 50)
    if eval_file_count > 0:
        overall_avg_steps = total_avg_steps / eval_file_count
        print(f"{'Avg Steps':15} {overall_avg_steps:10.2f}")
    else:
        print("No eval_result.json files found.")

if __name__ == "__main__":
    # Use the correct directory path
    directory = os.path.abspath("/home/yangqingzheng/HAZARD/outputs/fire/fine_tune_gpt3.5_with_inference")
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        print("Current working directory:", os.getcwd())
        exit(1)
        
    analyze_directory(directory)