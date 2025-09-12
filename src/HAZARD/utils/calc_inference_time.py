import os
from collections import defaultdict

def analyze_actions_file(filepath):
    # Dictionary to store total time and count for each action type
    action_stats = {}
    
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
                if 'time' in parts:

                    elapsed_time = float(parts[-1])
                    total_times += elapsed_time
                    total_steps += 1
        
            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error details: {e}")
                continue
    average_time = total_times / total_steps if total_steps > 0 else 0
    action_stats['total'] = {'total_time': total_times, 'count': total_steps}
    action_stats['average'] = {'avg_time': average_time}
    return action_stats

def analyze_directory(directory_path):
    # Combined stats across all files
    combined_stats = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
    file_count = 0
    
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
            
            
    
    # Print combined averages
    print(f"\nCombined averages across {file_count} files:")
    print("-" * 50)
    if file_count > 0:
        overall_avg_time = total_average_time / file_count
        print(f"{'Overall':15} {overall_avg_time:10.2f}s")

if __name__ == "__main__":
    # Use the correct directory path
    directory = os.path.abspath("/home/yangqingzheng/HAZARD/outputs/fire/plan_fine_tune2_with_inference")
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        print("Current working directory:", os.getcwd())
        exit(1)
        
    analyze_directory(directory)