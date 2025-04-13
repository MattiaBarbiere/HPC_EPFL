import re
import os
path = os.path.dirname(os.path.abspath(__file__))

# Path to your file
file_path = f"{path}/tot_time_p1.txt"

# Regex patterns
loading_pattern = re.compile(r'Average loading time:\s*([0-9.e+-]+)')
solving_pattern = re.compile(r'Average solving time:\s*([0-9.e+-]+)')

# Store times
loading_times = []
solving_times = []

# Read the file lines
with open(file_path, 'r') as f:
    lines = f.readlines()

# Extract times
for line in lines:
    load_match = loading_pattern.search(line)
    solve_match = solving_pattern.search(line)
    if load_match:
        loading_times.append(float(load_match.group(1)))
    if solve_match:
        solving_times.append(float(solve_match.group(1)))

# Compute averages
avg_loading = sum(loading_times) / len(loading_times) if loading_times else 0
avg_solving = sum(solving_times) / len(solving_times) if solving_times else 0

# Total time
total_time = avg_loading + avg_solving

# Format output to append
average_info = f" | Total time: {total_time:.6f} s"

# Append to first line
lines[0] = lines[0].rstrip('\n') + average_info + '\n'

# Write back to file
with open(file_path, 'w') as f:
    f.writelines(lines)