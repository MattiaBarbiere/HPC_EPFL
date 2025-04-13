import re
import os

# Path to your file
path = os.path.dirname(os.path.abspath(__file__))
file_path = f"{path}/tot_time_p1.txt"  # Replace with your actual filename

# Initialize variables
mat_vec_time = None
avg_loading_time = None
avg_solving_time = None

# Patterns
mat_vec_pattern = re.compile(r'mat_vec time = ([\d.e+-]+)')
avg_loading_pattern = re.compile(r'Average loading time:\s*([\d.e+-]+)')
avg_solving_pattern = re.compile(r'Average solving time:\s*([\d.e+-]+)')

# Read file and extract data
with open(file_path, 'r') as f:
    for line in f:
        if mat_vec_time is None:
            match = mat_vec_pattern.search(line)
            if match:
                mat_vec_time = float(match.group(1))
        if avg_loading_time is None:
            match = avg_loading_pattern.search(line)
            if match:
                avg_loading_time = float(match.group(1))
        if avg_solving_time is None:
            match = avg_solving_pattern.search(line)
            if match:
                avg_solving_time = float(match.group(1))

# Compute total average time
total_avg_time = None
if avg_loading_time is not None and avg_solving_time is not None:
    total_avg_time = avg_loading_time + avg_solving_time

# Output the results
print(f"mat_vec time: {mat_vec_time:.6f} s")
print(f"Total average time: {total_avg_time:.6f} s")