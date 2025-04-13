import re
import os

# Path to your file
path = os.path.dirname(os.path.abspath(__file__))
file_path = f"{path}/tot_time.txt"  # Replace with your actual filename

# Initialize variables
mat_vec_time = None
avg_time = None

# Patterns
mat_vec_pattern = re.compile(r'mat_vec time = ([\d.e+-]+)')
avg_pattern = re.compile(r'p\s*=\s*1\s*Time:\s*([\d.e+-]+)')

# Read the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Extract data
for line in lines:
    if mat_vec_time is None:
        match = mat_vec_pattern.search(line)
        if match:
            mat_vec_time = float(match.group(1))
    if avg_time is None:
        match = avg_pattern.search(line)
        if match:
            avg_time = float(match.group(1))

# Compute total average and ratio
print("mat_vec_time:", mat_vec_time)
print("avg_time:", avg_time)
if mat_vec_time and avg_time:
    alpha = mat_vec_time / avg_time
    result_line = f"Alpha = {alpha:.6f}\n"

    # Append to file
    with open(file_path, 'a') as f:
        f.write(result_line)

    print("Appended to file:", result_line.strip())
else:
    print("Could not extract all required values.")