import re
import os

path = os.path.dirname(os.path.abspath(__file__))
file = 'tot_time.txt'

input_path = os.path.join(path, file)

# Patterns
mat_vec_pattern = re.compile(r'mat_vec\s*time\s*=\s*([\d.e+-]+)')
avg_pattern = re.compile(r'p\s*=\s*1\s*Time\s*=\s*([\d.e+-]+)')

with open(input_path, 'r') as f:
    lines = f.readlines()

updated_lines = []
mat_vec_time = None

for line in lines:
    # Check for mat-vec time
    match_matvec = re.search(mat_vec_pattern, line)
    if match_matvec:
        mat_vec_time = float(match_matvec.group(1))

    # Check for Time line
    match_time = re.search(avg_pattern, line)
    if match_time and mat_vec_time is not None:
        total_time = float(match_time.group(1))
        alpha = mat_vec_time / total_time if total_time != 0 else 0
        line = line.rstrip() + f" | alpha: {alpha:.6f}\n"
        mat_vec_time = None
        print(alpha)
    updated_lines.append(line)

# Add alpha to the end of the line
with open(input_path, 'w') as f:
    f.writelines(updated_lines)