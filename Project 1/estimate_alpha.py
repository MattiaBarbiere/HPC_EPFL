import re
import os

# Paths to your files
path = "./Project 1/"
file1_path = 'mat_vec_timings/mat_vec_time_p1.txt'
file1_path = os.path.join(path, file1_path)
file2_path = 'tot_timings/tot_time_p1.txt'
file2_path = os.path.join(path, file2_path)

# Read file1: extract count and average time
with open(file1_path, 'r') as f1:
    content1 = f1.read()
    match1 = re.search(r'Average over\s+(\d+):\s*([\d.]+)', content1)
    if match1:
        count = int(match1.group(1))
        avg_time = float(match1.group(2))
    else:
        count, avg_time = None, None

# Read file2: extract total time
with open(file2_path, 'r') as f2:
    content2 = f2.read()
    match2 = re.search(r'Total time:\s*([\d.]+)', content2)
    if match2:
        total_time = float(match2.group(1))
    else:
        total_time = None

# Output the results
print(f"Time in mat_vec: {count * avg_time:.6f} s")
print(f"Total time = {total_time} s")