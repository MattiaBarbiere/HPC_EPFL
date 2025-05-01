import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Store results
threads_list = []
time_per_iter_list = []

# Read all relevant files
for filename in os.listdir('.'):
    if filename.startswith("times_") and filename.endswith(".txt"):
        match = re.search(r'times_(\d+)\.txt', filename)
        if not match:
            continue
        threads = int(match.group(1))

        with open(filename, 'r') as f:
            content = f.read()

            # Extract time
            time_match = re.search(r'Time for CG .*?=\s*([\d.eE+-]+)', content)
            time = float(time_match.group(1)) if time_match else None

            # Extract step number
            step_match = re.search(r'\[STEP\s+(\d+)\]', content)
            steps = int(step_match.group(1)) if step_match else None

            if time is not None and steps is not None:
                threads_list.append(threads)
                time_per_iter_list.append(time / steps)

# Sort by thread count for proper plotting
sorted_data = sorted(zip(threads_list, time_per_iter_list))
threads_list, time_per_iter_list = zip(*sorted_data)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(threads_list, time_per_iter_list, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Threads')
plt.ylabel('Time per iteration (s)')
plt.title('CG Solver: Time per Iteration vs Threads')
plt.grid(True, which='both', linestyle='--')
plt.savefig('cg_time_per_iteration.png')
plt.show()
