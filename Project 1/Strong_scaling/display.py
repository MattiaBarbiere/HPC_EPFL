import os
import re
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))

# List of processor counts
processor_counts = [1, 2, 4, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]
alpha = 0.892760

# Data will be organized as: {n: {p: time}}
data = {}

for p in processor_counts:
    filename = f'{path}/SS_{p}.txt'
    if not os.path.isfile(filename):
        print("here")
        continue

    with open(filename, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines)-1, 3):
        matrix_line = lines[i]
        time_line = lines[i+1]

        match_n = re.search(r'n(\d+)', matrix_line)
        match_time = re.search(r'Time\s*=\s*([0-9.]+)', time_line)

        if match_n and match_time:
            n = int(match_n.group(1))
            time = float(match_time.group(1))

            if n not in data:
                data[n] = {}
            data[n][p] = time

# Plotting: for each fixed problem size n, plot p vs time
fig, ax = plt.subplots(figsize=(10, 6))

values_n_to_avoid = [1000, 2000]

for n in sorted(data):
    if n in values_n_to_avoid:
        continue
    p_vals = sorted(data[n])
    t_vals = [data[n][p] for p in p_vals]
    plt.plot(p_vals, t_vals, marker='o', label=f'n = {n}')

plt.xlabel('Number of Processors (p)')
plt.ylabel('Time (s)')
plt.title('Execution Time vs Number of Processors')
plt.legend(title='Problem Size n')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save image under images
fig.savefig('Project 1/images/execution_time_vs_processors.png')
