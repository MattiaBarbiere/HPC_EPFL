import os
import re
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))

# List of processor counts
processor_counts = [1, 2, 4, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]
alpha = 0.750815

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

# Compute empirircal speedup and efficiency
speedup = {}
efficiency = {}
ideal_speedup = {}
ideal_efficiency = {}

for n in data:
    if 1 not in data[n]:
        continue
    serial_time = data[n][1]
    speedup[n] = {}
    efficiency[n] = {}
    ideal_speedup[n] = {}
    ideal_efficiency[n] = {}
    for p in data[n]:
        speedup[n][p] = serial_time / data[n][p]
        efficiency[n][p] = speedup[n][p] / p
        ideal_speedup[n][p] = 1/((1 - alpha) + alpha / p)
        ideal_efficiency[n][p] = ideal_speedup[n][p] / p



# Plotting: for each fixed problem size n, plot p vs time
fig, ax = plt.subplots(2,1, figsize=(10, 6), dpi=100)

# Note there is an outlier at N=2000 and p = 40 
values_n_to_avoid = [200, 300, 2000]

for n in sorted(speedup):
    if n in values_n_to_avoid:
        continue
    p_vals = sorted(speedup[n])

    # Plot the efficiency vs number of processors
    y_vals = [efficiency[n][p] for p in p_vals]
    ax[0].plot(p_vals, y_vals, marker='o', label=f'N = {n}')

    # Plot the speedup vs number of processors
    y_vals = [speedup[n][p] for p in p_vals]
    ax[1].plot(p_vals, y_vals, marker='o', label=f'N = {n}')

# Plot the ideal speedup vs number of processors only once
for n in sorted(speedup):
    if n in values_n_to_avoid:
        continue
    p_vals = sorted(speedup[n])
    y_vals_ideal = [ideal_speedup[n][p] for p in p_vals]
    ax[1].plot(p_vals, y_vals_ideal, marker='o', label=f'Ideal', linestyle='--')
    y_vals_ideal = [ideal_efficiency[n][p] for p in p_vals]
    ax[0].plot(p_vals, y_vals_ideal, marker='o', label=f'Ideal', linestyle='--')
    break



ax[0].set_ylabel('Efficiency')
ax[0].set_title('Efficiency vs Number of Processors')
ax[0].legend(title='Problem Size N')
ax[0].grid(True)

ax[1].set_ylabel('Speedup')
ax[1].set_title('Speedup vs Number of Processors')
ax[1].legend(title='Problem Size N')
ax[1].grid(True)
ax[1].set_xlabel('Number of Processors (p)')
plt.tight_layout()
plt.show()

# Save image under images
fig.savefig('Project 1/images/execution_time_vs_processors.png')
