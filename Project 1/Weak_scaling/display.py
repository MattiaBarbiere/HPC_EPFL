import os
import re
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))

# List of processor counts
processor_counts = [1, 2, 4, 8, 10, 15, 20, 30, 40, 50, 75, 100]

# Data will be stored as a dictionary {p: time}
data_p = {}

# Alphas will be stored as a dictionary {p: alpha}
alphas = {}

# Values of the single processor time {p: time_single_processor} for matrix weak_matrix_p.mtx
data_1 = {}

# Fill up the alphas and data_1 dict
input_path = os.path.join(path, 'tot_timings/tot_time.txt')
with open(input_path, 'r') as f:
    lines = f.readlines()

current_matrix_num = None
time_val = None
alpha_val = None

for line in lines:
    # Extract matrix number from the filename
    match_matrix = re.search(r'weak_matrix_(\d+)\.mtx', line)
    if match_matrix:
        current_matrix_num = int(match_matrix.group(1))

    # Extract Time and alpha values
    match_time = re.search(r'Time\s*=\s*([\d.eE+-]+)', line)
    match_alpha = re.search(r'alpha:\s*([\d.eE+-]+)', line)

    if match_time and match_alpha and current_matrix_num is not None:
        time_val = float(match_time.group(1))
        alpha_val = float(match_alpha.group(1))
        alphas[current_matrix_num] = alpha_val
        data_1[current_matrix_num] = time_val


for p in processor_counts:
    filename = f'{path}/WS_{p}.txt'
    if not os.path.isfile(filename):
        raise ValueError(f"File {filename} does not exist.")
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract matrix number from first line
    matrix_match = re.search(r'weak_matrix_(\d+)\.mtx', lines[0])
    matrix_num = int(matrix_match.group(1)) if matrix_match else None

    # Extract Time from second line
    time_match = re.search(r'Time\s*=\s*([\d.]+)', lines[1])
    time_val = float(time_match.group(1)) if time_match else None

    if matrix_num is not None and time_val is not None:
        data_p[matrix_num] = time_val

assert data_p.keys() == data_1.keys(), f"Mismatch in matrix numbers between data_p and data_1{data_p.keys()}{data_1.keys()}"
assert alphas.keys() == data_1.keys(), f"Mismatch in matrix numbers between alphas and data_1{alphas.keys()}{data_1.keys()}"
assert list(alphas.keys()) == processor_counts, f"Mismatch in matrix numbers between alphas and data_p{alphas.keys()}{data_p.keys()}"

# Compute speed up and efficiency
speedup = {}
efficiency = {}
ideal_speedup = {}
ideal_efficiency = {}

for p in data_1.keys():
    ideal_speedup[p] = 1 + (p - 1) * alphas[p]
    ideal_efficiency[p] = ideal_speedup[p] / p
    speedup[p] = data_1[p] / data_p[p]
    efficiency[p] = speedup[p] / p

# Plot everything
fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)

ax[0].plot(processor_counts, efficiency.values(), marker='o', label='Empirical Efficiency')
ax[0].plot(processor_counts, ideal_efficiency.values(), marker='o', label='Ideal Efficiency', linestyle='--', color="red")

ax[1].plot(processor_counts, speedup.values(), marker='o', label='Empirical Speedup')
ax[1].plot(processor_counts[:5], list(ideal_speedup.values())[:5], marker='o', label='Ideal Speedup', linestyle='--', color="red")


ax[0].set_title('Weak scaling')
ax[0].set_ylabel('Efficiency')
ax[0].set_title('Efficiency vs Number of Processors')
ax[0].legend(title='Problem Size N')
ax[0].grid(True)

ax[1].set_ylabel('Speedup')
ax[1].set_title('Speedup vs Number of Processors')
ax[1].legend(title='Problem Size N')
ax[1].grid(True)
ax[1].set_yticks([1,2, 4, 6, 8])
ax[1].set_xlabel('Number of Processors (p)')
plt.tight_layout()
plt.show()

# Save image under images
fig.savefig('Project 1/images/weak_scaling.png', dpi=fig.dpi, bbox_inches='tight')