import os
import re
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
#Setting theme
sns.set_theme(style="whitegrid")

# Directory containing your log files
path = os.path.dirname(os.path.abspath(__file__))

# Alpha for Gustafson's Law
# Note: In theory one should estimate ALPHA for every problem size, 
# as a simplying assuption we use a constant value for all grid sizes.
# ALPHA was estimated with 8 processors with a grid size of 32x32 per processor.
ALPHA = 0.9651

# Lists to store extracted data as dicts
results = []

# Regex patterns
time_pattern = re.compile(r"Elapsed time:\s*([\d.]+)\s*seconds")
processor_pattern = re.compile(r"Number of processors used:\s*(\d+)")
grid_pattern = re.compile(r"Running test case \d+ with grid (\d+)x\d+")

# Iterate over each file in the directory
for filename in os.listdir(path):
    print(f"Processing file: {filename}")
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), 'r') as file:
            content = file.read()
            
            time_match = time_pattern.search(content)
            processor_match = processor_pattern.search(content)
            grid_match = grid_pattern.search(content)
            
            if time_match and processor_match and grid_match:

                
                results.append({
                    "elapsed_time": float(time_match.group(1)),
                    "n_processors": int(processor_match.group(1)),
                    "grid_size": int(grid_match.group(1))//int(sqrt(int(processor_match.group(1))))
                })

# Make sure that for each processor there is three grid sizes
for vals in results:
    assert vals["grid_size"] in [64, 128, 256], "Grid size must be one of [64, 128, 256]"




# Get unique grid sizes
unique_grid_sizes = sorted(set([r["grid_size"] for r in results]))
unique_n_processors = sorted(set([r["n_processors"] for r in results]))


# Dicts to keep track of efficiency values: grid_size: [values]
speedups = {}
efficiencies = {}

for idx, grid in enumerate(unique_grid_sizes):
    # Filter results for this grid size
    filtered = [r for r in results if r["grid_size"] == grid]
    filtered.sort(key=lambda x: x["n_processors"])
    n_procs = [r["n_processors"] for r in filtered]
    times = [r["elapsed_time"] for r in filtered]

    # Calculate speedup and efficiency
    speedup = [times[0] / t for t in times]
    efficiency = [s / p for s, p in zip(speedup, n_procs)]

    # Add to dictionaries
    speedups[grid] = speedup
    efficiencies[grid] = efficiency


# Prepare for plotting
colors = sns.color_palette("tab10", n_colors=len(unique_grid_sizes))
GLOBAL_FONT_SIZE = 17
plt.rcParams.update({
    'font.size': GLOBAL_FONT_SIZE,
    'axes.titlesize': GLOBAL_FONT_SIZE,
    'axes.labelsize': GLOBAL_FONT_SIZE,
    'xtick.labelsize': GLOBAL_FONT_SIZE,
    'ytick.labelsize': GLOBAL_FONT_SIZE,
    'legend.fontsize': GLOBAL_FONT_SIZE,
    'legend.title_fontsize': GLOBAL_FONT_SIZE
})

# Plotting speedups
plt.figure(figsize=(10, 6))
for idx, grid in enumerate(unique_grid_sizes):
    times = speedups[grid]
    plt.plot(sorted(unique_n_processors), times, marker='o', linestyle='-', color=colors[idx], label=f'{grid}x{grid}')

# Ideal speedup line based on Gustafson's Law
ideal_speedup = [1 + (p - 1) * ALPHA for p in sorted(unique_n_processors)]
plt.plot(sorted(unique_n_processors), ideal_speedup, linestyle='--', color='black', label='Ideal (Gustafson\'s Law)')
plt.title('Speedup vs Number of Processors')
plt.xlabel('Number of processors used (log scale)')
plt.ylabel('Speedup (log scale)')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.xticks(sorted(unique_n_processors), labels=[str(p) for p in sorted(unique_n_processors)])
plt.legend(title="Grid Size per processor")

# Shift the "100" label to the left if present
ax = plt.gca()
xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()
for i, label in enumerate(xticklabels):
    if label.get_text() == "100":
        # Hide the original label
        label.set_visible(False)
        # Annotate a new label slightly to the left
        ax.annotate(
            "100",
            xy=(xticks[i], 0), xycoords=('data', 'axes fraction'),
            xytext=(20, -10), textcoords='offset points',
            ha='right', va='top', fontsize=GLOBAL_FONT_SIZE
        )
        break

# Save the plot
saving_path = os.path.join(os.path.dirname(os.path.dirname(path)), "scaling_images")
plt.savefig(os.path.join(saving_path, 'speedup_weak_scaling.png'))
plt.show()



# Plotting efficencies
plt.figure(figsize=(10, 6))
for idx, grid in enumerate(unique_grid_sizes):
    times = efficiencies[grid]
    plt.plot(sorted(unique_n_processors), times, marker='o', linestyle='-', color=colors[idx], label=f'{grid}x{grid}')

# Ideal efficiency line based on Gustafson's Law
ideal_eff = [ideal_speedup[i]/p for i, p in enumerate(sorted(unique_n_processors))]
plt.plot(sorted(unique_n_processors), ideal_eff, linestyle='--', color='black', label='Ideal (Gustafson\'s Law)')
plt.title('Efficiency vs Number of Processors')
plt.xlabel('Number of processors used (log scale)')
plt.ylabel('Efficiency (log scale)')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.xticks(sorted(unique_n_processors), labels=[str(p) for p in sorted(unique_n_processors)])
plt.legend(title="Grid Size per processor")

# Shift the "100" label to the left if present
ax = plt.gca()
xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()
for i, label in enumerate(xticklabels):
    if label.get_text() == "100":
        # Hide the original label
        label.set_visible(False)
        # Annotate a new label slightly to the left
        ax.annotate(
            "100",
            xy=(xticks[i], 0), xycoords=('data', 'axes fraction'),
            xytext=(20, -10), textcoords='offset points',
            ha='right', va='top', fontsize=GLOBAL_FONT_SIZE
        )
        break

# Save the plot
saving_path = os.path.join(os.path.dirname(os.path.dirname(path)), "scaling_images")
plt.savefig(os.path.join(saving_path, 'efficiency_weak_scaling.png'))
plt.show()
