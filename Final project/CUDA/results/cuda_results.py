import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
#Setting theme
sns.set_theme(style="whitegrid")

# Directory containing your log files
path = os.path.dirname(os.path.abspath(__file__))

# Lists to store extracted data
elapsed_times = []
threads_per_block = []

# Regex patterns
time_pattern = re.compile(r"Elapsed time:\s*([\d.]+)\s*seconds")
threads_pattern = re.compile(r"Number of threads per block:\s*(\d+)")

# Iterate over each file in the directory
for filename in os.listdir(path):
    print(f"Processing file: {filename}")
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), 'r') as file:
            content = file.read()
            
            time_match = time_pattern.search(content)
            threads_match = threads_pattern.search(content)
            
            if time_match and threads_match:
                elapsed_times.append(float(time_match.group(1)))
                threads_per_block.append(int(threads_match.group(1)))

# Plotting
plt.figure(figsize=(10, 6))
threads_per_block, elapsed_times = zip(*sorted(zip(threads_per_block, elapsed_times)))
plt.plot(range(len(elapsed_times)), elapsed_times, marker='o', linestyle='-')
plt.title('Elapsed Time vs. Threads per block')
plt.xlabel('Threads per Block (log scale)')
plt.ylabel('Elapsed time (seconds)')
plt.xticks(range(len(threads_per_block)), threads_per_block)
plt.grid(True)

# Save the plot
saving_path = os.path.join(os.path.dirname(os.path.dirname(path)), "scaling_images")
plt.savefig(os.path.join(saving_path, 'cuda_scaling.png'))
plt.show()
