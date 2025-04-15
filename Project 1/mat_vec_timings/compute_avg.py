import os
import re
import matplotlib.pyplot as plt

# get the path to this file
path = os.path.dirname(os.path.abspath(__file__))
processor_counts = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100]

def process_files():

    for count in processor_counts:
        filename = f"{path}/mat_vec_time_p{count}.txt"

        try:
            with open(filename, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue

        times = []
        for line in lines:
            if "time in mat_vec" in line:
                try:
                    time_str = line.strip().split(":")[-1].strip().replace(" s", "")
                    times.append(float(time_str))
                except ValueError:
                    continue

        if times and lines:
            avg_time = sum(times) / len(times)
            average_info = f"  |  Average over {len(times)}: {avg_time:.7f} s"

            # Remove newline from first line if it exists, then append average info
            lines[0] = lines[0].rstrip('\n') + average_info + '\n'

            with open(filename, "w") as file:
                file.writelines(lines)

            print(f"Processed {filename}: average added to first line.")
        else:
            print(f"No valid time entries found in {filename} or file is empty.")

def plot_times_vs_processes():
    processes = []
    times = []

    for p in processor_counts:
        file_name = f"{path}/mat_vec_time_p{p}.txt"
        print(file_name)
        try:
            with open(file_name, 'r') as f:
                first_line = f.readline()
                time_match = re.search(r'Average over \d+: ([\d.]+) s', first_line)
                if time_match:
                    avg_time = float(time_match.group(1))
                    processes.append(p)
                    times.append(avg_time)
        except FileNotFoundError:
            print(f"Warning: {file_name} not found. Skipping.")

    # Compute efficiency and speedup of mat_vec function
    speed_up = [times[0] / t for t in times]
    efficiency = [s / p for s, p in zip(speed_up, processes)]

    # Ideal speedup and efficiency
    ideal_efficiency = [1] * len(processes)
    
    # Plot
    plt.figure(figsize=(6, 5), dpi=100)
    plt.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], linestyle='--', label='Speedup = p', linewidth=2.5)
    plt.plot(processes, ideal_efficiency, linestyle='--', label='Efficiency = 1', linewidth=2.5)
    plt.plot(processes, speed_up, marker='o', label='Speedup')
    plt.plot(processes, efficiency, marker='o', label='Efficiency')
    plt.xlabel('Number of Processes')
    plt.title('Speedup and efficiency of mat_vec funciton vs Number of Processes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Project 1/images/mat_vec_speedup_efficiency.png")
    plt.show()

if __name__ == "__main__":
    plot_times_vs_processes()