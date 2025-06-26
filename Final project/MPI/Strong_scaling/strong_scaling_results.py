"""
Strong Scaling Analysis for MPI Performance.

This script analyzes strong scaling performance from MPI log files and generates
speedup and efficiency plots based on Amdahl's Law.
"""

import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting theme
sns.set_theme(style="whitegrid")


### Configuration Parameters

# Directory containing log files
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Alpha parameter for Amdahl's Law
# This represents the fraction of the program that can be parallelized
AMDAHL_ALPHA = 0.9945

# Global font size for plots
PLOT_FONT_SIZE = 17


### Data Extraction

def extract_performance_data(directory_path):
    """
    Extract performance data from log files in the specified directory.
    
    Args:
        directory_path (str): Path to directory containing log files
        
    Returns:
        list: List of dictionaries containing performance metrics
    """
    results = []
    
    # Patterns for data extraction
    time_pattern = re.compile(r"Elapsed time:\s*([\d.]+)\s*seconds")
    processor_pattern = re.compile(r"Number of processors used:\s*(\d+)")
    grid_pattern = re.compile(r"Running test case \d+ with grid (\d+)x\d+")
    
    # Process each log file
    for filename in os.listdir(directory_path):
        if not filename.endswith(".txt"):
            continue
            
        print(f"Processing file: {filename}")
        
        with open(os.path.join(directory_path, filename), 'r') as file:
            file_content = file.read()
            
            # Extract data using patterns
            time_match = time_pattern.search(file_content)
            processor_match = processor_pattern.search(file_content)
            grid_match = grid_pattern.search(file_content)
            
            # Store results if all patterns matched
            if time_match and processor_match and grid_match:
                results.append({
                    "elapsed_time": float(time_match.group(1)),
                    "n_processors": int(processor_match.group(1)),
                    "grid_size": int(grid_match.group(1))
                })
    
    return results

def validate_data(results):
    """
    Validate extracted data to ensure consistency.
    
    Args:
        results (list): List of performance data dictionaries
    """
    if not results:
        raise ValueError("No performance data extracted from log files")
    
    # Extract unique values for validation
    processor_counts = [r["n_processors"] for r in results]
    elapsed_times = [r["elapsed_time"] for r in results]
    grid_sizes = [r["grid_size"] for r in results]
    
    # Ensure data consistency
    assert len(processor_counts) == len(elapsed_times) == len(grid_sizes), \
        "Data lists must have the same length"


### Performance Analysis

def calculate_performance_metrics(results):
    """
    Calculate speedup and efficiency metrics from performance data.
    
    Args:
        results (list): List of performance data dictionaries
        
    Returns:
        tuple: (speedups_dict, efficiencies_dict, unique_grid_sizes, unique_processors)
    """
    # Sort results by processor count for consistent processing
    results.sort(key=lambda x: x["n_processors"])
    
    # Get unique values for analysis
    unique_grid_sizes = sorted(set(r["grid_size"] for r in results))
    unique_processors = sorted(set(r["n_processors"] for r in results))
    
    speedups = {}
    efficiencies = {}
    
    # Calculate metrics for each grid size
    for grid_size in unique_grid_sizes:
        # Filter and sort results for current grid size
        filtered_results = [r for r in results if r["grid_size"] == grid_size]
        filtered_results.sort(key=lambda x: x["n_processors"])
        
        # Extract processor counts and times
        processor_counts = [r["n_processors"] for r in filtered_results]
        execution_times = [r["elapsed_time"] for r in filtered_results]
        
        # Calculate speedup (relative to single processor performance)
        baseline_time = execution_times[0]
        speedup_values = [baseline_time / time for time in execution_times]
        
        # Calculate efficiency (speedup normalized by processor count)
        efficiency_values = [speedup / processors 
                           for speedup, processors in zip(speedup_values, processor_counts)]
        
        # Store results
        speedups[grid_size] = speedup_values
        efficiencies[grid_size] = efficiency_values
    
    return speedups, efficiencies, unique_grid_sizes, unique_processors


### Plotting

def setup_plot_style():
    """Configure matplotlib plotting parameters."""
    plt.rcParams.update({
        'font.size': PLOT_FONT_SIZE,
        'axes.titlesize': PLOT_FONT_SIZE,
        'axes.labelsize': PLOT_FONT_SIZE,
        'xtick.labelsize': PLOT_FONT_SIZE,
        'ytick.labelsize': PLOT_FONT_SIZE,
        'legend.fontsize': PLOT_FONT_SIZE,
        'legend.title_fontsize': PLOT_FONT_SIZE
    })

def plot_speedup(speedups, unique_grid_sizes, unique_processors, save_path):
    """
    Create and save speedup plot.
    
    Args:
        speedups (dict): Speedup values by grid size
        unique_grid_sizes (list): List of grid sizes
        unique_processors (list): List of processor counts
        save_path (str): Directory to save the plot
    """
    colors = sns.color_palette("tab10", n_colors=len(unique_grid_sizes))
    
    plt.figure(figsize=(10, 6))
    
    # Plot speedup for each grid size
    for idx, grid_size in enumerate(unique_grid_sizes):
        speedup_values = speedups[grid_size]
        plt.plot(range(len(speedup_values)), speedup_values, 
                marker='o', linestyle='-', color=colors[idx], 
                label=f'{grid_size}x{grid_size}')
    
    # Plot ideal speedup based on Amdahl's Law
    ideal_speedup = [1 / (1 - AMDAHL_ALPHA + AMDAHL_ALPHA / p) for p in unique_processors]
    plt.plot(range(len(ideal_speedup)), ideal_speedup, 
            linestyle='--', color='black', label='Ideal (Amdahl\'s Law)')
    
    # Configure plot
    plt.title('Speedup vs Number of Processors')
    plt.xlabel('Number of processors (log scale)')
    plt.ylabel('Speedup (log scale)')
    plt.xticks(range(len(unique_processors)), unique_processors)
    plt.grid(True)
    plt.yscale('log')
    plt.legend(title="Grid Size")
    
    # Save and display
    plt.savefig(os.path.join(save_path, 'speedup_strong_scaling.png'))
    plt.show()

def plot_efficiency(efficiencies, unique_grid_sizes, unique_processors, save_path):
    """
    Create and save efficiency plot.
    
    Args:
        efficiencies (dict): Efficiency values by grid size
        unique_grid_sizes (list): List of grid sizes
        unique_processors (list): List of processor counts
        save_path (str): Directory to save the plot
    """
    colors = sns.color_palette("tab10", n_colors=len(unique_grid_sizes))
    
    plt.figure(figsize=(10, 6))
    
    # Plot efficiency for each grid size
    for idx, grid_size in enumerate(unique_grid_sizes):
        efficiency_values = efficiencies[grid_size]
        plt.plot(range(len(efficiency_values)), efficiency_values, 
                marker='o', linestyle='-', color=colors[idx], 
                label=f'{grid_size}x{grid_size}')
    
    # Plot ideal efficiency based on Amdahl's Law
    ideal_speedup = [1 / (1 - AMDAHL_ALPHA + AMDAHL_ALPHA / p) for p in unique_processors]
    ideal_efficiency = [speedup / p for speedup, p in zip(ideal_speedup, unique_processors)]
    plt.plot(range(len(ideal_efficiency)), ideal_efficiency, 
            linestyle='--', color='black', label='Ideal (Amdahl\'s Law)')
    
    # Configure plot
    plt.title('Efficiency vs Number of Processors')
    plt.xlabel('Number of processors (log scale)')
    plt.ylabel('Efficiency (log scale)')
    plt.xticks(range(len(unique_processors)), unique_processors)
    plt.grid(True)
    plt.yscale('log')
    plt.legend(title="Grid Size")
    
    # Save and display
    plt.savefig(os.path.join(save_path, 'efficiency_strong_scaling.png'))
    plt.show()


def main():
    """Main function to perform strong scaling plotting."""
    # Extract performance data from log files
    performance_data = extract_performance_data(SCRIPT_PATH)
    
    # Validate extracted data
    validate_data(performance_data)
    
    # Calculate performance metrics
    speedups, efficiencies, unique_grid_sizes, unique_processors = \
        calculate_performance_metrics(performance_data)
    
    # Setup plotting configuration
    setup_plot_style()
    
    # Determine save path for plots
    save_directory = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "scaling_images")
    
    # Generate and save plots
    plot_speedup(speedups, unique_grid_sizes, unique_processors, save_directory)
    plot_efficiency(efficiencies, unique_grid_sizes, unique_processors, save_directory)

if __name__ == "__main__":
    main()
