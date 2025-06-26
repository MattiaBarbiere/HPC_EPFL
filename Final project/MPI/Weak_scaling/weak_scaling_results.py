"""
Weak Scaling Analysis for MPI Performance.

This script analyzes weak scaling performance from MPI log files and generates
speedup and efficiency plots based on Gustafson's Law.
"""

import os
import re
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

# Configure plotting theme
sns.set_theme(style="whitegrid")


### Configuration Parameters

# Directory containing log files
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Alpha parameter for Gustafson's Law
# Note: In theory, alpha should be estimated for every problem size.
# As a simplifying assumption, we use a constant value for all grid sizes.
# This alpha was estimated with 8 processors and a 32x32 grid per processor.
GUSTAFSON_ALPHA = 0.9651

# Expected grid sizes per processor for validation
EXPECTED_GRID_SIZES = [64, 128, 256]

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
            lines = file.readlines()
            
            # Calculate number of iterations (total lines minus header and footer)
            num_iterations = len(lines) - 6
            
            # Extract data using patterns
            file_content = ''.join(lines)
            time_match = time_pattern.search(file_content)
            processor_match = processor_pattern.search(file_content)
            grid_match = grid_pattern.search(file_content)
            
            # Store results if all patterns matched
            if time_match and processor_match and grid_match:
                num_processors = int(processor_match.group(1))
                total_grid_size = int(grid_match.group(1))
                
                # Calculate grid size per processor
                grid_per_processor = total_grid_size // int(sqrt(num_processors))
                
                results.append({
                    "elapsed_time": float(time_match.group(1)),
                    "n_processors": num_processors,
                    "grid_size": grid_per_processor,
                    "n_iterations": num_iterations
                })
    
    return results

def validate_data(results):
    """
    Validate extracted data to ensure consistency.
    
    Args:
        results (list): List of performance data dictionaries
    """
    for result in results:
        assert result["grid_size"] in EXPECTED_GRID_SIZES, \
            f"Grid size {result['grid_size']} not in expected sizes {EXPECTED_GRID_SIZES}"


### Performance Analysis

def calculate_performance_metrics(results):
    """
    Calculate speedup and efficiency metrics from performance data.
    
    Args:
        results (list): List of performance data dictionaries
        
    Returns:
        tuple: (speedups_dict, efficiencies_dict, unique_grid_sizes, unique_processors)
    """
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
        
        # Extract processor counts and normalized times
        processor_counts = [r["n_processors"] for r in filtered_results]
        times_per_iteration = [r["elapsed_time"] / r["n_iterations"] for r in filtered_results]
        
        # Calculate speedup
        baseline_time = times_per_iteration[0]
        speedup_values = [baseline_time / time for time in times_per_iteration]
        
        # Calculate efficiency
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

def adjust_tick_labels(ax):
    """
    Adjust x-axis tick labels, specifically handling the '100' label positioning.
    
    Args:
        ax: Matplotlib axes object
    """
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    
    for i, label in enumerate(xticklabels):
        if label.get_text() == "100":
            # Hide original label and add repositioned one
            label.set_visible(False)
            ax.annotate(
                "100",
                xy=(xticks[i], 0), xycoords=('data', 'axes fraction'),
                xytext=(20, -10), textcoords='offset points',
                ha='right', va='top', fontsize=PLOT_FONT_SIZE
            )
            break

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
        plt.plot(unique_processors, speedup_values, 
                marker='o', linestyle='-', color=colors[idx], 
                label=f'{grid_size}x{grid_size}')
    
    # Plot ideal speedup based on Gustafson's Law
    ideal_speedup = [1 + (p - 1) * GUSTAFSON_ALPHA for p in unique_processors]
    plt.plot(unique_processors, ideal_speedup, 
            linestyle='--', color='black', label='Ideal (Gustafson\'s Law)')
    
    # Configure plot
    plt.title('Speedup vs Number of Processors')
    plt.xlabel('Number of processors used (log scale)')
    plt.ylabel('Speedup (log scale)')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(unique_processors, labels=[str(p) for p in unique_processors])
    plt.legend(title="Grid Size per processor")
    
    # Adjust tick labels
    adjust_tick_labels(plt.gca())
    
    # Save and display
    plt.savefig(os.path.join(save_path, 'speedup_weak_scaling.png'))
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
        plt.plot(unique_processors, efficiency_values, 
                marker='o', linestyle='-', color=colors[idx], 
                label=f'{grid_size}x{grid_size}')
    
    # Plot ideal efficiency based on Gustafson's Law
    ideal_speedup = [1 + (p - 1) * GUSTAFSON_ALPHA for p in unique_processors]
    ideal_efficiency = [speedup / p for speedup, p in zip(ideal_speedup, unique_processors)]
    plt.plot(unique_processors, ideal_efficiency, 
            linestyle='--', color='black', label='Ideal (Gustafson\'s Law)')
    
    # Configure plot
    plt.title('Efficiency vs Number of Processors')
    plt.xlabel('Number of processors used (log scale)')
    plt.ylabel('Efficiency (log scale)')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(unique_processors, labels=[str(p) for p in unique_processors])
    plt.legend(title="Grid Size per processor")
    
    # Adjust tick labels
    adjust_tick_labels(plt.gca())
    
    # Save and display
    plt.savefig(os.path.join(save_path, 'efficiency_weak_scaling.png'))
    plt.show()


def main():
    """Main function to perform weak scaling plotting."""
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
