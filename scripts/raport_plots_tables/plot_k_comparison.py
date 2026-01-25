#!/usr/bin/env python3
"""
Plotting script for K comparison results.
Creates two plots: one for size 50 and one for size 200.
Each plot shows K on x-axis with 3 lines (baseline, nsa, rlbsa).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_data_for_size(data, problem_name, size):
    """Extract data for a specific problem size."""
    results = data['results']
    K_values = []
    baseline_avgs = []
    nsa_avgs = []
    rlbsa_avgs = []
    
    # Get K values from config
    K_list = sorted(data['config']['experiments'][problem_name]['K'])
    
    for K in K_list:
        key = f"{problem_name}_{size}_K{K}"
        if key in results:
            K_values.append(K)
            
            # Extract averages for each method
            baseline_avg = results[key].get('baseline', {}).get('avg', None)
            nsa_avg = results[key].get('nsa', {}).get('avg', None)
            rlbsa_avg = results[key].get('rlbsa', {}).get('avg', None)
            
            baseline_avgs.append(baseline_avg)
            nsa_avgs.append(nsa_avg)
            rlbsa_avgs.append(rlbsa_avg)
    
    return K_values, baseline_avgs, nsa_avgs, rlbsa_avgs


def plot_size_comparison(data, problem_name, size, output_path):
    """Create a plot for a specific problem size."""
    K_values, baseline_avgs, nsa_avgs, rlbsa_avgs = extract_data_for_size(data, problem_name, size)
    
    # Filter out None values for each method separately
    K_baseline = [K for K, avg in zip(K_values, baseline_avgs) if avg is not None]
    baseline_plot = [-avg for avg in baseline_avgs if avg is not None]
    
    K_nsa = [K for K, avg in zip(K_values, nsa_avgs) if avg is not None]
    nsa_plot = [-avg for avg in nsa_avgs if avg is not None]
    
    K_rlbsa = [K for K, avg in zip(K_values, rlbsa_avgs) if avg is not None]
    rlbsa_plot = [-avg for avg in rlbsa_avgs if avg is not None]
    
    plt.figure(figsize=(10, 6))
    
    # Plot lines for each method
    if K_baseline:
        plt.plot(K_baseline, baseline_plot, marker='o', label='baseline', linewidth=2, markersize=8)
    if K_nsa:
        plt.plot(K_nsa, nsa_plot, marker='s', label='nsa', linewidth=2, markersize=8)
    if K_rlbsa:
        plt.plot(K_rlbsa, rlbsa_plot, marker='^', label='rlbsa', linewidth=2, markersize=8)
    
    plt.xlabel('K', fontsize=14)
    plt.ylabel('Average Quality', fontsize=14)
    plt.title(f'Knapsack Problem - Size {size}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "outputs" / "results" / "knap_final_k_comp.json"
    output_dir = project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {json_path}")
    data = load_results(json_path)
    
    # Get problem name from config
    problem_name = list(data['config']['experiments'].keys())[0]
    print(f"Problem: {problem_name}")
    
    # Create plots for size 50 and size 200
    plot_size_comparison(data, problem_name, 50, output_dir / "knap_k_comparison_size50.png")
    plot_size_comparison(data, problem_name, 200, output_dir / "knap_k_comparison_size200.png")
    
    print("Done!")


if __name__ == "__main__":
    main()
