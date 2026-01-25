#!/usr/bin/env python3
"""
Generate LaTeX table from rosenbrock_final_bad.json results.
Bolds best results and calculates percentages compared to best.
Shows size * K for each row.
"""

import json
import statistics
from pathlib import Path


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_data(data):
    """Extract data for all K values."""
    results = data['results']
    problem_name = list(data['config']['experiments'].keys())[0]
    problem_config = data['config']['experiments'][problem_name]
    problem_sizes = problem_config['problem_sizes']
    K_values = sorted(problem_config['K'])
    
    # For rosenbrock, we typically have size=2
    size = problem_sizes[0] if problem_sizes else 2
    
    table_data = []
    
    for K in K_values:
        key = f"{problem_name}_{size}_K{K}"
        if key in results:
            # Calculate median from results array (lower is better for rosenbrock)
            baseline_results = results[key].get('baseline', {}).get('results', [])
            nsa_results = results[key].get('nsa', {}).get('results', [])
            rlbsa_results = results[key].get('rlbsa', {}).get('results', [])
            
            # Calculate median
            baseline_median = statistics.median(baseline_results) if baseline_results else float('inf')
            nsa_median = statistics.median(nsa_results) if nsa_results else float('inf')
            rlbsa_median = statistics.median(rlbsa_results) if rlbsa_results else float('inf')
            
            # Find best (lowest value = best for rosenbrock)
            values = {
                'baseline': baseline_median,
                'nsa': nsa_median,
                'rlbsa': rlbsa_median
            }
            best_method = min(values, key=values.get)  # min value = best
            best_value = values[best_method]
            
            # Calculate percentages compared to best
            # Percentage shows how much WORSE each value is compared to best
            # Formula: ((value - best) / best) * 100
            baseline_pct = ((baseline_median - best_value) / best_value * 100) if best_value > 0 else 0
            nsa_pct = ((nsa_median - best_value) / best_value * 100) if best_value > 0 else 0
            rlbsa_pct = ((rlbsa_median - best_value) / best_value * 100) if best_value > 0 else 0
            
            # For the best method, percentage should be 0.00%
            if best_method == 'baseline':
                baseline_pct = 0.0
            elif best_method == 'nsa':
                nsa_pct = 0.0
            elif best_method == 'rlbsa':
                rlbsa_pct = 0.0
            
            table_data.append({
                'size': size,
                'K': K,
                'size_K': size * K,
                'baseline': {'value': baseline_median, 'pct': baseline_pct, 'is_best': best_method == 'baseline'},
                'nsa': {'value': nsa_median, 'pct': nsa_pct, 'is_best': best_method == 'nsa'},
                'rlbsa': {'value': rlbsa_median, 'pct': rlbsa_pct, 'is_best': best_method == 'rlbsa'},
            })
    
    return table_data


def format_value(value, pct, is_best):
    """Format a value with percentage, bolding if best."""
    value_str = f"{value:.4f}"
    pct_str = f"({pct:.2f}\%)"
    
    if is_best:
        return f"\\textbf{{{value_str}}} {pct_str}"
    else:
        return f"{value_str} {pct_str}"


def generate_latex_table(table_data):
    """Generate LaTeX table code."""
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{l|ccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Steps} & \\textbf{Vanilla SA} & \\textbf{Neural SA PPO} & \\textbf{Added LSTM and $\\Delta E$ (Ours)} \\\\")
    latex_lines.append("\\hline")
    
    for row in table_data:
        size_K_label = f"{row['size_K']}"
        baseline_str = format_value(row['baseline']['value'], row['baseline']['pct'], row['baseline']['is_best'])
        nsa_str = format_value(row['nsa']['value'], row['nsa']['pct'], row['nsa']['is_best'])
        rlbsa_str = format_value(row['rlbsa']['value'], row['rlbsa']['pct'], row['rlbsa']['is_best'])
        
        latex_lines.append(f"{size_K_label} & {baseline_str} & {nsa_str} & {rlbsa_str} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Wyniki algorytmów dla optymalizacji funkcji Rosenbrock'a dla różnej liczby kroków. Niższa wartość oznacza lepszy wynik.}")
    latex_lines.append("\\label{tab:rosenbrock_results}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "outputs" / "results" / "rosenbrock_final_1.json"
    
    # Load and process data
    print(f"Loading data from {json_path}")
    data = load_results(json_path)
    table_data = extract_data(data)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(table_data)
    
    # Print to console
    print("\n" + "="*80)
    print("LaTeX Table:")
    print("="*80)
    print(latex_table)
    print("="*80)
    
    # Also save to file
    output_file = project_root / "outputs" / "results" / "rosenbrock_table.tex"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    print(f"\nTable saved to: {output_file}")


if __name__ == "__main__":
    main()
