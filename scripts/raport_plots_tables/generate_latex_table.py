#!/usr/bin/env python3
"""
Generate LaTeX table from knap_final_1.json results.
Bolds best results and calculates percentages compared to best.
"""

import json
from pathlib import Path


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_data(data):
    """Extract data for all problem sizes."""
    results = data['results']
    problem_sizes = sorted(data['config']['experiments']['knapsack']['problem_sizes'])
    
    table_data = []
    
    for size in problem_sizes:
        key = f"knapsack_{size}_K5"
        if key in results:
            # Get original negative values
            baseline_orig = results[key].get('baseline', {}).get('avg', 0)
            nsa_orig = results[key].get('nsa', {}).get('avg', 0)
            rlbsa_orig = results[key].get('rlbsa', {}).get('avg', 0)
            
            # Convert to absolute values for display
            baseline_avg = abs(baseline_orig)
            nsa_avg = abs(nsa_orig)
            rlbsa_avg = abs(rlbsa_orig)
            
            # Find best (most negative original value = best)
            # Compare original values (more negative = better)
            values_orig = {
                'baseline': baseline_orig,
                'nsa': nsa_orig,
                'rlbsa': rlbsa_orig
            }
            best_method = min(values_orig, key=values_orig.get)  # min of negative = most negative
            best_value_orig = values_orig[best_method]
            best_value_abs = abs(best_value_orig)
            
            # Calculate percentages compared to best (using absolute values)
            # Percentage shows how much WORSE each value is compared to best
            # Formula: abs((value - best) / best) * 100
            # Higher absolute value = better (more negative original value)
            baseline_pct = abs((baseline_avg - best_value_abs) / best_value_abs * 100) if best_value_abs > 0 else 0
            nsa_pct = abs((nsa_avg - best_value_abs) / best_value_abs * 100) if best_value_abs > 0 else 0
            rlbsa_pct = abs((rlbsa_avg - best_value_abs) / best_value_abs * 100) if best_value_abs > 0 else 0
            
            # For the best method, percentage should be 0.00%
            if best_method == 'baseline':
                baseline_pct = 0.0
            elif best_method == 'nsa':
                nsa_pct = 0.0
            elif best_method == 'rlbsa':
                rlbsa_pct = 0.0
            
            table_data.append({
                'size': size,
                'baseline': {'value': baseline_avg, 'pct': baseline_pct, 'is_best': best_method == 'baseline'},
                'nsa': {'value': nsa_avg, 'pct': nsa_pct, 'is_best': best_method == 'nsa'},
                'rlbsa': {'value': rlbsa_avg, 'pct': rlbsa_pct, 'is_best': best_method == 'rlbsa'},
            })
    
    return table_data


def format_value(value, pct, is_best):
    """Format a value with percentage, bolding if best."""
    value_str = f"{value:.2f}"
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
    latex_lines.append("\\textbf{Problem Size} & \\textbf{Vanilla SA} & \\textbf{Neural SA PPO} & \\textbf{Added LSTM and $\\Delta E$ (Ours)} \\\\")
    latex_lines.append("\\hline")
    
    for row in table_data:
        size_label = f"Knap{row['size']}"
        baseline_str = format_value(row['baseline']['value'], row['baseline']['pct'], row['baseline']['is_best'])
        nsa_str = format_value(row['nsa']['value'], row['nsa']['pct'], row['nsa']['is_best'])
        rlbsa_str = format_value(row['rlbsa']['value'], row['rlbsa']['pct'], row['rlbsa']['is_best'])
        
        latex_lines.append(f"{size_label} & {baseline_str} & {nsa_str} & {rlbsa_str} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Wyniki algorytmów dla różnych rozmiarów problemu plecakowego. Wyższa wartość oznacza lepszy wynik.}")
    latex_lines.append("\\label{tab:knapsack_results}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    # json_path = project_root / "outputs" / "results" / "knap_final_1.json"
    json_path = project_root / "outputs" / "results" / "results_binpacking.json"

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
    output_file = project_root / "outputs" / "results" / "knapsack_table.tex"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    print(f"\nTable saved to: {output_file}")


if __name__ == "__main__":
    main()
