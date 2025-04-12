from helper_functions import read_helm_list, rank_questions, visualize_response_trends, calculate_ranked_results, load_or_calculate_results
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
import numpy as np
import seaborn as sns
from tabulate import tabulate
import pickle

versions = [f'v1.{str(i)}.0/' for i in range(14)]
runs = '/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/'

analysis_dir = 'analysis_output'
output_dir = os.path.join(analysis_dir, '04092025_analysis_plots')
os.makedirs(output_dir, exist_ok=True)

RANKED_RESULTS_FILE = os.path.join(analysis_dir, 'ranked_results.pkl')
BENCHMARK_MODEL_FILE = os.path.join(analysis_dir, 'benchmark_to_model.pkl')


########################################################
# Main
########################################################

metrics = [
    'stats.bleu_4',
    'stats.exact_match',
    'stats.math_equiv_chain_of_thought',
    'stats.f1_score',
    'stats.final_number_exact_match',
    'stats.quasi_exact_match'
]

# Load or calculate the results
ranked_results, benchmark_to_model = load_or_calculate_results(metrics, runs, versions, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE)

# Run the visualization for each benchmark
total_num_benchmarks = len(ranked_results)
print(f"\nStarting analysis of {total_num_benchmarks} benchmarks...")

for i, (benchmark, ranked_df) in enumerate(ranked_results.items(), 1):
    print(f"\nProcessing benchmark {i}/{total_num_benchmarks}: {benchmark}")
    print("=" * 80)
    
    # Get the model name from our stored mapping
    model_names = []
    if benchmark in benchmark_to_model and benchmark_to_model[benchmark]:
        model_names = benchmark_to_model[benchmark]  # Already a list of model names
        # print(f"Models identified: {model_names}")
    
    metric_name = None
    # assumes only 1 metric is used per benchmark in the HELM LITE dataframes
    for possible_metric in metrics:
        if any(col.startswith(possible_metric) for col in ranked_df.columns):
            metric_name = possible_metric
            break
    
    if metric_name:
        benchmark_dir = os.path.join(output_dir, benchmark.replace(':', '_'))
        os.makedirs(benchmark_dir, exist_ok=True)
        visualize_response_trends(ranked_df, metric_name, benchmark_dir=benchmark_dir, benchmark_name=benchmark, 
                                 benchmark_number=i, total_benchmarks=total_num_benchmarks,
                                 model_names=model_names)
    else:
        print(f"Warning: No known metric found for benchmark {benchmark}")

print(f"\nAnalysis complete! Results saved to {output_dir}")