from import_helm_lite import read_helm_list
from calculate_statistics import calculate_stats, num_words, rank_questions
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

output_dir = 'analysis_output'
os.makedirs(output_dir, exist_ok=True)

RANKED_RESULTS_FILE = os.path.join(output_dir, 'ranked_results.pkl')
BENCHMARK_MODEL_FILE = os.path.join(output_dir, 'benchmark_to_model.pkl')

# First, let's extract benchmark from instance string
def get_benchmark(instance_string):
    # Split on 'model=' and take everything before it
    return instance_string.split('model=')[0].strip(',')

def get_model_from_instance(instance_string):
    if 'model=' in instance_string:
        return instance_string.split('model=')[1].split(',')[0].strip()
    return None
 
def calculate_ranked_results(metrics):
    '''
    Calculate the ranked results for each benchmark and save them to a file
    '''
    
    ########################################################
    # Group dataframes by benchmark
    ########################################################

    benchmark_to_dfs = {}
    benchmark_to_model = {}  # New dictionary to store model names

    for version in versions[:1]:
        for instance_df in os.listdir(runs + version):
            if instance_df == 'eval_cache':
                continue
            
            df_1 = read_helm_list(version=version, instance=instance_df + '/')
            
            benchmark = get_benchmark(instance_df)
            model = get_model_from_instance(instance_df)
            
            if benchmark not in benchmark_to_dfs:
                benchmark_to_dfs[benchmark] = []
                benchmark_to_model[benchmark] = model
            benchmark_to_dfs[benchmark].append(df_1)

    ########################################################
    # Rank questions by metric
    ########################################################

    ranked_results = {}
    for benchmark, dfs in benchmark_to_dfs.items():
        
        sample_df = dfs[0]
        # Find which metric is present in this benchmark's dataframes
        metric_name = None
        for possible_metric in metrics:
            if possible_metric in sample_df.columns:
                metric_name = possible_metric
                break
            
        if metric_name is None:
            print(f"Warning: No known metric found in columns for benchmark {benchmark}")
            continue
        
        ranked_results[benchmark] = rank_questions(dfs, metric_name=metric_name, desired_stat_name='num_words')
    
    # Save the results
    with open(RANKED_RESULTS_FILE, 'wb') as f:
        pickle.dump(ranked_results, f)
    with open(BENCHMARK_MODEL_FILE, 'wb') as f:
        pickle.dump(benchmark_to_model, f)
    
    return ranked_results, benchmark_to_model
    
########################################################
# Calculate statistics
########################################################


def visualize_response_trends(benchmark_df, metric_name, n_groups=3, benchmark_name=None, benchmark_number=None, total_benchmarks=None, model_names=None):
    """
    Create visual analysis of response trends with plots and formatted tables
    """
    # Create agreement groups based on std
    benchmark_df['agreement_level'] = pd.qcut(benchmark_df['std'], n_groups, 
                                            labels=['High', 'Medium', 'Low'])
    
    pred_cols = [col for col in benchmark_df.columns if col.startswith('predicted_text_')]
    metric_cols = [col for col in benchmark_df.columns if col.startswith(metric_name + '_')]
    
    # Set up the plotting style
    plt.style.use('default')  # Use default style instead of seaborn
    fig = plt.figure(figsize=(20, 15))  # Increased figure size
    
    # 1. Response Length Distribution Plot
    plt.subplot(2, 2, 1)
    data_for_box = []
    labels = []
    for level in ['High', 'Medium', 'Low']:
        level_df = benchmark_df[benchmark_df['agreement_level'] == level]
        lengths = []
        for col in pred_cols:
            lengths.extend(level_df[col].apply(lambda x: len(str(x).split())).values)
        data_for_box.append(lengths)
    
    # Fix deprecation warning by using tick_labels instead of labels
    plt.boxplot(data_for_box, tick_labels=['High', 'Medium', 'Low'])
    plt.title('Response Length Distribution\nby Agreement Level', fontsize=12, pad=20)
    plt.ylabel('Number of Words', fontsize=10)
    plt.xlabel('Agreement Level', fontsize=10)
    
    # 2. Metric Scores Distribution
    plt.subplot(2, 2, 2)
    for level in ['High', 'Medium', 'Low']:
        level_df = benchmark_df[benchmark_df['agreement_level'] == level]
        scores = level_df[metric_cols].mean(axis=1)
        plt.hist(scores, alpha=0.5, label=level, bins=20)
    
    plt.title(f'{metric_name}\nDistribution by Agreement Level', fontsize=12, pad=20)
    plt.xlabel('Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend()
    
    # 3. Question Length vs Agreement
    plt.subplot(2, 2, 3)
    question_lengths = benchmark_df['input.text'].apply(len)
    plt.scatter(question_lengths, benchmark_df['std'], alpha=0.5, s=50)
    plt.title('Question Length vs Agreement Level', fontsize=12, pad=20)
    plt.xlabel('Question Length (characters)', fontsize=10)
    plt.ylabel('Standard Deviation\n(lower = higher agreement)', fontsize=10)
    
    # 4. Agreement Level Distribution
    plt.subplot(2, 2, 4)
    counts = benchmark_df['agreement_level'].value_counts()
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), counts.index, rotation=0)
    plt.title('Distribution of Agreement Levels', fontsize=12, pad=20)
    plt.xlabel('Agreement Level', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    
    # Add a title to the figure with benchmark and model information
    if benchmark_name and model_names:
        plt.suptitle(f"Analysis for {benchmark_name}\nModels: {', '.join(model_names)}", 
                    fontsize=14, y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    # Save the plot to a file
    plot_path = os.path.join(benchmark_dir, 'analysis_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary tables and write to file instead of printing
    with open(os.path.join(benchmark_dir, 'analysis_summary.txt'), 'w') as f:
        # Add progress tracking information to the output file
        if benchmark_name and benchmark_number and total_benchmarks:
            f.write(f"Benchmark {benchmark_number}/{total_benchmarks}: {benchmark_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Add model information
            if model_names:
                f.write("Models analyzed:\n")
                for i, model in enumerate(model_names, 1):
                    f.write(f"{i}. {model}\n")
                f.write("\n")
        
        f.write("=== Summary Statistics ===\n\n")
        
        # Summary table for each agreement level
        summary_data = []
        for level in ['High', 'Medium', 'Low']:
            level_df = benchmark_df[benchmark_df['agreement_level'] == level]
            
            # Calculate statistics - fix deprecation warning by using map instead of applymap
            avg_length = level_df[pred_cols].apply(lambda x: x.map(lambda y: len(str(y).split()))).mean().mean()
            avg_score = level_df[metric_cols].mean().mean()
            n_questions = len(level_df)
            std_range = (level_df['std'].min(), level_df['std'].max())
            
            summary_data.append([
                level, n_questions, f"{avg_length:.1f}", f"{avg_score:.3f}",
                f"{std_range[0]:.3f} - {std_range[1]:.3f}"
            ])
        
        f.write(tabulate(summary_data, 
                        headers=['Agreement', 'N Questions', 'Avg Length', f'Avg {metric_name}', 'Std Range'],
                        tablefmt='grid'))
        
        # Sample responses table with improved formatting
        f.write("\n\n=== Sample Questions and Responses ===\n\n")
        sample_data = []
        for level in ['High', 'Medium', 'Low']:
            level_df = benchmark_df[benchmark_df['agreement_level'] == level]
            # Get the row with median std for more representative samples
            median_idx = level_df['std'].abs().sort_values().index[len(level_df)//2]
            sample_row = level_df.loc[median_idx]
            
            responses = [sample_row[col] for col in pred_cols[:3]]
            response_text = ' | '.join(str(r) for r in responses)
            
            sample_data.append([
                level,
                sample_row['input.text'][:100] + '...' if len(sample_row['input.text']) > 100 else sample_row['input.text'],
                response_text
            ])
        
        f.write(tabulate(sample_data,
                        headers=['Agreement', 'Question', 'Model Responses (first 3)'],
                        tablefmt='grid',
                        maxcolwidths=[10, 50, 50]))
    
    # Save the full analysis data to CSV
    benchmark_df.to_csv(os.path.join(benchmark_dir, 'full_analysis_data.csv'))
    
    # Print a confirmation message to the terminal
    print(f"Analysis completed for benchmark. Results saved to {benchmark_dir}")



def load_or_calculate_results(metrics):
    if os.path.exists(RANKED_RESULTS_FILE) and os.path.exists(BENCHMARK_MODEL_FILE):
        print("Loading cached results...")
        with open(RANKED_RESULTS_FILE, 'rb') as f:
            ranked_results = pickle.load(f)
        with open(BENCHMARK_MODEL_FILE, 'rb') as f:
            benchmark_to_model = pickle.load(f)
    else:
        print("Calculating results...")
        ranked_results, benchmark_to_model = calculate_ranked_results(metrics)
    return ranked_results, benchmark_to_model

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
ranked_results, benchmark_to_model = load_or_calculate_results(metrics)

# Run the visualization for each benchmark
total_benchmarks = len(ranked_results)
print(f"\nStarting analysis of {total_benchmarks} benchmarks...")

for i, (benchmark, ranked_df) in enumerate(ranked_results.items(), 1):
    print(f"\nProcessing benchmark {i}/{total_benchmarks}: {benchmark}")
    print("=" * 80)
    
    # Get the model name from our stored mapping
    model_names = []
    if benchmark in benchmark_to_model and benchmark_to_model[benchmark]:
        model_names = [benchmark_to_model[benchmark]]
        print(f"Model identified: {model_names}")
    
    metric_name = None
    for possible_metric in metrics:
        if any(col.startswith(possible_metric) for col in ranked_df.columns):
            metric_name = possible_metric
            break
    
    if metric_name:
        benchmark_dir = os.path.join(output_dir, benchmark.replace(':', '_'))
        os.makedirs(benchmark_dir, exist_ok=True)
        visualize_response_trends(ranked_df, metric_name, benchmark_name=benchmark, 
                                 benchmark_number=i, total_benchmarks=total_benchmarks,
                                 model_names=model_names)
    else:
        print(f"Warning: No known metric found for benchmark {benchmark}")

print(f"\nAnalysis complete! Results saved to {output_dir}")