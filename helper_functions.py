import pandas as pd
from functools import reduce
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

########################################################
#### READ DATA #########################################
########################################################

def read_helm_list(runs='/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/', version='v1.11.0/', instance='natural_qa:mode=closedbook,model=meta_llama-3.3-70b-instruct-turbo/'):
    '''
    Given a directory path to HELM LITE data for instance-level runs for a particular model on a particular benchmark, returns any the relevant information as 1 dataframe.  

    Input:
        runs: path to Yifan's runs directory
        version: version of HELM LITE (as of April 2025, this is v1.0.0 to v1.13.0)
        instance: instance of HELM LITE (specifies model and benchmark)
    Output:
        df: dataframe of the HELM LITE data for this particular model-benchmark pair
    '''
    version = version
    instance = instance
    path = runs + version + instance
    
    # several json files to choose from in each directory, pick the appropriate ones
    relevant_files = {'display_predictions': ['instance_id', 'predicted_text', 'stats'],
    'instances': ['id', 'input', 'references']}
    
    df_list = []
    for file in relevant_files.keys():
        with open(path + file + '.json') as json_file:
            full_dict = json.load(json_file)
        
        if file == 'instances':
            # Process instances file differently to handle references
            processed_records = []
            for record in full_dict:
                # Start with the basic record info
                processed_record = {
                    'id': record['id'],
                    'input': record['input']
                }
                
                # Process references by combining text and tags
                reference_texts = []
                for ref in record['references']:
                    text = ref['output']['text']
                    tags = ref['tags']
                    if tags:  # Only add tags if they exist
                        text = f"{text} ({', '.join(tags)})"
                    reference_texts.append(text)
                
                # Join all references with newlines
                processed_record['references'] = 'SALILSPLITCHECK'.join(reference_texts)
                processed_records.append(processed_record)
                
            df_temp = pd.json_normalize(processed_records)
            df_temp = df_temp.rename(columns={'id': 'instance_id'})
        else:
            only_relevant_keys = [{key: record[key] for key in relevant_files[file]} for record in full_dict]
            df_temp = pd.json_normalize(only_relevant_keys)
            
        df_list.append(df_temp)
        
    df = reduce(
        lambda left, right: pd.merge(left, right, on="instance_id", how="outer"),
        df_list
    )
    
    return df

########################################################
#### PROCESS DATA ######################################
########################################################

def rank_questions(dfs, metric_name='stats.exact_match'):
    '''
    takes in a list of dfs from 1 benchmark, combines them into 1 df (rows = benchmark questions, columns = model predictions and metric values)
    and then ranks the questions by their agreement on the metric 
        so we have many models (many columns), and for one question (one row), each model has a prediction and a metric value. 
        We calculate std of the metric values for each question, and then rank the questions by their std. So std is proxy for model agreement on the question.
        
    output df has rows corresponding to instance_id (different questions) and cols predicted_text (for each model), metric value (for each model) and std (one column)
    '''
    
    processed_dfs = []

    for i, df in enumerate(dfs):
        if i == 0:
            # Include references in the first dataframe
            df_subset = df[['instance_id', 'input.text', 'references', 'predicted_text', metric_name]].copy()
        else:
            df_subset = df[['instance_id', 'predicted_text', metric_name]].copy()
            
        df_subset = df_subset.rename(columns={
            'predicted_text': f'predicted_text_{i}',
            metric_name: metric_name + f'_{i}',
        })
        processed_dfs.append(df_subset)
        
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='instance_id'), processed_dfs)
    
    # Compute standard deviation across the metric columns for each question.
    metric_cols = [col for col in merged_df.columns if col.startswith(metric_name)]
    merged_df['std'] = merged_df[metric_cols].std(axis=1)
    
    ranked_questions = merged_df.sort_values(by='std')
    
    return ranked_questions

def calculate_ranked_results(metrics, runs, versions, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE):
    '''
    Calculate the ranked results for each benchmark (by grabbing them from Yifan's runs directory) and save them to a file
    
    Input:
        metrics: list of the metrics that are used by benchmarks in HELM LITE (in Yifan's runs directory)
        runs: path to Yifan's runs directory
        versions: list of HELM LITE versions to consider (as of April 2025, this is v1.0.0 to v1.13.0)
    Output:
        ranked_results: dictionary of benchmark name to ranked dataframe
        benchmark_to_model: dictionary of benchmark name to list of model names
    '''
    
    ########################################################
    # HELPER FUNCTIONS ####################################
    ########################################################

    # First, let's extract benchmark from instance string
    def get_benchmark(instance_string):
        # Split on 'model=' and take everything before it
        return instance_string.split('model=')[0].strip(',')

    def get_model_from_instance(instance_string):
        if 'model=' in instance_string:
            return instance_string.split('model=')[1].split(',')[0].strip()
        return None
    
    ########################################################
    # Group dataframes by benchmark
    ########################################################

    benchmark_to_dfs = {}
    benchmark_to_model = {}  # New dictionary to store model names

    for version in versions[:1]:
        for instance_df in os.listdir(runs + version):
            if instance_df == 'eval_cache':
                continue
            
            df_1 = read_helm_list(runs=runs, version=version, instance=instance_df + '/')
            
            benchmark = get_benchmark(instance_df)
            model = get_model_from_instance(instance_df)
            
            if benchmark not in benchmark_to_dfs:
                benchmark_to_dfs[benchmark] = []
                benchmark_to_model[benchmark] = []
            benchmark_to_dfs[benchmark].append(df_1)
            benchmark_to_model[benchmark].append(model)
            
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
        
        ranked_results[benchmark] = rank_questions(dfs, metric_name=metric_name)
    
    # Save the results
    with open(RANKED_RESULTS_FILE, 'wb') as f:
        pickle.dump(ranked_results, f)
    with open(BENCHMARK_MODEL_FILE, 'wb') as f:
        pickle.dump(benchmark_to_model, f)
    
    return ranked_results, benchmark_to_model



########################################################
# PLOT DATA ############################################
########################################################


def visualize_response_trends(benchmark_df, metric_name, benchmark_dir, n_groups=3, benchmark_name=None, benchmark_number=None, total_benchmarks=None, model_names=None):
    """
    Create visual analysis of response trends with plots and formatted tables
    """
    # Create agreement groups based on fixed std ranges
    std_ranges = [0, 0.15, 0.30, float('inf')]
    benchmark_df['agreement_level'] = pd.cut(benchmark_df['std'], 
                                           bins=std_ranges,
                                           labels=['High', 'Medium', 'Low'])
    

    
    pred_cols = [col for col in benchmark_df.columns if col.startswith('predicted_text_')]
    metric_cols = [col for col in benchmark_df.columns if col.startswith(metric_name + '_')]
    
    # Set up the plotting style
    plt.style.use('default')  # Use default style instead of seaborn
    fig = plt.figure(figsize=(20, 18))  # Increased figure height for new plot
    
    # 1. Response Length Distribution Plot
    plt.subplot(3, 2, 1)  # Changed to 3x2 grid
    data_for_box = []
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
    plt.subplot(3, 2, 2)
    
    # First pass to determine overall range for consistent binning
    all_scores = []
    for level in ['High', 'Medium', 'Low']:
        level_df = benchmark_df[benchmark_df['agreement_level'] == level]
        scores = level_df[metric_cols].mean(axis=1)
        all_scores.extend(scores)
    
    # Create fixed bins across the entire range
    min_score, max_score = min(all_scores), max(all_scores)
    bins = np.linspace(min_score, max_score, 30)  # 20 evenly spaced bins
    
    # Plot histograms with consistent bins
    for level in ['High', 'Medium', 'Low']:
        level_df = benchmark_df[benchmark_df['agreement_level'] == level]
        scores = level_df[metric_cols].mean(axis=1)
        
        # plt.hist(scores, bins=bins, alpha=0.5, label=f"{level} (n={len(scores)})")
        std_range = (level_df['std'].min(), level_df['std'].max())
        plt.hist(scores, bins=bins, alpha=0.5, label=f"{level} (std={std_range[0]:.3f}-{std_range[1]:.3f}, n={len(scores)})")
    
    
    plt.title(f'{metric_name}\nDistribution by Agreement Level', fontsize=12, pad=20)
    plt.xlabel('Average Score (1 data point = 1 question)', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.legend()
    
    # metric_cols = [col for col in benchmark_df.columns if col.startswith(metric_name + '_')]

    # hist_bin_values = []
    # bin_edges = []

    # for level in ['High', 'Medium', 'Low']:
    #     level_df = benchmark_df[benchmark_df['agreement_level'] == level]
    #     scores = level_df[metric_cols].mean(axis=1)
    #     n, bins, _ = plt.hist(scores, alpha=0.5, label=level, bins=20)
    #     hist_bin_values.append(n)
    #     bin_edges.append(bins)

    # plt.title(f'{metric_name}\nDistribution by Agreement Level', fontsize=12, pad=20)
    # plt.xlabel('Score', fontsize=10)
    # plt.ylabel('Count', fontsize=10)
    # plt.legend()
    
    # 3. Question Length vs Agreement
    plt.subplot(3, 2, 3)
    question_lengths = benchmark_df['input.text'].apply(len)
    plt.scatter(question_lengths, benchmark_df['std'], alpha=0.5, s=50)
    plt.title('Question Length vs Agreement Level', fontsize=12, pad=20)
    plt.xlabel('Question Length (characters)', fontsize=10)
    plt.ylabel('Standard Deviation\n(lower = higher agreement)', fontsize=10)
    
    # 4. Agreement Level Distribution
    plt.subplot(3, 2, 4)
    counts = benchmark_df['agreement_level'].value_counts()
    plt.bar(range(len(counts)), counts.values)
    plt.xticks(range(len(counts)), counts.index, rotation=0)
    plt.title('Distribution of Agreement Levels', fontsize=12, pad=20)
    plt.xlabel('Agreement Level', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    
    # 5. Response Length vs Agreement Level (new plot)
    plt.subplot(3, 2, 5)
    # Calculate average response length for each row
    avg_response_lengths = []
    for _, row in benchmark_df.iterrows():
        lengths = []
        for col in pred_cols:
            lengths.append(len(str(row[col]).split()))
        avg_response_lengths.append(np.mean(lengths))
    
    plt.scatter(avg_response_lengths, benchmark_df['std'], alpha=0.5, s=50)
    plt.title('Response Length vs Agreement Level', fontsize=12, pad=20)
    plt.xlabel('Average Response Length (words)', fontsize=10)
    plt.ylabel('Standard Deviation\n(lower = higher agreement)', fontsize=10)
    
    # Add a title to the figure with benchmark and model information
    if benchmark_name:
        plt.suptitle(f"Analysis for {benchmark_name}", 
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
                        headers=['Agreement', 'N Questions', 'Avg Response Length', f'Avg {metric_name}', 'Std Range'],
                        tablefmt='grid'))
        
        # Sample responses table with improved formatting
        f.write("\n\n=== Sample Questions and Responses ===\n\n")
        sample_data = []
        for level in ['High', 'Medium', 'Low']:
            level_df = benchmark_df[benchmark_df['agreement_level'] == level]
            if len(level_df) > 0:  # Only process if we have examples for this level
                # Get the row with median std for more representative samples
                median_idx = level_df['std'].abs().sort_values().index[len(level_df)//2]
                sample_row = level_df.loc[median_idx]
                
                responses = [sample_row[col] for col in pred_cols[:3]]
                metric_values = [sample_row[col] for col in metric_cols[:3]]
                # Format responses with line breaks and indentation, including metric values
                response_text = '\n'.join(f"Sample response {i+1} ({metric_name}={metric_val:.3f}): {str(r)}" 
                                        for i, (r, metric_val) in enumerate(zip(responses, metric_values)))
                
                # Split references using the unique delimiter and format each reference
                references = sample_row['references'].split('SALILSPLITCHECK')
                formatted_references = '\n'.join(f"Reference {i+1}: {ref}" for i, ref in enumerate(references))
                
                sample_data.append([
                    level,
                    sample_row['input.text'],
                    formatted_references,
                    response_text
                ])
            else:
                # Add a placeholder row for empty groups
                sample_data.append([
                    level,
                    "No examples in this category",
                    "N/A",
                    "N/A"
                ])
        
        f.write(tabulate(sample_data,
                        headers=['Agreement', 'Question', 'References', 'Model Responses (first 3)'],
                        tablefmt='grid',
                        maxcolwidths=[10, 50, None, None]))
    
    # Save the full analysis data to CSV
    benchmark_df.to_csv(os.path.join(benchmark_dir, 'full_analysis_data.csv'))
    
    # Print a confirmation message to the terminal
    print(f"Analysis completed for benchmark. Results saved to {benchmark_dir}")
    

########################################################
# EFFICIENTLY LOAD OR CALCULATE RESULTS ################
########################################################

def load_or_calculate_results(metrics, runs, versions, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE):
    if os.path.exists(RANKED_RESULTS_FILE) and os.path.exists(BENCHMARK_MODEL_FILE):
        print("Loading cached results...")
        with open(RANKED_RESULTS_FILE, 'rb') as f:
            ranked_results = pickle.load(f)
        with open(BENCHMARK_MODEL_FILE, 'rb') as f:
            benchmark_to_model = pickle.load(f)
    else:
        print("Calculating results because nothing cached found...")
        ranked_results, benchmark_to_model = calculate_ranked_results(metrics, runs, versions, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE)
    return ranked_results, benchmark_to_model