import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import *
import os

def get_question_type(row):
    """Helper function to determine question type from references"""
    try:
        ref = row['references']
        # Check if it's a Yes/No MCQ
        is_mcq = ref.lower() == 'yes (correct)' or ref.lower() == 'no (correct)'
        if is_mcq:
            return 'Single-Letter MCQ'
        # Check if it's a math question
        elif any(symbol in str(ref) for symbol in ['\\frac', '\\boxed', '$', '\\[', '\\]', '+', '-', '=', '^']):
            return 'Math'
        else:
            return 'Text'
    except Exception as e:
        return 'Unknown'

def plot_violin(df, save_path):
    """Create violin plot showing distribution of agreement by question type."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='question_type', y='std')
    plt.title('Model Agreement Distribution by Question Type (Violin Plot)')
    plt.xlabel('Question Type')
    plt.ylabel('Model Agreement (Standard Deviation)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_violin_box(df, save_path):
    """Create violin plot with box plot overlay."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='question_type', y='std', inner='box')
    plt.title('Model Agreement Distribution by Question Type (Violin + Box Plot)')
    plt.xlabel('Question Type')
    plt.ylabel('Model Agreement (Standard Deviation)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_kde(df, save_path):
    """Create KDE plot for each question type."""
    plt.figure(figsize=(12, 6))
    for qtype in ['Single-Letter MCQ', 'Math', 'Text']:
        mask = df['question_type'] == qtype
        if any(mask):
            sns.kdeplot(data=df[mask]['std'], label=qtype)
    plt.title('Model Agreement Density by Question Type (KDE)')
    plt.xlabel('Model Agreement (Standard Deviation)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_stacked_histogram(df, save_path):
    """Create stacked histogram showing distribution of agreement levels."""
    plt.figure(figsize=(12, 6))
    
    # Define agreement levels using hardcoded std ranges
    df['agreement_level'] = pd.cut(df['std'], 
                                 bins=[0, 0.15, 0.3, float('inf')],
                                 labels=['High', 'Medium', 'Low'],
                                 include_lowest=True)
    
    # Create normalized stacked histogram
    df_pct = df.groupby(['question_type', 'agreement_level']).size().unstack()
    df_pct = df_pct.div(df_pct.sum(axis=1), axis=0) * 100
    
    df_pct.plot(kind='bar', stacked=True)
    plt.title('Distribution of Agreement Levels by Question Type\n(High: 0-0.15, Medium: 0.15-0.3, Low: >0.3 std)')
    plt.xlabel('Question Type')
    plt.ylabel('Percentage')
    plt.legend(title='Agreement Level')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_model_agreement(output_dir, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE):
    """
    Read data from helper_functions and create agreement plots for all benchmarks.
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, 'analysis_log.txt')
    
    def log_print(*args, **kwargs):
        """Helper function to print to console and append formatted text to log file"""
        print(*args, **kwargs)  # Print to console
        with open(log_file, 'a') as f:  # Append to log file
            print(*args, **kwargs, file=f)
    
    # Clear existing log file
    with open(log_file, 'w') as f:
        f.write("MODEL AGREEMENT ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
    
    # Define parameters as used in helper_functions
    metrics = [
        'stats.bleu_4',
        'stats.exact_match',
        'stats.math_equiv_chain_of_thought',
        'stats.f1_score',
        'stats.final_number_exact_match',
        'stats.quasi_exact_match'
    ]
    runs = '/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/'
    versions = ['v1.0.0/']
    
    # Load or calculate results
    ranked_results, benchmark_to_model = load_or_calculate_results(
        metrics, runs, versions, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE
    )
    
    # Process each benchmark and combine essential columns
    combined_data = []
    for benchmark_name, df in ranked_results.items():
        print(f"\nProcessing benchmark: {benchmark_name}")
        
        # Add question type to the DataFrame
        df['question_type'] = df.apply(get_question_type, axis=1)
        
        # Extract essential columns
        essential_df = df[['instance_id', 'std', 'question_type']].copy()
        essential_df['benchmark'] = benchmark_name
        combined_data.append(essential_df)
    
    # Combine all benchmarks
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    
    # Calculate overall statistics first
    with open(log_file, 'a') as f:
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write("Number of questions by type:\n")
        f.write(combined_df['question_type'].value_counts().to_string() + "\n\n")
        
        f.write("AGREEMENT STATISTICS BY QUESTION TYPE\n")
        f.write("-" * 30 + "\n")
        
        for qtype in ['Single-Letter MCQ', 'Math', 'Text']:
            mask = combined_df['question_type'] == qtype
            if any(mask):
                data = combined_df[mask]['std']
                f.write(f"\n{qtype}:\n")
                f.write(f"Total questions: {sum(mask)}\n")
                f.write(f"Standard deviation range: {data.min():.3f} - {data.max():.3f}\n")
                f.write(f"Mean standard deviation: {data.mean():.3f}\n")
                f.write(f"Quartiles (Q1-Q3): {data.quantile(0.25):.3f} - {data.quantile(0.75):.3f}\n")
                
                # Calculate agreement level distribution
                agreement_levels = pd.cut(data, 
                                       bins=[0, 0.15, 0.3, float('inf')],
                                       labels=['High', 'Medium', 'Low'],
                                       include_lowest=True)
                level_counts = agreement_levels.value_counts()
                level_percentages = level_counts / len(data) * 100
                
                f.write("Agreement Level Distribution:\n")
                for level in ['High', 'Medium', 'Low']:
                    if level in level_percentages:
                        f.write(f"  {level}: {level_percentages[level]:.1f}% ({level_counts[level]} questions)\n")
                f.write("\n")
    
    # Generate plots
    save_path = os.path.join(output_dir, 'scatter_agreement_by_type.png')
    plot_agreement_by_question_type(combined_df, save_path)
    
    save_path = os.path.join(output_dir, 'violin_agreement_by_type.png')
    plot_violin(combined_df, save_path)
    
    save_path = os.path.join(output_dir, 'violin_box_agreement_by_type.png')
    plot_violin_box(combined_df, save_path)
    
    save_path = os.path.join(output_dir, 'kde_agreement_by_type.png')
    plot_kde(combined_df, save_path)
    
    save_path = os.path.join(output_dir, 'stacked_histogram_agreement_by_type.png')
    plot_stacked_histogram(combined_df, save_path)
    
    with open(log_file, 'a') as f:
        f.write("\nAnalysis complete.\n")

def plot_agreement_by_question_type(df, save_path):
    """
    Plot model agreement (std) for different types of questions.
    
    Args:
        df: DataFrame containing 'std' and 'question_type' columns
        save_path: Path where to save the plot
        log_file: Path to the log file
    """
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot with jitter
    for i, qtype in enumerate(['Single-Letter MCQ', 'Math', 'Text']):
        mask = df['question_type'] == qtype
        if not any(mask):
            continue
            
        # Add jitter to x-coordinates
        x = np.random.normal(i, 0.04, size=sum(mask))
        y = df[mask]['std'].values
        
        # Check for NaN values
        valid_mask = ~np.isnan(y)
        if sum(valid_mask) > 0:
            plt.scatter(x[valid_mask], y[valid_mask], alpha=0.5, s=2, label=f"{qtype} (n={sum(valid_mask)})")
    
    # Customize plot
    plt.xticks(range(3), ['Single-Letter MCQ', 'Math', 'Text'])
    plt.xlabel('Question Type')
    plt.ylabel('Model Agreement (Standard Deviation)')
    plt.title('Model Agreement by Question Type')
    
    # Add mean and quartile lines for each category
    for i, qtype in enumerate(['Single-Letter MCQ', 'Math', 'Text']):
        mask = df['question_type'] == qtype
        if not any(mask):
            continue
            
        stds = df[mask]['std']
        valid_stds = stds[~stds.isna()]
        if len(valid_stds) == 0:
            continue
            
        mean = valid_stds.mean()
        q1 = valid_stds.quantile(0.25)
        q3 = valid_stds.quantile(0.75)
        
        # Plot mean line
        plt.hlines(mean, i-0.2, i+0.2, colors='red', linestyles='solid', label=f'{qtype} Mean' if i==0 else '')
        # Plot quartile lines
        plt.hlines([q1, q3], i-0.1, i+0.1, colors='black', linestyles='dashed', 
                  label='Quartiles' if i==0 else '')
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    analysis_dir = 'analysis_output'
    output_dir = os.path.join(analysis_dir, '04142025_predict_agreement_clustering')
    os.makedirs(output_dir, exist_ok=True)
    
    RANKED_RESULTS_FILE = os.path.join(analysis_dir, 'ranked_results.pkl')
    BENCHMARK_MODEL_FILE = os.path.join(analysis_dir, 'benchmark_to_model.pkl')
    
    analyze_model_agreement(output_dir, RANKED_RESULTS_FILE, BENCHMARK_MODEL_FILE)



