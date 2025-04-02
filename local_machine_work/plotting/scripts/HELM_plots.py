import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

print('libraries loaded')

###############################
# PROCESS DATA INTO DATAFRAME #
###############################

dfs = []

for file in os.listdir("../../HELM-LITE/HELM_lite_accuracy_csvs"):
    file_path = os.path.join("../../HELM-LITE/HELM_lite_accuracy_csvs", file)
    
    match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", file)
    if match:
        date_str = match.group(1)
    else:
        # If the date is not found, skip this file
        continue
    
    df = pd.read_csv(file_path)
    
    # UNCOMMENT THIS TO SEE THE NUMBER OF MODELS AT EACH DATE
    # print(len(df), date_str)
    df['Date'] = pd.to_datetime(date_str)
    
    dfs.append(df)
    
# Concatenate all the DataFrames
all_results = pd.concat(dfs, ignore_index=True)

def plot_multiple_normalized(df, date_col, metric_cols, output_file, normalization='Raw'):
    """
    For each metric in `metric_cols`, this function:
      - Groups the DataFrame by `date_col` and extracts the maximum value of the metric per date.
      - Normalizes the timeseries by dividing each value by the overall max for that metric.
      - Plots all normalized timeseries on the same plot using Seaborn.
      
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame which includes a date column and one or more metric columns.
    date_col : str
        The name of the column containing date information.
    metric_cols : list of str
        A list of column names (metrics) to process and plot.
    output_file : str
        The path to save the output plot.
    normalization : str
        The column name to plot, either 'raw', 'divide_by_max', or 'zero_to_one'
        
    Returns
    -------
    pd.DataFrame
        A DataFrame in long format containing the date, metric, and normalized values.
    """
    # Ensure the date column is in datetime format.
    df[date_col] = pd.to_datetime(df[date_col])
    
    normalized_list = []
    
    # Process each metric column individually.
    for metric in metric_cols:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        # Group by date and get the maximum value for the current metric.
        max_series = df.groupby(date_col)[metric].max().reset_index()
        
        ## ADD NORMALIZATION COLUMNS
        overall_max = max_series[metric].max()
        max_series['divide_by_max'] = max_series[metric] / overall_max
        max_series['raw'] = max_series[metric]
        
        # Get the first and last values of the metric timeseries.
        first_val = max_series.iloc[0][metric]
        last_val = max_series.iloc[-1][metric]
        range_val = last_val - first_val
        
        # Compute the normalized value such that the first value is 0 and the last is 1.
        if range_val == 0:
            # Avoid division by zero. If the timeseries does not change, set all normalized values to 0.
            max_series['zero_to_one'] = 0
        else:
            max_series['zero_to_one'] = (max_series[metric] - first_val) / range_val
        
        # Add a column to indicate which metric this row corresponds to.
        max_series['Metric'] = metric
        
        # Keep only the date, normalized value, and metric columns.
        normalized_list.append(max_series[[date_col, normalization, 'Metric']])
    
    # Combine all metrics into one DataFrame in long format.
    combined_normalized = pd.concat(normalized_list, ignore_index=True)
    
    # Create the plot.
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_normalized, x=date_col, y=normalization, hue='Metric', marker='o')
    plt.xlabel("Date")
    plt.ylabel(f"Metric ({normalization})")
    plt.title("Maximum Metrics HELM-LITE")
    
    unique_dates = combined_normalized[date_col].unique()
    plt.xticks(unique_dates, [date.strftime("%Y-%m-%d") for date in unique_dates], rotation=45)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_file)
    
    return combined_normalized

# Example usage:
# Assuming `combined_df` is your DataFrame with a "Date" column and multiple metric columns:
metrics_to_plot = ['Mean win rate', 'NarrativeQA - F1',
       'NaturalQuestions (open) - F1', 'NaturalQuestions (closed) - F1',
       'OpenbookQA - EM', 'MMLU - EM', 'MATH - Equivalent (CoT)', 'GSM8K - EM',
       'LegalBench - EM', 'MedQA - EM', 'WMT 2014 - BLEU-4']

normalization_options = ['raw', 'divide_by_max', 'zero_to_one']

normalization_choice = normalization_options[2] # CHOOSE THIS!!

normalized_data = plot_multiple_normalized(all_results, 
                                           date_col="Date", 
                                           metric_cols=metrics_to_plot,
                                           output_file=f"../plots/HELM/{normalization_choice}.png",
                                           normalization=normalization_choice)

print('plot saved')

