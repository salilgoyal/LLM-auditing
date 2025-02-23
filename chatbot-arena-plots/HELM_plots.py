import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

###############################
# PROCESS DATA INTO DATAFRAME #
###############################

dfs = []

for file in os.listdir("../HELM-LITE/HELM_lite_accuracy_csvs"):
    file_path = os.path.join("../HELM-LITE/HELM_lite_accuracy_csvs", file)
    
    match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", file)
    if match:
        date_str = match.group(1)
    else:
        # If the date is not found, skip this file
        continue
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(date_str)
    
    dfs.append(df)
    
# Concatenate all the DataFrames
all_results = pd.concat(dfs, ignore_index=True)

# records = []

# for file in os.listdir("../HELM-LITE/HELM_lite_accuracy_csvs"):
    
#     file_path = os.path.join("../HELM-LITE/HELM_lite_accuracy_csvs", file)
    
#     # Parse the date from the filename; e.g. "v1.7.0 (2024-08-08)-Table 1.csv"
#     match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", file_path)
#     if not match:
#         # If we can't parse a date, skip this file
#         continue
    
#     date_str = match.group(1)  # e.g. "2024-08-08"
#     date = pd.to_datetime(date_str)
    
#     # Read the CSV
#     df = pd.read_csv(file_path)
    
#     # Identify the row with the maximum "Mean win rate"
#     max_idx = df["Mean win rate"].idxmax()
#     max_rate = df.loc[max_idx, "Mean win rate"]
#     max_model = df.loc[max_idx, "Model"]
    
#     # Store the results
#     records.append({
#         "date": date,
#         "max_mean_win_rate": max_rate,
#         "model_with_max": max_model
#     })

# # Convert the collected records into a DataFrame
# results_df = pd.DataFrame(records)

# # Sort by date to ensure chronological plotting
# results_df.sort_values(by="date", inplace=True)

###############################
#      FUNCTION TO PLOT       #
###############################

# def plot_max_over_time(df, date_col, metric_col, output_file, model_col="Model"):
#     """
#     Groups `df` by `date_col`, finds the row with the maximum `metric_col`
#     for each date, and plots the time series of that maximum.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The DataFrame containing your data.
#     date_col : str
#         The name of the column in `df` that holds date information.
#     metric_col : str
#         The name of the numeric column in `df` for which you want to
#         plot the max value over time.
#     model_col : str, optional
#         The name of the column containing model identifiers (default "Model").
        
#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame with one row per date, containing the maximum value
#         of `metric_col` and associated model(s).
#     """
#     # Ensure the date column is in datetime format (if it's not already)
#     df[date_col] = pd.to_datetime(df[date_col])
    
#     # Group by date and find index of the maximum value in `metric_col`
#     max_indices = df.groupby(date_col)[metric_col].idxmax()
    
#     # Extract those rows
#     max_df = df.loc[max_indices].copy()
    
#     # Sort by date for chronological plotting
#     max_df.sort_values(by=date_col, inplace=True)
    
#     # Plot
#     plt.figure(figsize=(8, 5))
#     sns.lineplot(data=max_df, x=date_col, y=metric_col, marker='o')
#     plt.xlabel("Date")
#     plt.ylabel(metric_col)
#     plt.title(f"Maximum {metric_col} on HELM-LITE")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     plt.savefig(output_file)
    
#     # Optional: display which model had the maximum for each date
#     # print(max_df[[date_col, model_col, metric_col]])
    
#     return max_df

# max_mean_win_rate_df = plot_max_over_time(
#     all_results,
#     date_col="Date",
#     metric_col="Mean win rate",
#     output_file="plots/HELM/mean_win_rate.png",
#     model_col="Model"
# )

def plot_multiple_normalized(df, date_col, metric_cols, output_file, rawOrNormalized='NormalizedValue'):
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
    rawOrNormalized : str
        The column name to plot, either 'RawValue' or 'NormalizedValue'.
        
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
        
        # Normalize by dividing by the overall maximum for that metric.
        overall_max = max_series[metric].max()
        max_series['NormalizedValue'] = max_series[metric] / overall_max
        max_series['RawValue'] = max_series[metric]
        
        # Add a column to indicate which metric this row corresponds to.
        max_series['Metric'] = metric
        
        # Keep only the date, normalized value, and metric columns.
        normalized_list.append(max_series[[date_col, rawOrNormalized, 'Metric']])
    
    # Combine all metrics into one DataFrame in long format.
    combined_normalized = pd.concat(normalized_list, ignore_index=True)
    
    # Create the plot.
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_normalized, x=date_col, y=rawOrNormalized, hue='Metric', marker='o')
    plt.xlabel("Date")
    plt.ylabel(f"Metric {rawOrNormalized}")
    plt.title("Maximum Metrics HELM-LITE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_file)
    
    return combined_normalized

# Example usage:
# Assuming `combined_df` is your DataFrame with a "Date" column and multiple metric columns:
metrics_to_plot = ['Mean win rate', 'NarrativeQA - F1',
       'NaturalQuestions (open) - F1', 'NaturalQuestions (closed) - F1',
       'OpenbookQA - EM', 'MMLU - EM', 'MATH - Equivalent (CoT)', 'GSM8K - EM',
       'LegalBench - EM', 'MedQA - EM', 'WMT 2014 - BLEU-4']
normalized_data = plot_multiple_normalized(all_results, 
                                           date_col="Date", 
                                           metric_cols=metrics_to_plot,
                                           output_file="plots/HELM/raw_all_cols.png",
                                           rawOrNormalized='RawValue')


