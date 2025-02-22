import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('imported libs')

save_dir = '/Users/salilgoyal/Stanford/LLM-auditing/chatbot-arena-data'

csvs = {}
for file in os.listdir(save_dir):
  if file.endswith('csv'):
    filepath = os.path.join(save_dir, file)
    csvs[file[:-4]] = pd.read_csv(filepath)
    
print('loaded data')

csvs = dict(sorted(csvs.items()))
    
# for now, remove leaderboards from 20240125 and before because they have different formats
list(csvs.keys()).index('leaderboard_table_20240125') # = 13

# replace '-' with nans
csvs = dict(list(csvs.items())[14:])
for key in csvs.keys():
  csvs[key].replace('-', np.nan, inplace=True)
  
for key, df in csvs.items():
    if 'Model' in df.columns:
        df['Model'] = df['Model'].str.lower().str.strip()

for key, df in csvs.items():
    # Convert the columns to float, errors='coerce' handles invalid values as NaN
    df['MMLU'] = pd.to_numeric(df['MMLU'], errors='coerce')
    df['MT-bench (score)'] = pd.to_numeric(df['MT-bench (score)'], errors='coerce')
    
print('created dataframe')
    
# CREATE PLOT #
###############

# List to store processed data
trends = []

# Convert "online" to a placeholder date
far_future_date = pd.Timestamp('2100-01-01')
def parse_knowledge_cutoff(value):
    if value == 'Online':
        return far_future_date  # or today if preferred
    elif value == "Unknown":
      return pd.NaT
    else:
        return pd.to_datetime(value, format='%Y/%m')

# Extract relevant data from dfs
for date, df in csvs.items():
    df = df.copy()
    # df['knowledge cutoff date'] = df['Knowledge cutoff date'].apply(parse_knowledge_cutoff)
    df['date'] = pd.to_datetime(date[-8:])
    df["model"] = df["key"]
    trends.append(df[["date", "model", "MT-bench (score)", "MMLU"]])
    

# Combine all data into a single DataFrame
trends_df = pd.concat(trends)

# Compute aggregate statistics
trends_df_grouped = (
    trends_df.groupby(["date"])
    .agg({
        "MMLU": ["max", "median"],  # Max and median MMLU
        "MT-bench (score)": ["mean", "median"],  # Mean and median MT-Bench
    })
    .reset_index()
)

# Ensure 'date' is in datetime format
trends_df_grouped["date"] = pd.to_datetime(trends_df_grouped["date"])

# Flatten column names after aggregation
trends_df_grouped.columns = ["date", "MMLU_max", "MMLU_median", "MT-bench_mean", "MT-bench_median"]

# Plot
plt.figure(figsize=(10, 5))

# Max MMLU
plt.plot(trends_df_grouped["date"], trends_df_grouped["MMLU_max"], marker="o", linestyle="-", label="Max MMLU Score")

# Median MMLU
plt.plot(trends_df_grouped["date"], trends_df_grouped["MMLU_median"], marker="s", linestyle="--", label="Median MMLU Score")

# Mean MT-Bench
plt.plot(trends_df_grouped["date"], trends_df_grouped["MT-bench_mean"], marker="d", linestyle="-.", label="Mean MT-Bench Score")

# Median MT-Bench
plt.plot(trends_df_grouped["date"], trends_df_grouped["MT-bench_median"], marker="^", linestyle=":", label="Median MT-Bench Score")

# Labels and title
plt.xlabel("Date")
plt.ylabel("Performance Score")
plt.title("MMLU and MT-Bench Performance Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Save the plot as a file
plt.savefig("performance_trends_plot.png", dpi=300, bbox_inches="tight")  # Saves as PNG

print("Saved plot")

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(trends_df_grouped["date"], trends_df_grouped["MMLU"], marker="o", linestyle="-", label="Max MMLU Score")

# # Labels and title
# plt.xlabel("Date")
# plt.ylabel("Max MMLU Score")
# plt.title("Max MMLU Score Over Time")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)

# # Save the plot as a file (change format as needed: 'png', 'pdf', 'svg', etc.)
# plt.savefig("max_mmlu_score_plot.png", dpi=300, bbox_inches="tight")  # Saves as PNG

# print('saved plot')