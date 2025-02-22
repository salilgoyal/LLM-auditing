import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
import os
import numpy as np

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


####################   
# CREATE DATAFRAME #
####################

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
trends_df = pd.concat(trends).reset_index()

# Convert timestamps to just the date portion:
trends_df["date"] = trends_df["date"].dt.normalize()

# Compute the rows with the max MMLU and max MT-bench scores for each date
max_mmlu = trends_df.loc[trends_df.groupby("date")["MMLU"].idxmax()]
max_mt = trends_df.loc[trends_df.groupby("date")["MT-bench (score)"].idxmax()]

# ----------------------------------------------------------------
# 1) Prepare DataFrames: assume max_mmlu and max_mt are computed
# as one row per date for each metric.
# For example, using:
# max_mmlu = trends_df.sort_values(["date", "MMLU"], ascending=[True, False])\
#              .drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
# max_mt   = trends_df.sort_values(["date", "MT-bench (score)"], ascending=[True, False])\
#              .drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
# ----------------------------------------------------------------

# Keep only the relevant columns and rename the score columns to a common name
df_mmlu = max_mmlu[['date', 'model', 'MMLU']].copy()
df_mmlu = df_mmlu.rename(columns={'MMLU': 'score'})
df_mmlu['metric'] = 'MMLU'

df_mt = max_mt[['date', 'model', 'MT-bench (score)']].copy()
df_mt = df_mt.rename(columns={'MT-bench (score)': 'score'})
df_mt['metric'] = 'MT-bench'

# Combine the two dataframes into one long-form dataframe
combined = pd.concat([df_mmlu, df_mt], ignore_index=True)

# ----------------------------------------------------------------
# 2) Build a color palette mapping for the models
# ----------------------------------------------------------------
unique_models = sorted(set(df_mmlu["model"]).union(df_mt["model"]))
# Use a colormap (e.g., tab10) to get a unique color per model
colors = plt.cm.get_cmap("tab10", len(unique_models))
color_map = {model: colors(i) for i, model in enumerate(unique_models)}
# Convert to hex strings (palette for seaborn)
palette = {model: mcolors.to_hex(color_map[model]) for model in unique_models}

# ----------------------------------------------------------------
# 3) Create a FacetGrid with one row per metric
# ----------------------------------------------------------------
g = sns.FacetGrid(
    combined,
    row="metric",
    sharex=True,
    height=4,
    aspect=3,
    hue="model",
    palette=palette,
    legend_out=True
)

g.map_dataframe(sns.scatterplot, x="date", y="score", s=50)

# Set axis labels and titles for each facet
g.set_axis_labels("Date", "Score")
g.set_titles("{row_name}")

# Format x-axis tick labels to show dates nicely
for ax in g.axes.flatten():
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True)
g.fig.autofmt_xdate()

# ----------------------------------------------------------------
# 4) Add a combined legend
# ----------------------------------------------------------------
g.add_legend(title="Model", bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig("max_mmlu_mtbench_subplots_with_colors_sns.png", dpi=300, bbox_inches="tight")
print('saved plot')
