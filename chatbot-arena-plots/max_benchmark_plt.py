import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

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

# # Compute aggregate statistics
# trends_df_grouped = (
#     trends_df.groupby(["date"])
#     .agg({
#         "MMLU": ["max", "median"],  # Max and median MMLU
#         "MT-bench (score)": ["max", "median"],  # Max and median MT-Bench
#     })
#     .reset_index()
# )

# Compute the rows with the max MMLU and max MT-bench scores for each date
max_mmlu = trends_df.loc[trends_df.groupby("date")["MMLU"].idxmax()]
max_mt = trends_df.loc[trends_df.groupby("date")["MT-bench (score)"].idxmax()]

# Build a color mapping for all models that achieved a max score
unique_models = sorted(set(max_mmlu["model"]).union(max_mt["model"]))
# Use a colormap with enough colors (here tab10, change if you have more than 10 models)
colors = plt.cm.get_cmap("tab10", len(unique_models))
color_map = {model: colors(i) for i, model in enumerate(unique_models)}

# Create vertically stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot max MMLU scores on the top subplot
ax1.scatter(
    max_mmlu["date"],
    max_mmlu["MMLU"],
    c=[color_map[m] for m in max_mmlu["model"]],
    marker='o',
    linestyle='-',
    alpha=0.8
)
ax1.set_title("Max MMLU Scores over Time")
ax1.set_ylabel("Max MMLU Score")
ax1.grid(True)

# Plot max MT-bench scores on the bottom subplot
ax2.scatter(
    max_mt["date"],
    max_mt["MT-bench (score)"],
    c=[color_map[m] for m in max_mt["model"]],
    marker='o',
    linestyle='-',
    alpha=0.8
)
ax2.set_title("Max MT-bench Scores over Time")
ax2.set_ylabel("Max MT-bench Score")
ax2.grid(True)

# Format the x-axis to display dates nicely
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate()

# Create a combined legend on the side
# Here we get handles and labels from one axis since the model set is common to both plots
handles = []
labels = []
for model in unique_models:
    handles.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color=color_map[model],
            linestyle='',
            markersize=8
        )
    )
    labels.append(model)

fig.legend(handles, labels, title="Model", loc="center right", bbox_to_anchor=(1, 0.5))


plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig("max_mmlu_mtbench_subplots_with_colors.png", dpi=300, bbox_inches="tight")
print('saved plot')


# # Ensure 'date' is in datetime format
# trends_df_grouped["date"] = pd.to_datetime(trends_df_grouped["date"])

# # Flatten column names after aggregation
# trends_df_grouped.columns = ["date", "MMLU_max", "MMLU_median", "MT-bench_max", "MT-bench_median"]

# # Find the model that achieved the max MMLU/MT-Bench score for each date
# max_mmlu_models = trends_df.loc[trends_df.groupby("date")["MMLU"].idxmax(), ["date", "model", "MMLU"]]
# max_mtbench_models = trends_df.loc[trends_df.groupby("date")["MT-bench (score)"].idxmax(), ["date", "model", "MT-bench (score)"]]

# Assign a unique color to each model
# model_colors = {model: color for model, color in zip(max_mmlu_models["model"].unique(), sns.color_palette("husl", n_colors=len(max_mmlu_models["model"].unique())))}

########
# Plot #
########

# # Assign unique colors to models
# all_models = pd.concat([max_mmlu_models["model"], max_mtbench_models["model"]]).unique()
# model_colors = {model: color for model, color in zip(all_models, sns.color_palette("husl", n_colors=len(all_models)))}

# # Create subplots with shared x-axis
# fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={'hspace': 0.3})

# ## ---------------------- PLOT 1: MMLU PERFORMANCE ----------------------
# ax1.set_title("Max MMLU Score Over Time")
# ax1.plot(max_mmlu_models["date"], max_mmlu_models["MMLU"], 
#             color=[model_colors[m] for m in max_mmlu_models["model"]], label="Max MMLU Score", s=80, alpha=0.8)
# ax1.set_ylabel("MMLU Score")
# ax1.grid(True)

# ## ---------------------- PLOT 2: MT-BENCH PERFORMANCE ----------------------
# ax2.set_title("Max MT-Bench Score Over Time")
# ax2.plot(max_mtbench_models["date"], max_mtbench_models["MT-bench (score)"], 
#             color=[model_colors[m] for m in max_mtbench_models["model"]], label="Max MT-Bench Score", marker="s", s=80, alpha=0.8)
# ax2.set_ylabel("MT-Bench Score")
# ax2.set_xlabel("Date")
# ax2.grid(True)



# # Create a combined legend for models (placing it outside)
# handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[model], markersize=8, label=model) for model in model_colors]
# fig.legend(handles=handles, title="Models", loc="upper right", bbox_to_anchor=(1.15, 1))

# plt.savefig("max_mmlu_mtbench_subplots_with_colors.png", dpi=300, bbox_inches="tight")
# print('saved plot')

####### PREVIOUS WITHOUT COLORS ########

# # Create subplots with shared x-axis
# fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={'hspace': 0.3})

# ## ---------------------- PLOT 1: MMLU PERFORMANCE ----------------------
# ax1.set_title("Max MMLU Score Over Time")
# ax1.plot(trends_df_grouped["date"], trends_df_grouped["MMLU_max"], marker="o", linestyle="-", linewidth=1,
#             label="Max MMLU Score", alpha=0.8)
# ax1.set_ylabel("MMLU Score")
# ax1.grid(True)
# ## ---------------------- PLOT 2: MT-BENCH PERFORMANCE ----------------------
# ax2.set_title("Max MT-Bench Score Over Time")
# ax2.plot(trends_df_grouped["date"], trends_df_grouped["MT-bench_max"], 
#             label="Max MT-Bench Score", marker="s", alpha=0.8)
# ax2.set_ylabel("MT-Bench Score")
# ax2.set_xlabel("Date")
# ax2.grid(True)

# plt.legend()
# # Save the plot
# plt.savefig("max_mmlu_mtbench_subplots_with_colors.png", dpi=300, bbox_inches="tight")

# print('saved plot')

# plt.figure(figsize=(10, 5))
# # Max MMLU
# plt.plot(trends_df_grouped["date"], trends_df_grouped["MMLU_max"], marker="o", linestyle="-", label="Max MMLU Score")
# # Median MMLU
# plt.plot(trends_df_grouped["date"], trends_df_grouped["MMLU_median"], marker="s", linestyle="--", label="Median MMLU Score")
# # Labels and title
# plt.xlabel("Date")
# plt.ylabel("Performance Score")
# plt.title("MMLU Performance Over Time")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# # Save the plot as a file
# plt.savefig("MMLU_trends.png", dpi=300, bbox_inches="tight")  # Saves as PNG

# plt.close()

# plt.figure(figsize=(10, 5))
# # Mean MT-Bench
# plt.plot(trends_df_grouped["date"], trends_df_grouped["MT-bench_max"], marker="d", linestyle="-.", label="Max MT-Bench Score")
# # Median MT-Bench
# plt.plot(trends_df_grouped["date"], trends_df_grouped["MT-bench_median"], marker="^", linestyle=":", label="Median MT-Bench Score")
# # Labels and title
# plt.xlabel("Date")
# plt.ylabel("Performance Score")
# plt.title("MT-Bench Performance Over Time")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# # Save the plot as a file
# plt.savefig("MT-Bench_trends.png", dpi=300, bbox_inches="tight")  # Saves as PNG

# print("Saved plots")