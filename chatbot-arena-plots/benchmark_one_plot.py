import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# 1) Merge the two DataFrames by date
df = pd.merge(
    max_mmlu[['date', 'MMLU']],
    max_mt[['date', 'MT-bench (score)']],
    on='date',
    how='outer'
)

# 2) Normalize each metric by its maximum value
df['MMLU_norm'] = df['MMLU'] / df['MMLU'].max()
df['MT_norm']   = df['MT-bench (score)'] / df['MT-bench (score)'].max()

# 3) Reshape (melt) so both normalized columns appear in one "score" column
df_melted = df.melt(
    id_vars='date',
    value_vars=['MMLU_norm', 'MT_norm'],
    var_name='metric',
    value_name='normalized_score'
)

# 4) Plot both normalized MMLU & MT-bench on the same chart
sns.lineplot(data=df_melted, 
             x='date', 
             y='normalized_score', 
             hue='metric', 
             style='metric',
             markers=True,
             marker='o', alpha=0.7)
plt.title("Normalized MMLU and MT-bench Over Time")
plt.ylabel("Normalized Score")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/normalized_mmlu_mt.png')
