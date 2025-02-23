import re
import pandas as pd
import matplotlib.pyplot as plt
import os

records = []

for file in os.listdir("../HELM-LITE/HELM_lite_accuracy_csvs"):
    
    file_path = os.path.join("../HELM-LITE/HELM_lite_accuracy_csvs", file)
    
    # Parse the date from the filename; e.g. "v1.7.0 (2024-08-08)-Table 1.csv"
    match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", file_path)
    if not match:
        # If we can't parse a date, skip this file
        continue
    
    date_str = match.group(1)  # e.g. "2024-08-08"
    date = pd.to_datetime(date_str)
    
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Identify the row with the maximum "Mean win rate"
    max_idx = df["Mean win rate"].idxmax()
    max_rate = df.loc[max_idx, "Mean win rate"]
    max_model = df.loc[max_idx, "Model"]
    
    # Store the results
    records.append({
        "date": date,
        "max_mean_win_rate": max_rate,
        "model_with_max": max_model
    })

# Convert the collected records into a DataFrame
results_df = pd.DataFrame(records)

# Sort by date to ensure chronological plotting
results_df.sort_values(by="date", inplace=True)

# Plot the time series
plt.figure(figsize=(8, 5))
plt.plot(results_df["date"], results_df["max_mean_win_rate"], marker="o", linestyle="-")
plt.xlabel("Date")
plt.ylabel("Max Mean Win Rate")
plt.title("Maximum Mean Win Rate on HELM (avg. over all tasks)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("plots/HELM/max_mean_win_rate_over_time.png")

# Optional: Inspect which model had the maximum for each date
print(results_df)
