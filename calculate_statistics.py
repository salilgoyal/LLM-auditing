import pandas as pd
from functools import reduce

def calculate_stats(df, metric, metric_name):
    # assumes that revelant column names are input.text and predicted_text
    # metric should be a function that takes in input text and predicted text
    df[metric_name] = df.apply(lambda x: metric(x['input.text'], x['predicted_text']), axis=1)
    return df

# example metric
def num_words(input_text, output_text):
    # dumb metric, returns number of words in input and output strings
    return len(input_text.split()) + len(output_text.split())

# example usage
# df2 = calculate_stats(df, num_words, 'num_words')

def rank_questions(dfs, metric_name='stats.exact_match', desired_stat_name='num_words'):
    '''
    takes in a list of dfs
    output df has rows corresponding to instance_id (different questions) and cols predicted_text, exact_match and std'''
    processed_dfs = []

    for i, df in enumerate(dfs):
        if i == 0:
            df = calculate_stats(df, num_words, desired_stat_name)
            df_subset = df[['instance_id', 'input.text', 'predicted_text', metric_name, desired_stat_name]].copy()
        else:
            df = calculate_stats(df, num_words, desired_stat_name)
            df_subset = df[['instance_id', 'predicted_text', metric_name, desired_stat_name]].copy()
        df_subset = df_subset.rename(columns={
            'predicted_text': f'predicted_text_{i}',
            metric_name: metric_name + f'_{i}',
            desired_stat_name: desired_stat_name + f'_{i}'
        })
        processed_dfs.append(df_subset)
        
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='instance_id'), processed_dfs)
    
    # Compute standard deviation across the metric columns for each question.
    metric_cols = [col for col in merged_df.columns if col.startswith(metric_name)]
    merged_df['std'] = merged_df[metric_cols].std(axis=1)
    
    ranked_questions = merged_df.sort_values(by='std')
    
    return ranked_questions