import pandas as pd
import json
from functools import reduce

def read_helm_list(version='v1.11.0/', runs='/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/', instance='natural_qa:mode=closedbook,model=meta_llama-3.3-70b-instruct-turbo/'): 
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
        
        only_relevant_keys = [{key: record[key] for key in relevant_files[file]} for record in full_dict]

        df_temp = pd.json_normalize(only_relevant_keys)
        if file=='instances':
            df_temp = df_temp.rename(columns={'id': 'instance_id'})
        df_list.append(df_temp)
        
    df = reduce(
        lambda left, right: pd.merge(left, right, on="instance_id", how="outer"),
        df_list
    )
    
    return df