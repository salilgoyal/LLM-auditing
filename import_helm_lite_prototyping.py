import marimo

__generated_with = "0.12.0"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import json
    from functools import reduce
    import os
    return json, os, pd, reduce


@app.cell
def _():
    from import_helm_lite import read_helm_list
    return (read_helm_list,)


@app.cell
def _():
    runs = '/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/'
    version = 'v1.11.0/'
    _instance = 'natural_qa:mode=closedbook,model=meta_llama-3.3-70b-instruct-turbo/'
    path = runs + version + _instance
    return path, runs, version


@app.cell
def _(json, path):
    with open(path + 'display_requests.json') as _json_file:
        d = json.load(_json_file)
    return (d,)


@app.cell
def _():
    relevant_files = {'display_predictions': ['instance_id', 'predicted_text', 'stats'],
     # 'display_requests': ['instance_id', 'prompt'],
     'instances': ['id', 'input', 'references']}
    return (relevant_files,)


@app.cell
def _():
    # pd.json_normalize(test)
    return


@app.cell
def _(json, path, pd, relevant_files):
    df_list = []
    for file in relevant_files.keys():
        with open(path + file + '.json') as _json_file:
            full_dict = json.load(_json_file)
        only_relevant_keys = [{key: record[key] for key in relevant_files[file]} for record in full_dict]
        df_temp = pd.json_normalize(only_relevant_keys)
        if file == 'instances':
            df_temp = df_temp.rename(columns={'id': 'instance_id'})
        df_list.append(df_temp)
    return df_list, df_temp, file, full_dict, only_relevant_keys


@app.cell
def _(df_list, pd, reduce):
    df = reduce(
        lambda left, right: pd.merge(left, right, on="instance_id", how="outer"),
        df_list
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df['references'].iloc[0:5].values
    return


@app.cell
def _(df):
    df['all_tags_correct'] = df['references'].apply(
        lambda refs: all(tag == 'correct' for ref in refs for tag in ref['tags'])
    )
    return


@app.cell
def _(df):
    df['all_tags_correct'].sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Checking other folders
        """
    )
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    runs_1 = '/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/'
    version_1 = 'v1.1.0/'
    _instance = 'math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,model=microsoft_phi-2'
    path_1 = runs_1 + version_1 + _instance + '/'
    return path_1, runs_1, version_1


@app.cell
def _(json, os, pd, runs_1, version_1):
    all_cols = []
    for _instance in os.listdir(runs_1 + version_1):
        if _instance == 'eval_cache':
            continue
        path_2 = runs_1 + version_1 + _instance + '/'
        with open(path_2 + 'display_predictions.json') as _json_file:
            d_1 = json.load(_json_file)
        df_1 = pd.json_normalize(d_1)
        if _instance == 'med_qa:model=mistralai_mistral-medium-2312':
            all_cols.append(df_1.columns[-2])
        else:
            all_cols.append(df_1.columns[-1])
    return all_cols, d_1, df_1, path_2


@app.cell
def _(all_cols, np):
    unique_cols = set(np.unique(np.array(all_cols)).tolist())
    unique_cols
    return (unique_cols,)


@app.cell
def _(json, os, pd, runs_1, unique_cols, version_1):
    for _instance in os.listdir(runs_1 + version_1):
        if _instance == 'eval_cache':
            continue
        path_3 = runs_1 + version_1 + _instance + '/'
        with open(path_3 + 'display_predictions.json') as _json_file:
            d_2 = json.load(_json_file)
        df_2 = pd.json_normalize(d_2)
        cols = set(df_2.columns)
        if len(cols.intersection(unique_cols)) > 1:
            print(cols.intersection(unique_cols), _instance)
    return cols, d_2, df_2, path_3


@app.cell
def _(json, path_3):
    with open(path_3 + 'display_predictions.json') as _json_file:
        d_3 = json.load(_json_file)
    return (d_3,)


@app.cell
def _(d_3, pd):
    pd.json_normalize(d_3)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

