import marimo

__generated_with = "0.12.0"
app = marimo.App()


@app.cell
def _():
    from import_helm_lite import read_helm_list
    from helper_functions import calculate_stats, num_words, rank_questions
    from functools import reduce
    import pandas as pd
    import matplotlib.pyplot as plt
    return (
        calculate_stats,
        num_words,
        pd,
        plt,
        rank_questions,
        read_helm_list,
        reduce,
    )


@app.cell
def _():
    # df = read_helm_list(version='v1.11.0/', 
    #                     instance='mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=meta_llama-3.3-70b-instruct-turbo/')
    # df2 = read_helm_list(version='v1.11.0/', 
    #                     instance='mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=upstage_solar-pro-241126/')
    # df3 = read_helm_list(version='v1.10.0/',
    #                      instance='mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=anthropic_claude-3-5-sonnet-20241022/')
    # df4 = read_helm_list(version='v1.10.0/',
    #                      instance='mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=google_gemini-1.5-flash-002/')
    # df5 = read_helm_list(version='v1.10.0/',
    #                      instance='mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=google_gemini-1.5-pro-002/')
    # df6 = read_helm_list(version='v1.10.0/',
    #                      instance='mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=openai_gpt-4o-2024-08-06/')
    return


@app.cell
def _():
    # dfs = [df, df2, df3, df4, df5, df6]
    return


@app.cell
def _():
    # ranked_questions = rank_questions(dfs, metric_name='stats.exact_match', desired_stat_name='num_words')
    return


@app.cell
def _():
    # ranked_questions.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Below is a dumb example of a plot because for MMLU all the model responses have 1 letter, so all the word counts (input text + output text) are the exact same""")
    return


@app.cell
def _():
    # # Create the plot:
    # plt.figure(figsize=(10, 6))
    # x_values = range(1, ranked_questions.shape[0] + 1)  # ranking positions

    # # Plot word counts for each model.
    # for i in range(len(dfs)):
    #     plt.plot(x_values, ranked_questions[f'num_words_{i}'], marker='o', label=f'Model {i}')

    # plt.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Attempt to read all files in all versions""")
    return


@app.cell
def _():
    import os
    return (os,)


@app.cell
def _():
    versions = [f'v1.{str(i)}.0/' for i in range(14)]
    return (versions,)


@app.cell
def _(os, versions):
    runs = '/nlp/scr4/nlp/crfm/yifanmai/helm-release/benchmark_output/runs/'
    os.listdir(runs + versions[0])
    return (runs,)


@app.cell
def _():
    metrics = {
        'stats.bleu_4': [],
        'stats.exact_match': [],
        'stats.math_equiv_chain_of_thought': [],
        'stats.f1_score': [],
        'stats.final_number_exact_match': [],
        'stats.quasi_exact_match': []
    }
    return (metrics,)


@app.cell
def _(metrics, os, rank_questions, read_helm_list, runs, versions):
    dfs_1 = []
    metric_to_dfs = metrics.copy()  # Create a copy to store the grouped dataframes

    # First, read and group all dataframes
    for version in versions[:2]:
        for instance in os.listdir(runs + version):
            if instance == 'eval_cache':
                continue
            df_1 = read_helm_list(version=version, instance=instance + '/')
            dfs_1.append(df_1)

            # Determine which metric this dataframe uses
            for metric in metric_to_dfs.keys():
                if metric in df_1.columns:
                    metric_to_dfs[metric].append(df_1)
                    break

    # Then create ranked questions for each metric group
    ranked_questions_by_metric = {}
    for metric, dfs in metric_to_dfs.items():
        if dfs:  # Only process if there are dataframes for this metric
            ranked_questions_by_metric[metric] = rank_questions(dfs, metric_name=metric, desired_stat_name='num_words')
            print(f"\nRanked questions for {metric}:")
            print(ranked_questions_by_metric[metric].head())
    return (
        df_1,
        dfs,
        dfs_1,
        instance,
        metric,
        metric_to_dfs,
        ranked_questions_by_metric,
        version,
    )


@app.cell
def _():
    # # Display the first few rows of ranked questions for each metric
    # for metric, ranked_df in ranked_questions_by_metric.items():
    #     print(f"\nRanked questions for {metric}:")
    #     print(ranked_df.head())
    return


@app.cell
def _(dfs_1):
    dfs_1[100]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
