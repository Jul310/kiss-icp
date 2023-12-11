import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from typing import Dict

from tools.internal.plot_settings import get_figsize

from enum import Enum
from evo.tools import file_interface, plot, pandas_bridge
from evo.tools.settings import SETTINGS

class TestSets(Enum):
    NO_KISS = "01_no_kiss", "EKF",
    WITH_KISS = "02_with_kiss", "EKF with KISS-ICP",
    # OUTAGE_STRAIGHT = "04_straight_outage", "Outage Scenario Straight",
    OUTAGE_TURN = "05_turn_outage", "Outage (scenario 'Turn')",
    

def create_df(results: Dict, prefix=""):
    df = pd.DataFrame()
    prefix = f"{prefix}_" if prefix else ""
    for test_set in TestSets:
        test_name, test_id = test_set.value
        df = pd.concat([df, pandas_bridge.result_to_df(results[f"{prefix}{test_name}_tranlsation"], test_id)], axis="columns")
    return df

def create_tidy_df(results, prefix="", label=""):
    df = create_df(results, prefix=prefix)
    keys = df.columns.values.tolist()    
    error_df = pd.DataFrame(df.loc["np_arrays", "error_array"].tolist(), index=keys).T
    return pd.melt(error_df, value_vars=list(error_df.columns.values), var_name="estimate", value_name=label)


def plot_line_chart(results, prefix=""):
    df = create_df(results, prefix)
    keys = df.columns.values.tolist()    
    error_df = pd.DataFrame(df.loc["np_arrays", "error_array"].tolist(),
                                    index=keys).T

    fig = plt.figure(figsize=get_figsize(wf = 2, hf = 1))
    # handle NaNs from concat() above
    error_df.interpolate(method="index", limit_area="inside").plot(
        ax=fig.gca(), title="", alpha=SETTINGS.plot_trajectory_alpha)
    # plt.xlabel(index_label)
    # plt.ylabel(metric_label)
    plt.legend(frameon=True)
    plt.show()
    
def plot_boxchart(results, xlabel="", prefix=""):
    fig = plt.figure(figsize=get_figsize(wf=1, hf=1))

    df = create_df(results, prefix=prefix)

    include = df.loc["stats"].index.isin(["std", "median", "mean", "rmse"])
    if any(include):
        df.loc["stats"][include].plot(kind="barh", ax=fig.gca(), stacked=False)
        plt.xlabel(xlabel)
        plt.legend(frameon=True)
        
    plt.show()
    

    
def plot_boxplot(results, prefix="", label=""):
    fig = plt.figure(figsize=get_figsize(wf=1, hf=1))
    
    raw_tidy = create_tidy_df(results, prefix=prefix, label=label)

    for test_set in TestSets:
        test_name, test_id = test_set.value
        ax = sns.boxplot(x=raw_tidy["estimate"], y=raw_tidy[label], ax=fig.gca())
        ax.set_xticklabels(labels=[item.get_text() for item in ax.get_xticklabels()], rotation=30)
    plt.show()

def plot_violin(results, prefix="", label=""):
    fig = plt.figure(figsize=get_figsize(wf=1, hf=1))

    # fig = plt.figure()
    raw_tidy = create_tidy_df(results, prefix=prefix, label=label)
    
    ax = sns.violinplot(x=raw_tidy["estimate"], y=raw_tidy[label], ax=fig.gca(),
                        legend='full', hue=raw_tidy['estimate'])
    # ax.set_xticklabels(labels=[item.get_text() for item in ax.get_xticklabels()], rotation=30)
    ax.set_xticklabels(labels=["1", "2", "3"])
    # plt.legend().set_title("")
    plt.show()
    

def plot_hist(results, prefix="", label=""):
    df = create_df(results, prefix=prefix)
    keys = df.columns.values.tolist()    
    error_df = pd.DataFrame(df.loc["np_arrays", "error_array"].tolist(), index=keys).T
    raw_tidy = pd.melt(error_df, value_vars=list(error_df.columns.values), var_name="estimate", value_name=label)
    raw_tidy = raw_tidy.dropna(subset=['APE'])
    
    g = sns.displot(data=raw_tidy, x=label, col='estimate',kde=True, stat='density')
    g.set_titles(col_template="{col_name}")

    # sns.histplot(data=raw_tidy, x=label, hue='estimate',kde=True, stat='density')

    plt.show()
    
def plot_ecdf(results, prefix="", label=""):
    # plt.figure(figsize=get_figsize(wf=1, hf=1))
    w,h = get_figsize(wf=1, hf=1)
    df = create_df(results, prefix=prefix)
    keys = df.columns.values.tolist()    
    error_df = pd.DataFrame(df.loc["np_arrays", "error_array"].tolist(), index=keys).T
    raw_tidy = pd.melt(error_df, value_vars=list(error_df.columns.values), var_name="estimate", value_name=label)
    raw_tidy = raw_tidy.dropna(subset=['APE'])
    
    # sns.displot(data=raw_tidy, x=label, col='estimate',kde=True, stat='density')
    g = sns.displot(data=raw_tidy, x=label, hue='estimate', kind='ecdf', 
                    stat='percent', legend=True, facet_kws=dict(legend_out=False),
                    height=h, aspect=w/h)
    # sns.move_legend(g, "lower right")
    g._legend.set_title("")

    # sns.histplot(data=raw_tidy, x=label, hue='estimate',kde=True, stat='density')

    plt.show()