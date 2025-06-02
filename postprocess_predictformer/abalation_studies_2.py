import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import pandas as pd
from jarvis.utils.metrics import Metrics
from jarvis.utils import plot_config
from typing import List, Dict, Any, Tuple

def parse_dataframe(
    df: pd.DataFrame,
    n_agents: int = 3) -> Tuple:
    
    agent_names = [f'Agent {i}' for i in range(n_agents)]
    long_parts = []
    for i, name in enumerate(agent_names):
        suffix = '' if i==0 else f'.{i}'
        sub = df[
            ['Time (s)'+suffix,
            'Mean MSE'+suffix,
            'Standard Deviation'+suffix,
            'SEM'+suffix,
            'CI'+suffix]
        ].copy()
        sub.columns = ['Time','Mean MSE','SD','SEM','CI']
        sub['Agent'] = name
        long_parts.append(sub)

    # 4) concatenate
    long_df = pd.concat(long_parts, ignore_index=True)
    return long_df


def plot_mse_lines(
    short_model: pd.DataFrame,
    long_model: pd.DataFrame,
    title: str = "MSE Comparison",
    xlabel: str = "Time (s)",
    ylabel: str = "MSE",
    save_name: str = "mse_comparison.png"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    pass
    # plt.figure(figsize=(10, 6))
    # plt.plot(short_model['time'], short_model['mse'], label='Short Model', color='blue')
    # plt.plot(long_model['time'], long_model['mse'], label='Long Model', color='orange')
    
    # plt.title(title)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.legend()
    # plt.grid()
    
    # plt.savefig(save_name)
    # plt.show()
    
def plot_time_error(
    short_model: pd.DataFrame,
    long_model: pd.DataFrame,
    title: str = "Time Error Comparison",
    xlabel: str = "Projected Time (s)",
    ylabel: str = "Time Error (s)",
    save_name: str = "time_error_comparison.png"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), nrows=3, sharex=True)
    agent_counter = 0
    for agent_name, sub_df in short_model.groupby('Agent'):
        std_deviation = sub_df['SD'].values
        times = sub_df['Time'].values
        colors = ['b', 'r']
        for i, v in enumerate([45, 80]):
            sd_v = std_deviation/v
            ax[agent_counter].plot(times, sd_v, 
                       label=f'{agent_name} {v} m/s', color=colors[i])
            ax[agent_counter].legend()
            fig.supxlabel('Projected Time (s)')
            fig.supylabel('Time Error (s)')
        agent_counter += 1
    
    fig.tight_layout()
    # save as svg
    plt.savefig(folder_name + "time_error.svg")
    plt.savefig(folder_name + "time_error.png")    
    

if __name__ == '__main__':
    folder_name: str = "postprocess_predictformer/"
    csv_one = "highspeed_predictformer_output_mse.csv"
    csv_two = "highspeed_predictformer_output_120_mse.csv"
    
    short_model = pd.read_csv(folder_name + csv_one)
    long_model = pd.read_csv(folder_name + csv_two)
    short_model = parse_dataframe(short_model)
    long_model = parse_dataframe(long_model)
    print(short_model.columns)
    print(long_model.columns)

    short_model = short_model.drop(short_model.index[0])
    long_model = long_model.drop(long_model.index[0])
    print(short_model)
            
    # let's plot the time jitter to make an argument
    # that the large standard deviation and is also not a big deal
    # The idea is that even though we have large standard deviations
    # if we look at how fast our systems are moving its not that bad 
    # from the looks of it worst case scenario we are off by 1/4 second
    plot_time_error(short_model, long_model)
     
    # get every 5 rows
    row_desired = 5
    short_model = short_model.iloc[::row_desired, :]
    long_model = long_model.iloc[::row_desired, :]
    models = [short_model, long_model]
    
    for i, model in enumerate(models):
        for agent_name, sub_df in model.groupby('Agent'):
            if '0' in agent_name:
                print(agent_name)
            
    plt.show()                
        
