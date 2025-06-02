import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
from jarvis.utils.metrics import Metrics
from jarvis.utils import plot_config

def load_model(pickle_filename:str,
               start_idx:int=21) -> Metrics:
    return Metrics(pickle_filename,
                   start_idx=start_idx)

if __name__ == '__main__':
    folder_name: str = "postprocess_predictformer/"
    pickle_filename:str = "highspeed_predictformer_output_1.pkl"
    predictformer_onetwenty = "high_speed_predictformer_output_120.pkl"
    # check if the file exists
    try:
        with open(folder_name + pickle_filename, 'rb') as f:
            pass
    except FileNotFoundError:
        print(f"File {folder_name + pickle_filename} not found.")
        exit(1)
        
    full_file:str = folder_name + pickle_filename
    predictformer_metrics = load_model(full_file)
    predictformer_2 = load_model(folder_name+predictformer_onetwenty,
                                 start_idx=120)
    
    overall_metrics = predictformer_metrics.predictformer_mse(
        slice_size=1)

    predictformer_metrics.plot_mse_metrics(overall_metrics=overall_metrics)
    predictformer_metrics.plot_mse_lines(overall_metrics=overall_metrics)
    predictformer_metrics.plot_mse_lines(overall_metrics=overall_metrics, 
                                         plot_subplots=True,
                                         to_save=True,
                                         save_name=folder_name + "highspeed_predictformer_output_mse",)
    
    overall_metrics_2 = predictformer_2.predictformer_mse(
        slice_size=1)
    predictformer_2.plot_mse_metrics(overall_metrics=overall_metrics)
    predictformer_2.plot_mse_lines(overall_metrics=overall_metrics_2, 
                                         plot_subplots=True)
    predictformer_2.save_mse_metrics(
        overall_metrics=overall_metrics_2,
        save_name=folder_name + "highspeed_predictformer_output_120_mse",
        to_save=True)
    
    predictformer_metrics.save_mse_metrics(
        overall_metrics=overall_metrics,
        save_name=folder_name + "highspeed_predictformer_output_mse",
        to_save=True)
    
    plt.show()