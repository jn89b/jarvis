import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
from jarvis.utils.metrics import Metrics

def load_model(pickle_filename:str) -> Metrics:
    return Metrics(pickle_filename)

if __name__ == '__main__':
    folder_name: str = "postprocess_predictformer/"
    pickle_filename:str = "high_speed_predictformer_output.pkl"
    full_file:str = folder_name + pickle_filename
    predictformer_metrics = load_model(full_file)
    
    # continous_mse: np.array = predictformer_metrics.continous_mse()
    
    overall_metrics = predictformer_metrics.predictformer_mse(
        slice_size=1)
    
    predictformer_metrics.plot_mse_metrics(overall_metrics=overall_metrics)
    predictformer_metrics.plot_mse_lines(overall_metrics=overall_metrics)
    predictformer_metrics.plot_mse_lines(overall_metrics=overall_metrics, 
                                         plot_subplots=True)
    plt.show()