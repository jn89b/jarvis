import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D  


"""
Metrics comparison class to compare forecasting models
- Compare predictformer, ukf, and lstm on the same dataset

"""

class Metrics():
    def __init__(
        self,
        predictformer_pkl_filename:str,
        start_idx:int = 21,
        delta_t:float =0.1):
        self.predictformer_file = pkl.load(open(predictformer_pkl_filename, "rb"))
        self.start_idx:int = start_idx
        self.process_predictformer_file()
    
    def process_predictformer_file(self) -> None:
        self.center_gt_trajs:List[np.array] = self.predictformer_file["center_gt_trajs"]
        self.center_objects_world:List[np.array] = self.predictformer_file["center_objects_world"]
        self.predicted_probs: List[np.array] = [output['predicted_probability'] for output in self.predictformer_file["output"]]
        self.predicted_trajectories: List[np.array] = [output['predicted_ground_traj'] for output in self.predictformer_file["output"]]
        self.infer_time: List[float] = self.predictformer_file["infer_time"]
        self.total_steps = len(self.center_gt_trajs)
        self.num_agents = self.center_gt_trajs[0].shape[1]
        self.num_modes = self.predicted_probs[0].shape[1]
        self.overall_agents = []
        
        for i in range(self.num_agents):
            agent = {}
            overall_positions = []
            predicted_trajs = []
            predicted_modes = []
            x_history = []
            y_history = []
            z_history = []
            for j in range(self.total_steps):
                # the ground truth trajectory is a [num_agents, num_timesteps, num_attributes]
                self.center_objects_world[j] = self.center_objects_world[j].squeeze()
                self.predicted_trajectories[j] = self.predicted_trajectories[j].squeeze()
                self.predicted_probs[j] = self.predicted_probs[j].squeeze()
                self.center_gt_trajs[j] = self.center_gt_trajs[j].squeeze()
                x_gt = self.center_objects_world[j][i,:, 0]
                y_gt = self.center_objects_world[j][i,:, 1]
                z_gt = self.center_objects_world[j][i,:, 2]
                # the predicted trajectory is [num_agents, num_modes, num_timesteps, num_attributes]
                x_pred = self.predicted_trajectories[j][i,:,:, 0] + x_gt[self.start_idx]
                y_pred = self.predicted_trajectories[j][i,:,:, 1] + y_gt[self.start_idx]
                z_pred = self.predicted_trajectories[j][i,:,:, 2] + z_gt[self.start_idx]
                x_history.extend(x_gt)
                y_history.extend(y_gt)
                z_history.extend(z_gt)
                overall_positions.append([x_gt, y_gt, z_gt])
                predicted_trajs.append([x_pred, y_pred, z_pred])
                predicted_modes.append(self.predicted_probs[j][i])
                
            x_positions = np.array([pos[0] for pos in overall_positions])
            y_positions = np.array([pos[1] for pos in overall_positions])
            z_positions = np.array([pos[2] for pos in overall_positions])
            x_positions = x_positions.flatten()
            y_positions = y_positions.flatten()
            z_positions = z_positions.flatten()
            agent['position'] = [x_positions, y_positions, z_positions]
            agent['ground_truth'] = overall_positions
            agent['predicted_trajectory'] = predicted_trajs
            agent['predicted_probability'] = predicted_modes
            self.overall_agents.append(agent)
            
    def predictformer_mse(self,
                          slice_size:int=10,
                          to_plot:bool=False) -> List[float]:
        
        overall_metrics = []
        for i, agent in enumerate(self.overall_agents):
            predicted_trajectory = np.array(agent["predicted_trajectory"])
            ground_truth_trajectory = np.array(agent["ground_truth"])
            mse_total = 0.0
            slice_count = 0
            mse_metrics = {
                'overall_mse': [],
                'slice_mse': []
            }
            prev_mse: float = 0.0
            for j in range(len(predicted_trajectory)):        
                best_modes = np.argsort(agent['predicted_probability'][j])[-3:]
                current_predicted_trajectory = predicted_trajectory[j]
                lowest_mse = np.inf
                best_mode:int = 0
                for idx in best_modes:
                    selected_traj = current_predicted_trajectory[:, idx, :]
                    gt = ground_truth_trajectory[j][:, self.start_idx:]
                    mse = np.mean((selected_traj - gt) ** 2)
                    if mse < lowest_mse:
                        lowest_mse = mse
                        best_mode = idx
                        
                best_predicted_trajectory = current_predicted_trajectory[:, best_mode, :]
                
                # Slice ground truth from start_idx onward
                current_ground_truth_trajectory = ground_truth_trajectory[j][:, self.start_idx:]
                
                # Determine number of steps after start_idx
                num_steps = current_ground_truth_trajectory.shape[1]
                # Loop over time slices in steps of 10s
                mse_bins = []
                for step in range(0, num_steps, slice_size):
                    pred_slice = best_predicted_trajectory[:, step:step + slice_size]
                    gt_slice = current_ground_truth_trajectory[:, step:step + slice_size]
                    
                    # Compute the MSE for the slice
                    #slice_mse = np.mean((pred_slice - gt_slice) ** 2)
                    # compute mean absolute error
                    slice_mse = np.abs(pred_slice - gt_slice)
                    slice_mse = np.sum(slice_mse)/gt_slice.shape[1]
                    # slice_mse = (slice_mse)
                    mse_bins.append(slice_mse)
                    # print(f"Agent {j}, trajectory {i}, slice {step}:{step+slice_size} MSE = {slice_mse}")
                    
                    mse_total += slice_mse + prev_mse
                    slice_count += 1
                
                prev_mse = mse_bins[-1]
                mse_metrics['slice_mse'].append(mse_bins)

            # Compute overall average MSE across all slices for the agent
            overall_mse = mse_total / slice_count if slice_count > 0 else None
            if j == 0:
                print(f"Agent {j} Overall Average MSE = {overall_mse}")
            mse_metrics['overall_mse'] = overall_mse
            overall_metrics.append(mse_metrics)
            
        #compute the overall            
        # Compute overall average MSE across all slices for the agent
        overall_mse = mse_total / slice_count if slice_count > 0 else None
        if j == 0:
            print(f"Agent {j} Overall Average MSE = {overall_mse}")
            
        mse_metrics['overall_mse'] = overall_mse
        overall_metrics.append(mse_metrics)
        
        if to_plot:
            self.plot_mse_metrics(overall_metrics)
        
        return overall_metrics
            
            
    def plot_mse_metrics(self, overall_metrics,
                         to_save:bool=False,
                         save_name:str="",
                         to_break:bool=True,
                         x_min:float=1,
                         x_max:float=6) -> None:
        # Determine the number of bins from the first agent's data
        num_bins = len(np.mean(np.array(overall_metrics[0]['slice_mse']), axis=0))
        print(f"Number of bins: {num_bins}")
        # Create x positions linearly spaced from 1 to 6 seconds
        x = np.linspace(x_min, x_max, num=num_bins)

        # Determine the width for grouped bars
        num_agents = len(overall_metrics)
        width = 0.8 / num_agents

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_palette("Set1")
        colors = ["blue", "orange", "green"]
        for i, agent in enumerate(overall_metrics):
            if i == len(overall_metrics)-1 and to_break:
                break
            mse_bins = np.array(agent['slice_mse'])  # shape: (num_runs, num_bins)
            print(len(mse_bins))
            mean_mse = np.mean(mse_bins, axis=0)
            std_mse = np.std(mse_bins, axis=0)
            n = mse_bins.shape[0]
            sem = std_mse / np.sqrt(n)      # standard error of the mean
            ci = 1.96 * sem                 # 95% confidence interval

            # Offset each agent's bars so they appear side by side
            positions = x - 0.4 + width/2 + i * width
            ax.bar(positions, mean_mse, width=width, yerr=ci, capsize=5,
                label=f"Agent {i}", align='center', color=colors[i])

        ax.set_xlabel("Projected Time (s)", fontsize=14)
        ax.set_ylabel("Magnitude Distance MSE (m)", fontsize=14)
        ax.set_title("MSE Error Propagation for Predicted Trajectories", fontsize=16)
        ax.legend()
        plt.tight_layout()

        # save as an svg
        if to_save:
            filename:str = save_name+".svg"
            plt.savefig(filename)

        


        
        
        