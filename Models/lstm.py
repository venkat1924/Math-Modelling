# -*- coding: utf-8 -*-
"""
Inverse ODE Parameter Recovery using a Neural Network

This script trains a model (CNN or LSTM) to predict the 13 parameters
of a prey-predator ODE system given a time series trajectory.

Key features:
- CNN and LSTM model options with command-line selection.
- Parameter clipping for predicted parameters during ODE simulation for stability.
- Robust handling of ODE simulation failures with penalties during validation.
- Correct data scaling practices (scaler fitted only on training data).
- Comprehensive training loop with checkpointing, early stopping, and metric tracking.
"""

import os
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# Import core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.integrate import solve_ivp

# Suppress specific UserWarnings from LSODA and general FutureWarning
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.integrate._ivp.lsoda')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Global Constants ---
PARAM_NAMES = ['r', 'p1', 'p2', 'd1', 'b', 'c1', 'a', 'm',
               'h1', 'd2', 'd3', 'lambda1', 'q']
NUM_PARAMS = len(PARAM_NAMES)
STATE_VARIABLES = ['X', 'Y', 'Z']
NUM_STATE_VARIABLES = len(STATE_VARIABLES)

PARAMETER_RANGES = { # Based on original simulation script's parameter_grid
    'r': [0.1, 2.0], 'p1': [0.0, 0.4], 'p2': [0.1, 0.3],
    'd1': [0.05, 0.1], 'b': [0.01, 0.05], 'c1': [0.1, 0.2],
    'a': [0.5, 1.0], 'm': [0.0, 0.1], 'h1': [0.2, 0.3],
    'd2': [0.05, 0.1], 'd3': [0.02, 0.05], 'lambda1': [0.5, 1.0],
    'q': [0.5, 1.0]
}
PARAMETER_BOUNDS_MIN = np.array([PARAMETER_RANGES[name][0] for name in PARAM_NAMES], dtype=np.float32)
PARAMETER_BOUNDS_MAX = np.array([PARAMETER_RANGES[name][1] for name in PARAM_NAMES], dtype=np.float32)
DEFAULT_TRAJ_FAIL_PENALTY = 1e6 # Arbitrary high penalty

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True # For reproducibility
    torch.backends.cudnn.benchmark = False   # For reproducibility

# --- ODE Simulation Utilities ---
def ode_system(t: float, state: np.ndarray, params_dict: Dict[str, float]) -> List[float]:
    """Defines the ODE system dynamics."""
    X, Y, Z = state
    p = params_dict # Alias for brevity

    # Denominators with safeguards
    denominator1 = max(1 + p['p1']*Y + p['p2']*Z, 1e-12)
    denominator2 = max(p['a'] + (1 - p['m'])*X, 1e-12)
    one_minus_m = 1 - p['m']

    # Safeguard for extreme state values if solver somehow lets them explode
    # This is a heuristic; ideal scenario is solver stability via good parameters
    if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
        return [1e12] * NUM_STATE_VARIABLES # Signal severe instability

    # Heuristic cap to prevent overflow if states grow excessively large
    # This might be hit if predicted parameters are valid but lead to explosions.
    cap_val = 1e9
    if X > cap_val or Y > cap_val or Z > cap_val:
        # Attempt to return large "corrective" derivatives
        return [-(X-cap_val)*1e3 if X > cap_val else 0.0, # Use float
                -(Y-cap_val)*1e3 if Y > cap_val else 0.0,
                -(Z-cap_val)*1e3 if Z > cap_val else 0.0]
    try:
        dXdt = (p['r'] * X / denominator1) \
               - p['d1']*X \
               - p['b']*X**2 \
               - (p['c1']*one_minus_m*X*(Y + p['q']*Z) / denominator2)
        dYdt = (p['h1']*one_minus_m*X*(Y + p['q']*Z) / denominator2) \
               - p['d2']*Y \
               - p['lambda1']*Y*Z
        dZdt = p['lambda1']*Y*Z \
               - (p['d2'] + p['d3'])*Z
    except OverflowError: # Catch overflow during calculation of derivatives
        return [1e12] * NUM_STATE_VARIABLES # Signal severe instability
    return [dXdt, dYdt, dZdt]

def integrate_ode(t_span: Tuple[float, float], y0: np.ndarray,
                  params_values: np.ndarray, t_eval: Optional[np.ndarray] = None,
                  param_names: List[str] = PARAM_NAMES, apply_bounds: bool = True
                  ) -> Optional[pd.DataFrame]:
    """Integrates the ODE system, optionally clipping parameters to known bounds."""
    if len(params_values) != len(param_names):
        raise ValueError(f"Length mismatch: {len(params_values)} params vs {len(param_names)} names.")

    params_values_to_use = np.clip(params_values, PARAMETER_BOUNDS_MIN, PARAMETER_BOUNDS_MAX) if apply_bounds else params_values
    params_dict = dict(zip(param_names, params_values_to_use))

    try:
        sol = solve_ivp(
            fun=lambda t, y_state: ode_system(t, y_state, params_dict), # y_state to avoid conflict
            t_span=t_span, y0=y0, method='LSODA', t_eval=t_eval,
            rtol=1e-6, atol=1e-8 # Standard tolerances
        )
        if sol.success:
            df_res = pd.DataFrame({'t': sol.t, 'X': sol.y[0], 'Y': sol.y[1], 'Z': sol.y[2]})
            df_res[STATE_VARIABLES] = df_res[STATE_VARIABLES].clip(lower=0) # Ensure non-negativity
            return df_res
        # Optional: log sol.message if not successful
        return None
    except Exception: # Catch any other integration error (e.g. from ode_system heuristics)
        # Optional: log the exception e
        return None

# --- Dataset Class ---
class ODESimulationDataset(Dataset):
    """PyTorch Dataset for loading, preprocessing, and serving ODE simulation data."""
    def __init__(self, csv_path: str, param_names: List[str], state_variables: List[str],
                 param_scaler: Optional[MinMaxScaler] = None, fit_scaler_on_this_data: bool = False, # Renamed for clarity
                 use_noisy_data: bool = True):
        super().__init__()
        self.param_names = param_names
        self.state_variables = state_variables
        self.use_noisy_data = use_noisy_data
        self.input_cols = [f"{v}_noisy" if use_noisy_data else v for v in state_variables]

        print(f"Loading data from {csv_path}...")
        try:
            self.data_df = pd.read_csv(csv_path)
        except FileNotFoundError: raise FileNotFoundError(f"Data file not found: {csv_path}")
        except Exception as e: raise RuntimeError(f"Error loading CSV {csv_path}: {e}")

        required_cols = ['run_id', 't'] + self.input_cols + self.param_names
        missing_cols = [col for col in required_cols if col not in self.data_df.columns]
        if missing_cols: raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        print("Processing and grouping data by run_id...")
        self.trajectories, self.parameters = [], []
        self.run_ids = self.data_df['run_id'].unique()
        grouped_by_run = self.data_df.groupby('run_id')
        for run_id_val in self.run_ids: # run_id_val to avoid conflict
            run_data = grouped_by_run.get_group(run_id_val)
            self.trajectories.append(run_data[self.input_cols].values)
            self.parameters.append(run_data[self.param_names].iloc[0].values) # Params are same for all rows of a run

        self.parameters = np.array(self.parameters, dtype=np.float32)
        self.trajectories = np.array(self.trajectories, dtype=np.float32)

        print(f"Found {len(self.run_ids)} unique runs.")
        if self.trajectories.shape[0] == 0: raise ValueError("No valid runs found in the dataset.")
        self.num_timesteps = self.trajectories.shape[1]
        self.num_variables = self.trajectories.shape[2]
        print(f"Trajectory shape: (Runs, Timesteps, Variables) = {self.trajectories.shape}")

        # Parameter Scaling (MinMax to [0,1] typically)
        if fit_scaler_on_this_data:
            if param_scaler is not None:
                print("Warning: fit_scaler_on_this_data=True but a param_scaler was provided. Using the provided scaler.")
            else:
                print("Fitting new MinMaxScaler on parameters of this dataset instance...")
                self.param_scaler = MinMaxScaler(feature_range=(0, 1)) # Default [0,1]
                self.scaled_parameters = self.param_scaler.fit_transform(self.parameters)
        elif param_scaler is not None:
            print("Using provided pre-fitted parameter scaler...")
            self.param_scaler = param_scaler
            self.scaled_parameters = self.param_scaler.transform(self.parameters)
        else: # No scaler provided, and not fitting one on this instance
            print("Warning: No parameter scaling applied during this ODESimulationDataset initialization.")
            self.param_scaler = None # Explicitly set
            self.scaled_parameters = self.parameters # Use original parameters if no scaling

        # Trajectory Normalization (StandardScaler: zero mean, unit variance per trajectory)
        print("Normalizing trajectories (standardization per run)...")
        self.trajectory_scalers = [StandardScaler() for _ in range(len(self.trajectories))]
        self.normalized_trajectories = np.array([
            scaler.fit_transform(traj) for scaler, traj in zip(self.trajectory_scalers, self.trajectories)
        ], dtype=np.float32)

        # Reshape for PyTorch Conv1D/LSTM: (Batch, Channels/Features, SequenceLength)
        self.normalized_trajectories = np.transpose(self.normalized_trajectories, (0, 2, 1))
        print(f"Final trajectory tensor shape for model: {self.normalized_trajectories.shape}")

    def __len__(self) -> int: return len(self.run_ids)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # scaled_parameters should have been set during __init__
        return torch.from_numpy(self.normalized_trajectories[idx]), torch.from_numpy(self.scaled_parameters[idx])
    def get_original_params(self, idx: int) -> np.ndarray: return self.parameters[idx]
    def get_run_id(self, idx: int) -> int: return self.run_ids[idx]

# --- Model Architectures ---
class ParameterPredictorCNN(nn.Module):
    """1D CNN for parameter prediction from time series."""
    def __init__(self, num_params: int = NUM_PARAMS,
                 num_state_variables: int = NUM_STATE_VARIABLES,
                 num_timesteps: int = 1000): # num_timesteps needed for FC layer calculation
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_state_variables, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), # L/2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), # L/4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), # L/8
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2), # L/16
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)  # L/32
        )
        conv_output_length = num_timesteps // (2**5)
        flat_features = 512 * conv_output_length
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_params)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class ParameterPredictorLSTM(nn.Module):
    """LSTM based model for parameter prediction from time series."""
    def __init__(self, num_params: int = NUM_PARAMS,
                 input_features: int = NUM_STATE_VARIABLES, # Num channels from trajectory
                 hidden_size: int = 256, num_layers: int = 2, # LSTM specific hyperparams
                 lstm_dropout_rate: float = 0.2, fc_dropout_rate: float = 0.5, # Dropout rates
                 bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, # Critical: input format (batch, seq, feature)
            dropout=lstm_dropout_rate if num_layers > 1 else 0, # Dropout between LSTM layers
            bidirectional=bidirectional
        )
        # Determine FC layer input size based on LSTM output (last hidden state)
        fc_input_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_features, fc_input_features // 2), nn.ReLU(),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(fc_input_features // 2, num_params)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x from DataLoader: (batch_size, num_input_channels, seq_len)
        # LSTM expects: (batch_size, seq_len, num_input_features)
        x = x.permute(0, 2, 1)

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Use the hidden state from the last layer
        if self.lstm.bidirectional:
            # Concatenate the final hidden states of the forward and backward LSTMs
            # h_n[-2] is the last forward hidden state, h_n[-1] is the last backward.
            last_hidden_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            # h_n[-1] is the last layer's hidden state.
            last_hidden_state = h_n[-1, :, :]

        params_pred = self.fc_layers(last_hidden_state)
        return params_pred

# --- Training & Validation Functions ---
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Performs one epoch of training."""
    model.train()
    total_loss = 0.0
    scaler = GradScaler()
    for inputs, targets in dataloader:
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()

      # 1) cast to mixed‑precision for forward
      predictions = model(inputs)
      loss = criterion(predictions, targets)

      # 2) scale, backward, step, update
      scaler.scale(loss).backward()     # scales loss → avoids underflow
      scaler.step(optimizer)            # un-scales grads → optimizer.step()
      scaler.update()                   # adjusts scale for next iteration

      total_loss += loss.item()
    return total_loss / len(dataloader) # Average loss per batch

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   param_scaler: MinMaxScaler, device: torch.device,
                   full_dataset_ref: Optional[ODESimulationDataset] = None, # Reference to full dataset
                   num_trajectory_comps: int = 0,
                   t_span_eval: Optional[Tuple[float, float]] = None,
                   t_eval_points: Optional[np.ndarray] = None,
                   trajectory_fail_penalty: float = DEFAULT_TRAJ_FAIL_PENALTY
                   ) -> Tuple[float, Dict[str, float]]: # Returns (primary_loss, metrics_dict)
    """Performs one epoch of validation."""
    model.eval()
    total_scaled_loss = 0.0         # Primary loss for early stopping
    total_unscaled_mae = 0.0        # MAE on original parameter scales
    sum_trajectory_mse = 0.0        # Sum of MSEs or penalties for trajectories
    num_trajectory_comps_attempted = 0 # Denominator for average trajectory MSE
    num_samples_processed = 0

    val_metrics_dict = {} # To store all computed metrics

    with torch.no_grad():
      with autocast():
        for batch_idx, (inputs, targets_scaled) in enumerate(dataloader):
            current_batch_size = inputs.size(0)
            num_samples_processed += current_batch_size
            inputs, targets_scaled = inputs.to(device), targets_scaled.to(device)

            predictions_scaled = model(inputs)
            scaled_loss_on_batch = criterion(predictions_scaled, targets_scaled)
            total_scaled_loss += scaled_loss_on_batch.item() * current_batch_size # Weighted by batch size

            # --- Unscaled Metrics and Trajectory Comparison ---
            try:
                # Determine original indices from the sampler used by DataLoader
                original_indices_in_batch = []
                if hasattr(dataloader.sampler, 'indices'): # True for SubsetRandomSampler
                    sampler_start_idx = batch_idx * dataloader.batch_size
                    sampler_end_idx = sampler_start_idx + current_batch_size
                    original_indices_in_batch = dataloader.sampler.indices[sampler_start_idx:sampler_end_idx]
                else: # Fallback for sequential or unknown sampler (less common for train/val split)
                     processed_so_far = batch_idx * dataloader.batch_size
                     original_indices_in_batch = list(range(processed_so_far, processed_so_far + current_batch_size))

                # Unscale predictions for MAE and trajectory simulation
                preds_unscaled_np = param_scaler.inverse_transform(predictions_scaled.cpu().numpy())

                if full_dataset_ref:
                    targets_unscaled_np = np.array([full_dataset_ref.get_original_params(i) for i in original_indices_in_batch])
                    unscaled_mae_on_batch = np.mean(np.abs(preds_unscaled_np - targets_unscaled_np))
                    total_unscaled_mae += unscaled_mae_on_batch * current_batch_size
                else: # Should ideally always have full_dataset_ref if MAE unscaled is desired
                    total_unscaled_mae += np.nan * current_batch_size

                # Trajectory Comparison (if enabled and quota not met)
                if full_dataset_ref and num_trajectory_comps > 0 and \
                   num_trajectory_comps_attempted < num_trajectory_comps:
                    if t_span_eval is None or t_eval_points is None:
                        # This warning should ideally appear only once
                        if batch_idx == 0: print("Warning: Skipping trajectory comparison, t_span_eval or t_eval_points not provided.")
                    else:
                        comps_to_attempt_in_this_batch = min(current_batch_size, num_trajectory_comps - num_trajectory_comps_attempted)
                        for i in range(comps_to_attempt_in_this_batch):
                            original_data_idx = original_indices_in_batch[i]
                            # Use original, unnormalized trajectory from full_dataset_ref
                            # Shape: (Timesteps, Variables)
                            original_traj_for_comp = full_dataset_ref.trajectories[original_data_idx]
                            initial_conditions_y0 = original_traj_for_comp[0, :] # First time point
                            predicted_params_for_sim = preds_unscaled_np[i, :]

                            simulated_df = integrate_ode(
                                t_span=t_span_eval, y0=initial_conditions_y0,
                                params_values=predicted_params_for_sim,
                                t_eval=t_eval_points, param_names=PARAM_NAMES,
                                apply_bounds=True # Critically important
                            )
                            num_trajectory_comps_attempted += 1

                            if simulated_df is not None:
                                simulated_traj_for_comp = simulated_df[STATE_VARIABLES].values
                                if len(simulated_traj_for_comp) == len(original_traj_for_comp):
                                    # MSE between original (unnormalized) and re-simulated trajectory
                                    current_traj_mse = np.mean((original_traj_for_comp - simulated_traj_for_comp)**2)
                                    sum_trajectory_mse += current_traj_mse
                                else: # Length mismatch (e.g., solver stopped early)
                                    sum_trajectory_mse += trajectory_fail_penalty
                            else: # Simulation failed (integrate_ode returned None)
                                sum_trajectory_mse += trajectory_fail_penalty
            except Exception as e:
                 print(f"\nError during validation metric calculation (batch {batch_idx}): {e}")
                 total_unscaled_mae += np.nan * current_batch_size # Mark as invalid if error

    avg_scaled_loss = total_scaled_loss / num_samples_processed if num_samples_processed > 0 else 0.0
    avg_unscaled_mae = total_unscaled_mae / num_samples_processed if num_samples_processed > 0 else np.nan

    val_metrics_dict['val_loss_scaled'] = avg_scaled_loss
    val_metrics_dict['val_mae_unscaled'] = avg_unscaled_mae

    if num_trajectory_comps_attempted > 0:
         avg_trajectory_mse = sum_trajectory_mse / num_trajectory_comps_attempted
         val_metrics_dict['val_trajectory_mse'] = avg_trajectory_mse
    # If no comps attempted, key 'val_trajectory_mse' won't be in dict (handled by .get in main loop)

    return avg_scaled_loss, val_metrics_dict

# --- Utility Functions ---
def save_checkpoint(epoch: int, model: nn.Module, optimizer: optim.Optimizer,
                    param_scaler: MinMaxScaler, loss: float, filepath: str):
    """Saves model, optimizer, scaler, epoch, and loss to a checkpoint file."""
    print(f"Saving checkpoint to {filepath}...")
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'param_scaler': param_scaler, 'loss': loss}
    torch.save(state, filepath)
    print("Checkpoint saved.")

def load_checkpoint(filepath: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
                    device: Optional[torch.device] = None) -> Tuple[int, float, Optional[MinMaxScaler]]:
    """Loads a checkpoint. Returns start_epoch, best_loss, and param_scaler."""
    if not os.path.exists(filepath): raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    effective_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {filepath} to {effective_device}...")
    checkpoint = torch.load(filepath, map_location=effective_device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(effective_device) # Ensure model is on the correct device
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state tensors to the correct device
        for opt_state_val in optimizer.state.values():
            for k, v_tensor in opt_state_val.items():
                if isinstance(v_tensor, torch.Tensor): opt_state_val[k] = v_tensor.to(effective_device)
    start_epoch = checkpoint.get('epoch', -1) + 1 # Start from NEXT epoch
    best_loss = checkpoint.get('loss', float('inf'))
    param_scaler_loaded = checkpoint.get('param_scaler')
    if not param_scaler_loaded: print("Warning: Parameter scaler not found in checkpoint.")
    print(f"Checkpoint loaded. Resuming from Epoch {start_epoch}. Previous best loss: {best_loss:.6f}")
    return start_epoch, best_loss, param_scaler_loaded

def plot_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plots training and validation loss curves, and optionally MAE and Trajectory MSE."""
    num_plots = 1 # Start with scaled loss plot
    if 'val_mae_unscaled' in history and any(m is not None and not np.isnan(m) for m in history['val_mae_unscaled']):
        num_plots += 1
    if 'val_trajectory_mse' in history and any(m is not None and not np.isnan(m) for m in history['val_trajectory_mse']):
        num_plots += 1

    plt.figure(figsize=(6 * num_plots, 5))
    plot_idx = 1

    # Plot 1: Scaled Parameter Loss (MSE)
    plt.subplot(1, num_plots, plot_idx); plt.title('Scaled Parameter Loss (MSE)')
    if 'train_loss' in history: plt.plot(history['train_loss'], label='Training Loss (Scaled)')
    if 'val_loss_scaled' in history: plt.plot(history['val_loss_scaled'], label='Validation Loss (Scaled)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plot_idx +=1

    # Plot 2: Unscaled Parameter MAE (if available)
    if 'val_mae_unscaled' in history and any(m is not None and not np.isnan(m) for m in history['val_mae_unscaled']):
        plt.subplot(1, num_plots, plot_idx); plt.title('Unscaled Parameter MAE')
        plt.plot(range(len(history['val_mae_unscaled'])), history['val_mae_unscaled'], label='Validation MAE (Unscaled)', marker='.')
        plt.xlabel('Epoch'); plt.ylabel('Mean Absolute Error'); plt.legend(); plt.grid(True); plot_idx +=1
    elif num_plots > 1 and ('val_mae_unscaled' not in history or not any(m is not None and not np.isnan(m) for m in history['val_mae_unscaled'])):
        # Placeholder if subplot was intended but no data
        plt.subplot(1, num_plots, plot_idx); plt.title('Unscaled Parameter MAE')
        plt.text(0.5,0.5,'No valid MAE data', ha='center',va='center',transform=plt.gca().transAxes)
        plt.xlabel('Epoch'); plt.ylabel('Mean Absolute Error'); plt.grid(True); plot_idx +=1

    # Plot 3: Validation Trajectory MSE (if available)
    if 'val_trajectory_mse' in history and any(m is not None and not np.isnan(m) for m in history['val_trajectory_mse']):
        plt.subplot(1, num_plots, plot_idx); plt.title('Validation Trajectory MSE')
        plt.plot(range(len(history['val_trajectory_mse'])), history['val_trajectory_mse'], label='Validation Trajectory MSE', marker='.')
        plt.yscale('log') # Trajectory MSE can vary widely
        plt.xlabel('Epoch'); plt.ylabel('Mean Squared Error (Log Scale)'); plt.legend(); plt.grid(True); plot_idx +=1
    elif num_plots > 1 and ('val_trajectory_mse' not in history or not any(m is not None and not np.isnan(m) for m in history['val_trajectory_mse'])):
        # Placeholder if subplot was intended but no data
        plt.subplot(1, num_plots, plot_idx); plt.title('Validation Trajectory MSE')
        plt.text(0.5,0.5,'No valid Trajectory MSE data', ha='center',va='center',transform=plt.gca().transAxes)
        plt.xlabel('Epoch'); plt.ylabel('Mean Squared Error'); plt.grid(True); plot_idx +=1

    plt.tight_layout()
    if save_path: print(f"Saving plot to {save_path}..."); plt.savefig(save_path); print("Plot saved.")
    plt.show(block=False); plt.pause(1) # Allow plot to render in non-blocking manner

def save_results(results_df: pd.DataFrame, filepath: str):
    """Saves a DataFrame (e.g., predictions vs targets) to a CSV file."""
    print(f"Saving results to {filepath}..."); results_df.to_csv(filepath, index=False); print("Results saved.")

# --- Main Execution Block ---
def main(args):
    """Main function to orchestrate data loading, model training, and evaluation."""
    main_start_time = time.time()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")

    # Define file paths
    paths = {
        'best_model': os.path.join(args.output_dir, "best_model.pth"),
        'last_model': os.path.join(args.output_dir, "last_model.pth"),
        'plot': os.path.join(args.output_dir, "training_curves.png"),
        'results': os.path.join(args.output_dir, "validation_results.csv")
    }

    # --- Data Loading and Preparation ---
    print("Loading and preparing dataset...")
    try:
        # Load full dataset first (param_scaler not fitted yet)
        full_dataset_obj = ODESimulationDataset(
            args.data_path, PARAM_NAMES, STATE_VARIABLES,
            param_scaler=None, fit_scaler_on_this_data=False, # IMPORTANT
            use_noisy_data=not args.use_clean_data
        )
    except Exception as e:
        print(f"Fatal Error during dataset initialization: {e}"); return

    # Split indices for training and validation
    run_indices = list(range(len(full_dataset_obj)))
    train_indices, val_indices = train_test_split(
        run_indices, test_size=args.val_split, random_state=SEED, shuffle=True
    )
    print(f"Dataset split: Total runs={len(full_dataset_obj)}, Training runs={len(train_indices)}, Validation runs={len(val_indices)}")

    # Fit param_scaler ONLY on the training portion of the parameters
    param_scaler_fitted = MinMaxScaler(feature_range=(0, 1))
    param_scaler_fitted.fit(full_dataset_obj.parameters[train_indices])
    print("Parameter scaler fitted on training data.")

    # Assign the fitted scaler to the dataset object and transform ALL its parameters
    # This way, DataLoader will serve correctly scaled parameters for both train and val subsets
    full_dataset_obj.param_scaler = param_scaler_fitted
    full_dataset_obj.scaled_parameters = param_scaler_fitted.transform(full_dataset_obj.parameters)

    # Create DataLoaders using SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices) # No shuffling for validation is fine
    train_loader = DataLoader(full_dataset_obj, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(full_dataset_obj, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # --- Model Initialization ---
    print(f"Initializing model type: {args.model_type}...")
    if args.model_type == 'cnn':
        model = ParameterPredictorCNN(NUM_PARAMS, NUM_STATE_VARIABLES, full_dataset_obj.num_timesteps).to(device)
    elif args.model_type == 'lstm':
        model = ParameterPredictorLSTM(
            num_params=NUM_PARAMS, input_features=NUM_STATE_VARIABLES,
            hidden_size=args.lstm_hidden_size, num_layers=args.lstm_num_layers,
            lstm_dropout_rate=args.lstm_dropout_rate, fc_dropout_rate=args.fc_dropout_rate_lstm,
            bidirectional=args.lstm_bidirectional
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}. Choose 'cnn' or 'lstm'.")
    print(f"Model selected: {args.model_type.upper()}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer, Criterion, Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss() # Loss on scaled parameters
    scaler = GradScaler()   # ← NEW: handles dynamic loss scaling

    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=args.scheduler_patience, verbose=True
        )
        print("Using ReduceLROnPlateau LR scheduler.")

    # --- Training Loop Preparation ---
    history = {'train_loss':[], 'val_loss_scaled':[], 'val_mae_unscaled':[], 'val_trajectory_mse':[]}
    best_val_loss_for_checkpoint = float('inf') # Tracks val_loss_scaled
    epochs_without_improvement = 0
    epoch_to_start_from = 0

    # Resume from checkpoint if specified
    if args.resume_from:
        try:
            epoch_to_start_from, best_val_loss_for_checkpoint, loaded_scaler = load_checkpoint(
                args.resume_from, model, optimizer, device
            )
            if loaded_scaler: # If checkpoint had a scaler, use it
                param_scaler_fitted = loaded_scaler # This is now the reference scaler
                full_dataset_obj.param_scaler = param_scaler_fitted
                full_dataset_obj.scaled_parameters = param_scaler_fitted.transform(full_dataset_obj.parameters)
                print("Parameter scaler state loaded and applied from checkpoint.")
            else: # Should not happen if saved correctly, but good to note
                 print("Warning: Checkpoint did not contain a parameter scaler. Continuing with newly fitted one.")
        except FileNotFoundError:
            print(f"Checkpoint file {args.resume_from} not found. Starting training from scratch.")
        except Exception as e:
             print(f"Error loading checkpoint {args.resume_from}: {e}. Starting training from scratch.")

    # Determine t_span and t_eval for trajectory comparisons using actual data time points
    # Assuming all runs in the CSV have the same time points as the first run
    first_run_id_val = full_dataset_obj.run_ids[0]
    time_points_for_first_run = full_dataset_obj.data_df[
        full_dataset_obj.data_df['run_id'] == first_run_id_val
    ]['t']
    t_span_for_eval = (time_points_for_first_run.min(), time_points_for_first_run.max())
    t_eval_points_for_eval = np.sort(time_points_for_first_run.unique())

    # --- Main Training Loop ---
    print(f"Starting training from epoch {epoch_to_start_from + 1}...")
    for epoch_idx in range(epoch_to_start_from, args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch_idx + 1}/{args.epochs} ---")

        # Training phase
        current_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(current_train_loss)
        print(f"Training Loss (Scaled MSE): {current_train_loss:.6f}")

        # Validation phase
        current_val_loss_scaled, val_metrics_all = validate_epoch(
            model, val_loader, criterion, param_scaler_fitted, device, # Use the single fitted scaler
            full_dataset_ref=full_dataset_obj, num_trajectory_comps=args.num_traj_comps,
            t_span_eval=t_span_for_eval, t_eval_points=t_eval_points_for_eval,
            trajectory_fail_penalty=args.trajectory_fail_penalty
        )
        history['val_loss_scaled'].append(current_val_loss_scaled)
        history['val_mae_unscaled'].append(val_metrics_all.get('val_mae_unscaled', np.nan)) # Use .get for safety
        history['val_trajectory_mse'].append(val_metrics_all.get('val_trajectory_mse', np.nan))

        print(f"Validation Loss (Scaled MSE): {current_val_loss_scaled:.6f}")
        if 'val_mae_unscaled' in val_metrics_all: print(f"Validation MAE (Unscaled): {val_metrics_all['val_mae_unscaled']:.6f}")
        if 'val_trajectory_mse' in val_metrics_all: print(f"Validation Trajectory MSE: {val_metrics_all['val_trajectory_mse']:.6f}")

        print(f"Epoch duration: {time.time() - epoch_start_time:.2f} seconds")

        # Learning rate scheduler step (if used)
        if scheduler: scheduler.step(current_val_loss_scaled) # Scheduler monitors scaled val loss

        # Checkpoint saving and Early Stopping (based on val_loss_scaled)
        if current_val_loss_scaled < best_val_loss_for_checkpoint:
            delta = best_val_loss_for_checkpoint - current_val_loss_scaled
            print(f"Validation loss improved by {delta:.6f} ({best_val_loss_for_checkpoint:.6f} --> {current_val_loss_scaled:.6f}). Saving best model...")
            best_val_loss_for_checkpoint = current_val_loss_scaled
            save_checkpoint(epoch_idx, model, optimizer, param_scaler_fitted, best_val_loss_for_checkpoint, paths['best_model'])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve for {epochs_without_improvement} epoch(s). Best was {best_val_loss_for_checkpoint:.6f}")

        # Save last model checkpoint (useful for ad-hoc resuming or inspection)
        save_checkpoint(epoch_idx, model, optimizer, param_scaler_fitted, current_val_loss_scaled, paths['last_model'])

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement on validation loss.")
            break

    total_training_duration = time.time() - main_start_time
    print(f"\n--- Training Finished ---")
    print(f"Total training duration: {total_training_duration:.2f} seconds ({total_training_duration/60:.2f} minutes)")
    print(f"Best validation loss (Scaled MSE) achieved: {best_val_loss_for_checkpoint:.6f}")

    # --- Post-Training: Final Evaluation on Validation Set using Best Model ---
    print("\nLoading best model for final evaluation on validation set...")
    try:
        # Load the best model state and its associated scaler
        _, _, final_evaluation_scaler = load_checkpoint(paths['best_model'], model, device=device)
        if final_evaluation_scaler is None: # Fallback if scaler wasn't in checkpoint (should not happen)
            print("Warning: Best model checkpoint did not contain scaler. Using last known scaler for final eval.")
            final_evaluation_scaler = param_scaler_fitted

        model.eval() # Ensure model is in evaluation mode
        all_final_preds_unscaled, all_final_targets_unscaled, all_final_run_ids = [], [], []
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(val_loader): # True targets not strictly needed, we get them from full_dataset_obj
                inputs_dev = inputs.to(device) # Renamed to avoid conflict
                predictions_scaled_final = model(inputs_dev)
                preds_unscaled_final = final_evaluation_scaler.inverse_transform(predictions_scaled_final.cpu().numpy())

                # Get original indices and corresponding original targets/run_ids
                current_batch_size_final = inputs.size(0)
                original_indices_in_batch_final = []
                if hasattr(val_loader.sampler, 'indices'):
                    sampler_start_idx_final = batch_idx * val_loader.batch_size
                    sampler_end_idx_final = sampler_start_idx_final + current_batch_size_final
                    original_indices_in_batch_final = val_loader.sampler.indices[sampler_start_idx_final:sampler_end_idx_final]
                else: # Fallback
                     processed_so_far_final = batch_idx * val_loader.batch_size
                     original_indices_in_batch_final = list(range(processed_so_far_final, processed_so_far_final + current_batch_size_final))

                targets_unscaled_final = np.array([full_dataset_obj.get_original_params(i) for i in original_indices_in_batch_final])
                run_ids_final = [full_dataset_obj.get_run_id(i) for i in original_indices_in_batch_final]

                all_final_preds_unscaled.append(preds_unscaled_final)
                all_final_targets_unscaled.append(targets_unscaled_final)
                all_final_run_ids.extend(run_ids_final)

        # Concatenate results from all validation batches
        final_predictions_np = np.concatenate(all_final_preds_unscaled, axis=0)
        final_targets_np = np.concatenate(all_final_targets_unscaled, axis=0)

        # Create DataFrame for detailed results
        results_data_dict = {'run_id': all_final_run_ids}
        for i, param_n in enumerate(PARAM_NAMES): # param_n to avoid conflict
            results_data_dict[f'target_{param_n}'] = final_targets_np[:, i]
            results_data_dict[f'predicted_{param_n}'] = final_predictions_np[:, i]
            results_data_dict[f'error_{param_n}'] = final_predictions_np[:, i] - final_targets_np[:, i]

        results_final_df = pd.DataFrame(results_data_dict)
        save_results(results_final_df, paths['results'])

    except FileNotFoundError:
        print(f"Best model checkpoint not found at {paths['best_model']}. Skipping final evaluation.")
    except Exception as e:
        print(f"Error during final evaluation phase: {e}")

    # Plot training curves from history
    print("\nPlotting training curves...")
    plot_curves(history, paths['plot'])

    print("\nScript finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train inverse model for ODE parameter recovery lstm.")
    # Data & Output Arguments
    parser.add_argument('--data_path', type=str, default='../Data/simulation_results.csv', help="Path to the simulation results CSV file.")
    parser.add_argument('--output_dir', type=str, default="../ode_inverse_output_lstm2", help="Directory to save checkpoints, plots, and results.")
    parser.add_argument('--use_clean_data', action='store_true', help="Use 'X','Y','Z' columns instead of 'X_noisy','Y_noisy','Z_noisy'.")

    # Model Choice Arguments
    parser.add_argument('--model_type', type=str, default='lstm', choices=['cnn', 'lstm'], help="Type of model architecture to use ('cnn' or 'lstm').")
    # LSTM Specific Arguments
    parser.add_argument('--lstm_hidden_size', type=int, default=512, help="Number of features in LSTM hidden state.")
    parser.add_argument('--lstm_num_layers', type=int, default=4, help="Number of recurrent LSTM layers.")
    parser.add_argument('--lstm_dropout_rate', type=float, default=0.2, help="Dropout rate between LSTM layers (if lstm_num_layers > 1).")
    parser.add_argument('--fc_dropout_rate_lstm', type=float, default=0.5, help="Dropout rate for the FC layers following the LSTM block.")
    parser.add_argument('--lstm_bidirectional', default=True, help="Use a bidirectional LSTM.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=5000, help="Maximum number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and validation.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Optimizer learning rate (AdamW).") # Adjusted default
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 penalty) for AdamW optimizer.") # Adjusted default
    parser.add_argument('--val_split', type=float, default=0.2, help="Fraction of data to use for validation (e.g., 0.2 for 20%).")
    parser.add_argument('--patience', type=int, default=20, help="Number of epochs with no improvement on val_loss_scaled to wait before early stopping.") # Adjusted default
    parser.add_argument('--use_scheduler', action='store_true', help="Enable ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument('--scheduler_patience', type=int, default=10, help="Patience for LR scheduler (epochs of no improvement before reducing LR).") # Adjusted default

    # System & Utility Arguments
    parser.add_argument('--num_workers', type=int, default=min(4, os.cpu_count() if os.cpu_count() else 1), help="Number of worker processes for DataLoader.")
    parser.add_argument('--no_gpu', action='store_true', help="Disable GPU usage even if CUDA is available.")
    parser.add_argument('--resume_from', type=str, default = "ode_inverse_output_lstm/last_model.pth", help="Path to a checkpoint file to resume training from.")

    # Validation & Trajectory Comparison Arguments
    parser.add_argument('--num_traj_comps', type=int, default=0, help="Number of validation examples for trajectory comparison per epoch.")
    parser.add_argument('--trajectory_fail_penalty', type=float, default=DEFAULT_TRAJ_FAIL_PENALTY, help="Penalty value for failed/mismatched trajectory simulations.")

    args, unknown = parser.parse_known_args()
    main(args)
