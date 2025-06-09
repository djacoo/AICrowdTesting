"""
Training script for an RL agent on the CustomCityEnv.

This script demonstrates a two-phase training process:
1. Initial training focused on two Key Performance Indicators (KPIs): comfort and emissions.
2. Continued training incorporating additional KPIs (e.g., grid impact),
   starting from the policy learned in the first phase.

It uses stable-baselines3 for the PPO agent and CustomCityEnv for the environment.
Evaluation results and trained models are saved for each phase.
TensorBoard logging is enabled for monitoring training progress.
"""
import os
import sys
import time
import torch
import random
import numpy as np
import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import json
import shutil
import glob
from pathlib import Path

# Import stable_baselines3 directly to ensure all components are available
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
# BaseCallback is now imported with other callbacks above
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor

# Import callbacks from stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# ===== Utility Functions =====

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_reward(reward: float, min_reward: float, max_reward: float) -> float:
    """Normalize reward to a 0-10 scale for comparison."""
    return ((reward - min_reward) / (max_reward - min_reward)) * 10 if max_reward > min_reward else reward

class TrainingLogger:
    """Helper class to log training metrics for comparison."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'episode': [],
            'reward': [],
            'normalized_reward': [],
            'episode_length': [],
            'phase': []
        }
    
    def log_episode(self, episode: int, reward: float, episode_length: int, phase: str, 
                   min_reward: float = 0, max_reward: float = 10) -> None:
        """Log metrics for a single episode."""
        norm_reward = normalize_reward(reward, min_reward, max_reward)
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['normalized_reward'].append(norm_reward)
        self.metrics['episode_length'].append(episode_length)
        self.metrics['phase'].append(phase)
        
        # Print episode summary
        print(f"\nEpisode {episode + 1} ({phase}):")
        print(f"  Raw reward: {reward:.4f}")
        print(f"  Normalized reward: {norm_reward:.4f}")
        print(f"  Episode length: {episode_length}")
    
    def save_to_csv(self, filename: str) -> None:
        """Save logged metrics to a CSV file."""
        df = pd.DataFrame(self.metrics)
        filepath = os.path.join(self.log_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\nMetrics saved to {filepath}")

# Clean up previous training artifacts
def cleanup_previous_runs():
    """Remove previous model files and logs to ensure a fresh start."""
    # Remove model files
    for model_file in glob.glob('ppo_*_model*') + glob.glob('models/ppo_*_model*'):
        try:
            if os.path.isfile(model_file):
                os.remove(model_file)
            elif os.path.isdir(model_file):
                shutil.rmtree(model_file)
        except Exception as e:
            print(f"Warning: Could not remove {model_file}: {e}")
    
    # Remove tensorboard logs
    for log_dir in ['ppo_tensorboard_logs_2kpi', 'ppo_tensorboard_logs_multi_kpi']:
        try:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
        except Exception as e:
            print(f"Warning: Could not remove {log_dir}: {e}")

# Run cleanup at the start
cleanup_previous_runs()

# Assuming AICrowdControl.py and CustomCityEnv.py are in the same directory or PYTHONPATH
from AICrowdControl import PhaseWeights, BASELINE_KPIS # BASELINE_KPIS is used by CustomCityEnv
from CustomCityEnv import CustomCityEnv

# --- Constants for training phases ---
SEED = 42
set_seed(SEED)

TOTAL_TIMESTEPS_2KPI = 200000
TOTAL_TIMESTEPS_MULTI = 250000
EVAL_FREQ = 5000  # Evaluate every 5k steps
N_EVAL_EPISODES = 5  # Number of episodes for evaluation

MODEL_SAVE_PATH_2KPI_BASE = "ppo_2kpi_model"
MODEL_SAVE_PATH_MULTI_BASE = "ppo_multi_kpi_model"
TENSORBOARD_LOG_PATH_2KPI_BASE = os.path.join("ppo_tensorboard_logs_2kpi")
TENSORBOARD_LOG_PATH_MULTI_BASE = os.path.join("ppo_tensorboard_logs_multi_kpi")

REWARD_BOUNDS = {
    '2kpi': {'min': 0, 'max': 10},    # Adjust these values based on your environment
    'multi': {'min': 0, 'max': 10}    # Adjust these values based on your environment
}

training_logger = TrainingLogger(log_dir='training_logs')
EVAL_RESULTS_CSV_2KPI = 'evaluation_results_2kpi.csv'
EVAL_RESULTS_CSV_MULTI = 'evaluation_results_multi_kpi.csv'

def smooth_data(y, window_size=5):
    """Smooth 1D array using a moving average."""
    if len(y) < window_size:
        return y, np.zeros_like(y)
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(y, window, mode='same')
    
    # Calculate rolling standard deviation
    rolling_std = np.zeros_like(y)
    for i in range(len(y) - window_size + 1):
        rolling_std[i + window_size//2] = np.std(y[i:i+window_size])
    
    # Handle edges
    rolling_std[:window_size//2] = rolling_std[window_size//2]
    rolling_std[-(window_size//2):] = rolling_std[-(window_size//2)-1]
    
    return smoothed, rolling_std

def plot_training_results(window_size=25):
    """
    Generate training and KPI visualisations using a unified style.
    
    Args:
        window_size (int): Size of the moving average window.
    
    Returns:
        None
    """
    try:
        import glob
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        def load_rewards(log_base):
            """
            Load reward history from all runs under a directory.
            
            Args:
                log_base (str): Base path for the log directory.
            
            Returns:
                pd.DataFrame: Reward history.
            """
            try:
                runs = [d for d in glob.glob(os.path.join(log_base, '*')) if os.path.isdir(d)]
                if not runs:
                    runs = [log_base]
                histories = []
                for run in runs:
                    event_files = glob.glob(os.path.join(run, 'events.out.tfevents.*'))
                    if not event_files:
                        continue
                    event_file = max(event_files, key=os.path.getmtime)
                    acc = EventAccumulator(run)
                    acc.Reload()
                    if 'rollout/ep_rew_mean' not in acc.Tags()['scalars']:
                        continue
                    events = acc.Scalars('rollout/ep_rew_mean')
                    steps = [e.step for e in events]
                    rewards = [e.value for e in events]
                    series = pd.Series(rewards, index=steps)
                    # Reset index to avoid duplicate indices when concatenating
                    series = series[~series.index.duplicated(keep='first')]
                    if not series.empty:
                        histories.append(series)
                if histories:
                    # Ensure all series have the same index before concatenation
                    result = pd.concat(histories, axis=1, join='outer')
                    # Forward fill any missing values that might have been introduced
                    return result.ffill().bfill()
            except Exception as e:
                print(f"Error loading rewards from {log_base}: {e}")
            return None

        # Load reward histories
        print("Loading training histories...")
        history_2kpi = load_rewards(TENSORBOARD_LOG_PATH_2KPI_BASE)
        history_multi = load_rewards(TENSORBOARD_LOG_PATH_MULTI_BASE)

        # Create figure for training curves
        print("Generating training curves...")
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

        # Plot training curves
        for df, label, color in [
            (history_2kpi, '2-KPI Training', 'green'),
            (history_multi, 'Multi-KPI Training', 'orange')
        ]:
            if df is None or df.empty:
                print(f"No data available for {label}")
                continue
                
            try:
                df.columns = [f'Run {i+1}' for i in range(df.shape[1])]
                mean = df.mean(axis=1)
                std = df.std(axis=1)
                smoothed = mean.rolling(window=window_size, min_periods=1).mean()

                ax1.plot(smoothed, linewidth=2, label=label, color=color)
                ax1.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.1, label=f'{label} ±1 std')
            except Exception as e:
                print(f"Error plotting {label}: {e}")

        ax1.set_title('Training Reward')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)

        # Save training curves plot
        os.makedirs('plots', exist_ok=True)
        plt.tight_layout()
        plt.savefig('plots/training_curves.png')
        plt.close(fig)

        # --- KPI Metrics ---
        print("Processing KPI metrics...")
        eval_dfs = []
        for csv_file, label, color in [
            (EVAL_RESULTS_CSV_2KPI, '2-KPI Eval', 'green'),
            (EVAL_RESULTS_CSV_MULTI, 'Multi-KPI Eval', 'orange')
        ]:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty and 'episode_reward' in df.columns:
                        eval_dfs.append((df, label, color))
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
            else:
                print(f"CSV file not found: {csv_file}")

        if eval_dfs:
            print("Generating KPI comparison plot...")
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
            kpi_cols = [c for c in eval_dfs[0][0].columns if c.startswith('KPI_')]
            if kpi_cols:
                try:
                    names = [k[4:].replace('_', ' ').title() for k in kpi_cols]
                    x = np.arange(len(names))
                    width = 0.35 if len(eval_dfs) > 1 else 0.7
                    for i, (df, label, color) in enumerate(eval_dfs):
                        means = df[kpi_cols].mean().values
                        stds = df[kpi_cols].std().values
                        offset = width * (i - (len(eval_dfs)-1)/2)
                        ax2.bar(x + offset, means, width, yerr=stds, label=label, color=color, alpha=0.7, capsize=5)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(names, rotation=45, ha='right')
                    ax2.set_title('KPI Comparison')
                    ax2.set_ylabel('Score')
                    ax2.legend()
                    plt.tight_layout()
                    plt.savefig('plots/kpi_comparison.png')
                    plt.close(fig2)
                except Exception as e:
                    print(f"Error generating KPI comparison plot: {e}")
        else:
            print("No evaluation data available for KPI comparison")
            
    except ImportError as e:
        print(f"Required libraries not found: {e}")
    except Exception as e:
        print(f"Error in plot_training_results: {str(e)}")
    finally:
        # Ensure all figures are closed
        plt.close('all')

def main_multi_kpi_training(num_buildings: int, timesteps_per_episode: int) -> PPO:
    """
    Trains a PPO agent focusing on multiple KPIs (comfort, emissions, grid impact, and resilience).
    
    Args:
        num_buildings: Number of buildings in the environment
        timesteps_per_episode: Number of timesteps per episode
        
    Returns:
        PPO: Trained PPO agent
    """
    # Import EvalCallback here to ensure it's in scope
    from stable_baselines3.common.callbacks import EvalCallback
    
    print("\n=== Starting Multi-KPI Training Phase ===")
    
    # Set fixed random seed for reproducibility
    set_seed(SEED)
    
    # Define weights for multi-KPI training (comfort, emissions, grid impact, resilience)
    multi_kpi_weights = PhaseWeights(
        w1=0.3,  # Comfort
        w2=0.1,  # Emissions
        w3=0.3,  # Grid impact
        w4=0.3   # Resilience
    )
    
    # Ensure directories exist
    os.makedirs(TENSORBOARD_LOG_PATH_MULTI_BASE, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for Multi-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_multi = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    # Set environment seed for reproducibility
    if hasattr(env_multi, 'np_random') and env_multi.np_random is not None:
        env_multi.np_random.seed(SEED)
    elif hasattr(env_multi, 'seed'):
        env_multi.seed(SEED)
    env_multi.reset()

    # --- Agent Training (Multi-KPIs) ---
    print(f"Instantiating PPO agent for Multi-KPI training (TensorBoard logs: {TENSORBOARD_LOG_PATH_MULTI_BASE})...")
    
    # Initialize PPO agent with configuration for multi-KPI training
    policy_kwargs = {
        'net_arch': dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # Same network architecture as 2-KPI
        'activation_fn': torch.nn.ReLU,
        'ortho_init': True
    }
    
    # Learning rate schedule with warmup
    def lr_schedule(progress_remaining):
        """Learning rate with warmup and cosine decay"""
        # Linear warmup for first 20% of training
        warmup_frac = 0.2
        if progress_remaining > 1.0 - warmup_frac:
            warmup_progress = (1.0 - progress_remaining) / warmup_frac
            return 1e-3 * warmup_progress  # Higher initial learning rate
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 1e-6  # Lower minimum learning rate
        max_lr = 1e-3   # Higher maximum learning rate
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env_multi,
        learning_rate=lr_schedule,
        n_steps=2048,  # Number of steps to run for each environment per update
        batch_size=512,  # Minibatch size
        n_epochs=10,  # Number of epochs for optimization
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,  # Clip parameter for the policy and value functions
        clip_range_vf=None,  # Clip parameter for the value function
        ent_coef=0.01,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Maximum norm for the gradient clipping
        policy_kwargs=policy_kwargs,
        tensorboard_log=TENSORBOARD_LOG_PATH_MULTI_BASE,
        verbose=1,
        device='auto',
        seed=SEED
    )
    
    # --- Callbacks ---
    # Create evaluation environment
    eval_env = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    eval_env.seed(SEED)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/',
        log_path='evaluation',
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Custom callback for additional logging
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.log_freq = 1000  # Log every 1000 steps
            
        def _on_step(self) -> bool:
            # Log additional metrics at specified frequency
            if self.n_calls % self.log_freq == 0:
                # Track episode rewards and lengths
                if len(self.model.ep_info_buffer) > 0:
                    for info in self.model.ep_info_buffer:
                        if 'r' in info:
                            self.episode_rewards.append(info['r'])
                        if 'l' in info:
                            self.episode_lengths.append(info['l'])
                
                # Log mean reward and length
                if len(self.episode_rewards) > 0:
                    self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                if len(self.episode_lengths) > 0:
                    self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                
                # Log training metrics
                if hasattr(self.model, 'log_std'):
                    self.logger.record('train/entropy', self.model.log_std.exp().mean().item())
                
                # Log learning rate and clip range
                if hasattr(self.model, 'lr_schedule'):
                    self.logger.record('train/learning_rate', self.model.lr_schedule(self.model._current_progress_remaining))
                if hasattr(self.model, 'clip_range'):
                    clip_range = self.model.clip_range(self.model._current_progress_remaining)
                    if isinstance(clip_range, (list, tuple)):
                        clip_range = clip_range[0]
                    self.logger.record('train/clip_range', clip_range)
                
                # Log explained variance if available
                if hasattr(self.model, 'explained_variance'):
                    self.logger.record('train/explained_variance', self.model.explained_variance)
                
                # Log to console
                print(f"Step: {self.num_timesteps}")
                if len(self.episode_rewards) > 0:
                    print(f"Last 100 episodes mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
                
                # Force log to TensorBoard
                self.logger.dump(self.num_timesteps)
                
            return True
    
    # Progress bar callback
    class ProgressBarCallback(BaseCallback):
        def __init__(self, total_timesteps):
            super(ProgressBarCallback, self).__init__()
            self.progress_bar = None
            self.total_timesteps = total_timesteps
            
        def _on_training_start(self):
            self.progress_bar = tqdm.tqdm(total=self.total_timesteps, desc="Training progress")
            
        def _on_step(self) -> bool:
            if self.progress_bar is not None:
                self.progress_bar.update(1)  # Update progress by 1 step
            return True
            
        def _on_training_end(self):
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
    
    # Combine all callbacks
    callbacks = [
        eval_callback,
        CustomCallback(),
        ProgressBarCallback(TOTAL_TIMESTEPS_MULTI)
    ]
    
    # --- Training ---
    print(f"Starting Multi-KPI training for {TOTAL_TIMESTEPS_MULTI} timesteps...")
    
    try:
        # Train the model
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_MULTI,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="PPO_MultiKPI"
        )
        
        # Save the final model
        model_path = os.path.join('models', 'ppo_multi_kpi_model')
        model.save(model_path)
        print(f"\nMulti-KPI Training complete! Model saved to {model_path}")
        
        # Save the final evaluation results
        eval_results = []
        obs = eval_env.reset()
        for _ in range(5):  # Run 5 evaluation episodes
            done = False
            episode_reward = 0
            obs = eval_env.reset()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            eval_results.append(episode_reward)
        
        print(f"\nEvaluation Results (5 episodes):")
        print(f"Mean Reward: {np.mean(eval_results):.2f} ± {np.std(eval_results):.2f}")
        
        return model
        
    except Exception as e:
        print(f"\nError during Multi-KPI training: {str(e)}")
        raise
    finally:
        # Clean up
        env_multi.close()
        eval_env.close()

def main_2kpi_training(num_buildings: int, timesteps_per_episode: int) -> PPO:
    """
    Trains a PPO agent focusing on 2 KPIs (comfort and emissions).
    
    Args:
        num_buildings: Number of buildings in the environment
        timesteps_per_episode: Number of timesteps per episode
        
    Returns:
        PPO: Trained PPO agent
    """
    # Import EvalCallback here to ensure it's in scope
    from stable_baselines3.common.callbacks import EvalCallback
    
    print("\n=== Starting 2-KPI Training Phase ===")
    
    # Set fixed random seed for reproducibility
    set_seed(SEED)
    
    # Set up weights for comfort and emissions only
    comfort_emissions_weights = PhaseWeights(w1=0.5, w2=0.5, w3=0.0, w4=0.0)
    
    # Ensure directories exist
    os.makedirs(TENSORBOARD_LOG_PATH_2KPI_BASE, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for 2-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_2kpi = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    # Set environment seed for reproducibility
    if hasattr(env_2kpi, 'np_random') and env_2kpi.np_random is not None:
        env_2kpi.np_random.seed(SEED)
    elif hasattr(env_2kpi, 'seed'):
        env_2kpi.seed(SEED)
    env_2kpi.reset()

    # --- Agent Training (2-KPIs) ---
    print(f"Instantiating PPO agent for 2-KPI training (TensorBoard logs: {TENSORBOARD_LOG_PATH_2KPI_BASE})...")
    
    # Initialize PPO agent with enhanced configuration for 2-KPI training
    policy_kwargs = {
        'net_arch': dict(pi=[512, 256, 128], vf=[512, 256, 128]),  # Deeper network for better learning
        'activation_fn': torch.nn.ReLU,  # ReLU for better gradient flow in deeper networks
        'ortho_init': True,
        'log_std_init': -0.5,  # Slightly higher std for better exploration
        'optimizer_kwargs': {
            'eps': 1e-5,  # Better numerical stability
            'weight_decay': 1e-6  # Small weight decay for regularization
        }
    }
    
    # More aggressive learning rate schedule with longer training
    def lr_schedule(progress_remaining):
        """Learning rate with warmup and cosine decay"""
        import numpy as np
        
        # Linear warmup for first 20% of training
        warmup_frac = 0.2
        if progress_remaining > 1.0 - warmup_frac:
            warmup_progress = (1.0 - progress_remaining) / warmup_frac
            return 1e-3 * warmup_progress  # Higher initial learning rate
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 1e-6  # Lower minimum learning rate
        max_lr = 1e-3   # Higher maximum learning rate
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    # Enhanced PPO configuration with improved learning dynamics
    model_2kpi = PPO(
        "MlpPolicy",
        env_2kpi,
        # Core hyperparameters
        learning_rate=3e-4,          # Fixed learning rate
        n_steps=2048,                # Longer rollouts for better advantage estimation
        batch_size=512,              # Larger batch size for more stable updates
        n_epochs=10,                 # More epochs for better sample efficiency
        gamma=0.99,                  # Standard discount factor
        gae_lambda=0.95,             # Standard GAE lambda
        
        # Clipping and constraints
        clip_range=0.2,              # Standard clip range
        clip_range_vf=None,          # Disable separate VF clipping
        max_grad_norm=0.5,           # Standard gradient clipping
        target_kl=0.03,              # Slightly more lenient KL threshold
        
        # Exploration vs exploitation
        ent_coef=0.02,               # Balanced entropy coefficient
        vf_coef=0.5,                 # Standard value function coefficient
        
        # Network architecture
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),  # Separate networks for policy and value
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=-0.5  # Moderate initial exploration
        ),
        
        # Logging and monitoring
        tensorboard_log=TENSORBOARD_LOG_PATH_2KPI_BASE,
        verbose=2,
        device='auto',
        seed=42,
        normalize_advantage=True,
        
        # Disable unused features
        use_sde=False,
        sde_sample_freq=-1,
        stats_window_size=100
    )

    print(f"Starting 2-KPI training for {TOTAL_TIMESTEPS_2KPI} timesteps...")
    
    # Setup evaluation environment
    eval_env = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Enhanced evaluation callback with more frequent evaluation and better logging
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join('models', 'best_2kpi'),
        log_path=os.path.join('logs', '2kpi_eval'),
        eval_freq=max(5000 // num_buildings, 1),  # More frequent evaluation
        deterministic=False,  # Use stochastic actions for evaluation for better exploration
        render=False,
        n_eval_episodes=10,  # More episodes for more stable metrics
        warn=False,
        # Add custom eval callback for additional metrics
        callback_on_new_best=None,
        # Log additional info
        verbose=1
    )
    
    # Add a simple custom callback for logging
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np
    
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.log_freq = 1000  # Log every 1000 steps
            
        def _on_step(self) -> bool:
            # Log additional metrics at specified frequency
            if self.n_calls % self.log_freq == 0:
                # Track episode rewards and lengths
                if len(self.model.ep_info_buffer) > 0:
                    for info in self.model.ep_info_buffer:
                        if 'r' in info:
                            self.episode_rewards.append(info['r'])
                        if 'l' in info:
                            self.episode_lengths.append(info['l'])
                
                # Log mean reward and length
                if len(self.episode_rewards) > 0:
                    self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                    self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                
                # Log learning rate
                if hasattr(self.model, 'lr_schedule'):
                    if callable(self.model.lr_schedule):
                        lr = self.model.lr_schedule(1.0 - (self.num_timesteps / self.model._total_timesteps))
                    else:
                        lr = self.model.lr_schedule
                    self.logger.record('train/learning_rate', lr)
                
                # Log clip range
                if hasattr(self.model, 'clip_range'):
                    if callable(self.model.clip_range):
                        clip_range = self.model.clip_range(1.0 - (self.num_timesteps / self.model._total_timesteps))
                    else:
                        clip_range = self.model.clip_range
                    self.logger.record('train/clip_range', clip_range)
                
                # Log explained variance if available
                if hasattr(self.model, 'explained_variance'):
                    self.logger.record('train/explained_variance', self.model.explained_variance)
                
                # Log to console
                print(f"Step: {self.num_timesteps}")
                if len(self.episode_rewards) > 0:
                    print(f"Last 100 episodes mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
                
                # Force log to TensorBoard
                self.logger.dump(self.num_timesteps)
                
            return True
    
    # Combine callbacks
    callbacks = [eval_callback, CustomCallback()]
    
    # Ensure the callback is properly initialized
    if not hasattr(eval_callback, 'evaluations_results'):
        eval_callback.evaluations_results = {'episodes_lengths': [], 'episodes_rewards': []}
    
    # Train the model with enhanced callbacks and progress tracking
    try:
        from tqdm.auto import tqdm
        progress_bar = tqdm(total=TOTAL_TIMESTEPS_2KPI, desc="Training progress")
        
        # Update progress bar callback
        class ProgressBarCallback(BaseCallback):
            def __init__(self, total_timesteps):
                super(ProgressBarCallback, self).__init__()
                self.progress_bar = None
                self.total_timesteps = total_timesteps
                
            def _on_training_start(self):
                self.progress_bar = tqdm(total=self.total_timesteps, desc="Training progress")
                
            def _on_step(self) -> bool:
                self.progress_bar.update(self.training_env.num_envs)
                return True
                
            def _on_training_end(self):
                self.progress_bar.close()
        
        # Add progress bar to callbacks
        callbacks.append(ProgressBarCallback(total_timesteps=TOTAL_TIMESTEPS_2KPI))
    except ImportError:
        print("tqdm not installed, progress bar will not be shown")
    
    # Train the model with all callbacks
    model_2kpi.learn(
        total_timesteps=TOTAL_TIMESTEPS_2KPI,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="ppo_2kpi_training",
        reset_num_timesteps=True
    )

    # Save the final model
    model_save_path = os.path.join('models', MODEL_SAVE_PATH_2KPI_BASE + ".zip")
    print(f"2-KPI Training finished. Saving model to {model_save_path} ...")
    model_2kpi.save(model_save_path)
    
    # Clean up
    eval_env.close()

    env_2kpi.close()
    print("2-KPI Training environment closed.")

    # --- Evaluation ---
    import numpy as np  # Import numpy for calculations
    print("\n=== Evaluating 2-KPI Trained Model ===")
    eval_env = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    eval_env.seed(SEED)  # Set fixed seed for evaluation
    
    # Evaluate the agent
    episode_rewards = []
    episode_lengths = []
    
    for i in range(N_EVAL_EPISODES):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"Starting 2-KPI Evaluation Episode {i+1}/{N_EVAL_EPISODES}")
        
        while not done:
            action, _ = model_2kpi.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log the evaluation episode
        training_logger.log_episode(
            episode=i,
            reward=episode_reward,
            episode_length=episode_length,
            phase='2kpi_eval',
            min_reward=REWARD_BOUNDS['2kpi']['min'],
            max_reward=REWARD_BOUNDS['2kpi']['max']
        )
        
        print(f"2-KPI Evaluation Episode {i+1} Reward: {episode_reward:.4f}")
    
    # Save evaluation results
    eval_results = pd.DataFrame({
        'episode': range(1, N_EVAL_EPISODES + 1),
        'reward': episode_rewards,
        'normalized_reward': [
            normalize_reward(r, REWARD_BOUNDS['2kpi']['min'], REWARD_BOUNDS['2kpi']['max']) 
            for r in episode_rewards
        ],
        'episode_length': episode_lengths,
        'phase': ['2kpi_eval'] * N_EVAL_EPISODES
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(EVAL_RESULTS_CSV_2KPI) or '.', exist_ok=True)
    eval_results.to_csv(EVAL_RESULTS_CSV_2KPI, index=False)
    
    # Log summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_normalized = np.mean(eval_results['normalized_reward'])
    
    print("\n=== 2-KPI Evaluation Summary ===")
    print(f"Episodes: {N_EVAL_EPISODES}")
    print(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean Normalized Reward: {mean_normalized:.4f}")
    print(f"Min/Max Reward: {min(episode_rewards):.4f}/{max(episode_rewards):.4f}")
    print(f"Results saved to: {EVAL_RESULTS_CSV_2KPI}")
    print("================================")
    
    # Close the environment
    eval_env.close()
    print(f"2-KPI Evaluation results saved to {EVAL_RESULTS_CSV_2KPI}")
    
    # Save training logger data
    training_logger.save_to_csv('training_2kpi_logs.csv')
    
    return model_2kpi  # Return the trained model for potential use in multi-KPI training
    print("2-KPI Evaluation environment closed.")
    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for Multi-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_multi = CustomCityEnv(
        phase_weights=all_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    env_multi.seed(SEED)  # Set environment seed for reproducibility

    # --- Agent Setup: Load policy from 2-KPI model ---
    path_to_2kpi_model = MODEL_SAVE_PATH_2KPI_BASE + ".zip"
    print(f"Loading policy from 2-KPI model: {path_to_2kpi_model}")

    # First create the environment to get the correct observation space
    print(f"Setting up environment to get observation space...")
    temp_env = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    obs_shape = temp_env.observation_space.shape[0]
    temp_env.close()
    
    # Initialize PPO agent with enhanced configuration for multi-KPI training
    policy_kwargs = {
        'net_arch': dict(pi=[256, 256], vf=[256, 256]), # MODIFIED: Smaller network
        'activation_fn': torch.nn.ReLU,  # ReLU for deeper networks
        'ortho_init': True,
        'log_std_init': -0.5,  # Slightly higher initial std for better exploration
        'optimizer_kwargs': {
            'eps': 1e-8,  # Increased numerical stability
            'weight_decay': 1e-6  # Small weight decay for better generalization
        }
    }
    
    # Learning rate schedule with warmup and cosine decay for multi-KPI
    def lr_schedule(progress_remaining):
        """Cosine decay with warmup for learning rate with longer warmup"""
        # Linear warmup for first 20% of training
        warmup_frac = 0.2
        if progress_remaining > 1.0 - warmup_frac:
            warmup_progress = (1.0 - progress_remaining) / warmup_frac
            return 2e-4 * warmup_progress  # MODIFIED: Lower initial LR
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 1e-6 # MODIFIED: Lower min LR
        max_lr = 2e-4 # MODIFIED: Lower max LR
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    # Initialize the model with optimized hyperparameters for multi-KPI training
    model_multi_kpi = PPO(
        "MlpPolicy",
        env_multi_kpi,
        learning_rate=lr_schedule,
        n_steps=4096,           # More steps per update for better gradient estimates
        batch_size=512,         # Larger batch size for more stable updates
        n_epochs=10,            # MODIFIED: Fewer epochs
        gamma=0.995,            # Slightly higher discount factor for longer-term rewards
        gae_lambda=0.97,        # Adjusted for better bias-variance tradeoff
        clip_range=0.18,        # Slightly higher clip range for more exploration
        clip_range_vf=0.18,     # Add clipping for value function
        ent_coef=0.02,         # MODIFIED: Increased entropy
        vf_coef=0.8,            # Higher weight on value function loss
        max_grad_norm=0.8,      # Increased gradient clipping for stability
        use_sde=False,          # Disable SDE for discrete action spaces
        target_kl=0.03,         # Slightly higher KL threshold
        tensorboard_log=TENSORBOARD_LOG_PATH_MULTI_BASE,
        policy_kwargs=policy_kwargs,
        verbose=2,              # More detailed logging
        device='auto',
        normalize_advantage=True,
        seed=42
    )
    
    # Load the 2-KPI model if it exists
    if os.path.exists(path_to_2kpi_model):
        print(f"Loading weights from 2-KPI model: {path_to_2kpi_model}")
        try:
            # Load the saved model with the original environment to extract weights
            saved_model = PPO.load(path_to_2kpi_model, device='auto')
            # Get the state dict
            saved_state_dict = saved_model.policy.state_dict()
            # Load the state dict into our new model
            model_multi_kpi.policy.load_state_dict(saved_state_dict, strict=False)
            print("Successfully loaded compatible weights from 2-KPI model")
        except Exception as e:
            print(f"Could not load 2-KPI model weights due to: {str(e)}")
            print("Starting training from scratch")
    else:
        print("2-KPI model not found, starting training from scratch")
    


    # --- Agent Training (Multi-KPIs) ---
    print(f"Starting Multi-KPI training for {TOTAL_TIMESTEPS_MULTI} timesteps...")
    
    # Add callbacks for evaluation and learning rate scheduling
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    import numpy as np
    
    # Create eval environment
    eval_env = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    
    # Wrap eval env if needed
    if not isinstance(eval_env, DummyVecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])
    
    # Define evaluation callback with more frequent checks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model_multi_kpi/',
        log_path='./logs_multi_kpi/',
        eval_freq=max(EVAL_FREQ // num_buildings, 1),  # Use the global EVAL_FREQ
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # More episodes for better evaluation
        warn=False
    )
    
    # Learning rate schedule
    def lr_schedule(progress_remaining):
        """Linear decay for learning rate"""
        initial_lr = 2e-4
        min_lr = 1e-5
        return min_lr + (initial_lr - min_lr) * progress_remaining
    
    # Update learning rate
    model_multi_kpi.learning_rate = lr_schedule(1.0)  # Start with initial LR
    
    # Train the agent with evaluation callback and progress tracking
    model_multi_kpi.learn(
        total_timesteps=TOTAL_TIMESTEPS_MULTI,
        callback=eval_callback,
        progress_bar=True,
        tb_log_name="PPO_MultiKPI",
        reset_num_timesteps=False  # Continue from previous training
    )

    model_save_path = MODEL_SAVE_PATH_MULTI_BASE + ".zip"
    print(f"Multi-KPI Training finished. Saving model to {model_save_path} ...")
    print("Multi-KPI Evaluation environment closed.")
    
    # Save training logger data
    training_logger.save_to_csv('training_multi_kpi_logs.csv')
    
    print("=== Multi-KPI Training and Evaluation Phase Finished ===")
    
    return model_multi_kpi

    # Plot training and evaluation results
    plot_training_results()

def compare_training_results():
    """Compare and visualize results from 2-KPI and Multi-KPI training."""
    try:
        # Define file paths
        eval_results_2kpi = 'evaluation_results_2kpi.csv'
        
        # Check if 2-KPI results exist
        if not os.path.exists(eval_results_2kpi):
            print(f"2-KPI evaluation results not found at {eval_results_2kpi}")
            return
            
        # Load and print 2-KPI results
        results_2kpi = pd.read_csv(eval_results_2kpi)
        print("\n=== 2-KPI Training Results ===")
        print(f"Mean Reward: {results_2kpi['reward'].mean():.4f} ± {results_2kpi['reward'].std():.4f}")
        print(f"Mean Normalized Reward: {results_2kpi['normalized_reward'].mean():.4f} ± {results_2kpi['normalized_reward'].std():.4f}")
        
        # Check for Multi-KPI results
        eval_results_multi = 'evaluation_results_multi.csv'
        if os.path.exists(eval_results_multi):
            results_multi = pd.read_csv(eval_results_multi)
            print("\n=== Multi-KPI Training Results ===")
            print(f"Mean Reward: {results_multi['reward'].mean():.4f} ± {results_multi['reward'].std():.4f}")
            print(f"Mean Normalized Reward: {results_multi['normalized_reward'].mean():.4f} ± {results_multi['normalized_reward'].std():.4f}")
        
    except Exception as e:
        print(f"Error comparing training results: {str(e)}")

if __name__ == "__main__":
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # Training parameters
    NUM_BUILDINGS_MAIN = 1  # Number of buildings to train on
    TIMESTEPS_PER_EPISODE_MAIN = 24 * 7  # One week of hourly timesteps
    ENABLE_MULTI_KPI_TRAINING = True  # Enable multi-KPI training
    
    # Set fixed random seed for reproducibility
    SEED = 42
    set_seed(SEED)
    
    print("\n" + "="*50)
    print("Starting Training with Fixed Random Seeds")
    print(f"Using random seed: {SEED}")
    print("="*50 + "\n")
    
    # Clean up previous runs
    cleanup_previous_runs()
    
    # ... (rest of the code remains the same)
    print("\n" + "="*50)
    print("STARTING 2-KPI TRAINING PHASE")
    print("="*50)
    
    model_2kpi = main_2kpi_training(
        num_buildings=NUM_BUILDINGS_MAIN,
        timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN
    )
    
    # After 2-KPI training completes, check if multi-KPI training is enabled
    if ENABLE_MULTI_KPI_TRAINING:
        print("\n" + "=" * 50)
        print("MULTI-KPI TRAINING")
        print("=" * 50 + "\n")
        
        try:
            model_multi_kpi = main_multi_kpi_training(
                num_buildings=NUM_BUILDINGS_MAIN,
                timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN
            )
            
            # Save the final multi-KPI model if training was successful
            model_multi_kpi.save(os.path.join('models', 'ppo_multi_kpi_model'))
            print("Multi-KPI Training finished. Model saved to models/ppo_multi_kpi_model.zip")
        except NameError:
            print("Skipping Multi-KPI training: main_multi_kpi_training function not implemented.")
        except Exception as e:
            print(f"Error during Multi-KPI training: {str(e)}")
    else:
        print("\nMulti-KPI training is disabled. Set ENABLE_MULTI_KPI_TRAINING = True to enable it.")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print("\nAll training phases complete.")
