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
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch  # For neural network operations
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

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

# --- Constants for training phases (can be overridden if needed) ---
TOTAL_TIMESTEPS_2KPI = 100000  # Increased from 30k to 100k
TOTAL_TIMESTEPS_MULTI = 150000  # Increased from 40k to 150k
EVAL_FREQ = 5000  # Evaluate every 5k steps
MODEL_SAVE_PATH_2KPI_BASE = "ppo_2kpi_model"
MODEL_SAVE_PATH_MULTI_BASE = "ppo_multi_kpi_model"
TENSORBOARD_LOG_PATH_2KPI_BASE = "./ppo_tensorboard_logs_2kpi/"
TENSORBOARD_LOG_PATH_MULTI_BASE = "./ppo_tensorboard_logs_multi_kpi/"
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
    """Generate training and KPI visualisations using a unified style."""
    try:
        import glob
        import pandas as pd
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        def load_rewards(log_base):
            """Load reward history from all runs under a directory."""
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
            return None

        # Load reward histories
        history_2kpi = load_rewards(TENSORBOARD_LOG_PATH_2KPI_BASE)
        history_multi = load_rewards(TENSORBOARD_LOG_PATH_MULTI_BASE)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

        for df, label, color in [
            (history_2kpi, '2-KPI Training', 'green'),
            (history_multi, 'Multi-KPI Training', 'orange')
        ]:
            if df is None:
                continue
            df.columns = [f'Run {i+1}' for i in range(df.shape[1])]
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            smoothed = mean.rolling(window=window_size, min_periods=1).mean()

            ax1.plot(smoothed, linewidth=2, label=label, color=color)
            ax1.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.1, label=f'{label} ±1 std')

        ax1.set_title('Training Reward')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)

        # --- KPI Metrics ---
        eval_dfs = []
        for csv_file, label, color in [
            (EVAL_RESULTS_CSV_2KPI, '2-KPI Eval', 'green'),
            (EVAL_RESULTS_CSV_MULTI, 'Multi-KPI Eval', 'orange')
        ]:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if not df.empty and 'episode_reward' in df.columns:
                    eval_dfs.append((df, label, color))

        if eval_dfs:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
            kpi_cols = [c for c in eval_dfs[0][0].columns if c.startswith('KPI_')]
            if kpi_cols:
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
                ax2.set_ylabel('KPI Value')
                ax2.set_title('Key Performance Indicators')
                ax2.legend()
                ax2.grid(True, axis='y')
            fig2.tight_layout()
            fig2.savefig('training_kpis.png', dpi=300, bbox_inches='tight')

        fig.tight_layout()
        fig.savefig('training_rewards.png', dpi=300, bbox_inches='tight')

    except ImportError as e:
        print(f'Could not import required libraries for plotting: {e}')
    except Exception as e:
        import traceback
        print(f'Error generating plots: {str(e)}\n{traceback.format_exc()}')

def main_2kpi_training(num_buildings, timesteps_per_episode):
    """Trains a PPO agent focusing on 2 KPIs (comfort and emissions)."""
    print("\n--- Starting 2-KPI Training Phase ---")

    # --- Configuration ---
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

    # --- Agent Training (2-KPIs) ---
    print(f"Instantiating PPO agent for 2-KPI training (TensorBoard logs: {TENSORBOARD_LOG_PATH_2KPI_BASE})...")
    
    # Initialize PPO agent with enhanced configuration for 2-KPI training
    policy_kwargs = {
        'net_arch': dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Deeper network
        'activation_fn': torch.nn.ReLU,  # ReLU for deeper networks
        'ortho_init': True,
        'log_std_init': -0.5,  # Slightly higher initial std for better exploration
        'optimizer_kwargs': {
            'eps': 1e-8,  # Increased numerical stability
            'weight_decay': 1e-6  # Small weight decay for better generalization
        }
    }
    
    # Learning rate schedule with warmup and cosine decay
    def lr_schedule(progress_remaining):
        """Cosine decay with warmup for learning rate with longer warmup"""
        # Linear warmup for first 20% of training
        warmup_frac = 0.2
        if progress_remaining > 1.0 - warmup_frac:
            warmup_progress = (1.0 - progress_remaining) / warmup_frac
            return 5e-4 * warmup_progress  # Higher initial learning rate
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 1e-5
        max_lr = 5e-4  # Higher max learning rate
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    # Initialize the model with optimized hyperparameters
    model_2kpi = PPO(
        "MlpPolicy",
        env_2kpi,
        learning_rate=lr_schedule,
        n_steps=4096,           # More steps per update for better gradient estimates
        batch_size=512,         # Larger batch size for more stable updates
        n_epochs=10,            # More epochs for better optimization
        gamma=0.995,            # Slightly higher discount factor for longer-term rewards
        gae_lambda=0.97,        # Adjusted for better bias-variance tradeoff
        clip_range=0.2,         # Slightly higher clip range for more exploration
        clip_range_vf=0.2,      # Add clipping for value function
        ent_coef=0.02,          # Slightly higher entropy for better exploration
        vf_coef=0.7,            # Higher weight on value function loss
        max_grad_norm=0.8,      # Increased gradient clipping for stability
        use_sde=False,          # Disable SDE for discrete action spaces
        target_kl=0.03,         # Slightly higher KL threshold
        tensorboard_log=TENSORBOARD_LOG_PATH_2KPI_BASE,
        policy_kwargs=policy_kwargs,
        verbose=2,              # More detailed logging
        device='auto',
        normalize_advantage=True,
        seed=42
    )

    print(f"Starting 2-KPI training for {TOTAL_TIMESTEPS_2KPI} timesteps...")
    
    # Setup evaluation environment
    eval_env = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_2kpi',
        log_path='./logs/2kpi_eval',
        eval_freq=max(10000 // num_buildings, 1),  # Evaluate more frequently for fewer buildings
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Evaluate on 5 episodes for more stable metrics
    )
    
    # Train the model with evaluation callback
    model_2kpi.learn(
        total_timesteps=TOTAL_TIMESTEPS_2KPI,
        callback=eval_callback,
        progress_bar=True,
        tb_log_name="PPO_2KPI",
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

    # --- Evaluation/Logging for 2-KPI Model ---
    print("\n--- Evaluating 2-KPI Trained Model ---")
    loaded_model = PPO.load(model_save_path)

    eval_env_2kpi = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )

    eval_episodes = 5
    episode_rewards_log = []
    all_episode_kpis_log = []

    for i in range(eval_episodes):
        obs = eval_env_2kpi.reset()
        done = False
        current_episode_reward = 0
        print(f"Starting 2-KPI Evaluation Episode {i+1}/{eval_episodes}")

        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env_2kpi.step(action)
            current_episode_reward += reward

        print(f"2-KPI Evaluation Episode {i+1} Reward: {current_episode_reward}")
        episode_rewards_log.append(current_episode_reward)
        if 'episode_kpis' in info and info['episode_kpis']:
            all_episode_kpis_log.append(info['episode_kpis'])
        else:
            all_episode_kpis_log.append({'error': 'missing_kpis'})

    if episode_rewards_log:
        avg_reward = sum(episode_rewards_log) / len(episode_rewards_log) if episode_rewards_log else 0
        print(f"Average Reward (2-KPI) over {eval_episodes} episodes: {avg_reward:.4f}")

    if all_episode_kpis_log:
        eval_df = pd.DataFrame(all_episode_kpis_log)
        eval_df['episode_reward'] = episode_rewards_log
        eval_df.to_csv(EVAL_RESULTS_CSV_2KPI, index=False)
        print(f"2-KPI Evaluation results saved to {EVAL_RESULTS_CSV_2KPI}")

    eval_env_2kpi.close()
    print("2-KPI Evaluation environment closed.")
    print("--- 2-KPI Training and Evaluation Phase Finished ---")

    # Plot training and evaluation results
    plot_training_results()

def main_multi_kpi_training(num_buildings, timesteps_per_episode):
    """Continues training the agent with multiple KPIs, including grid impact."""
    print("\n--- Starting Multi-KPI Training Phase ---")

    # --- Configuration ---
    # Adjusted weights for better balance between comfort, emissions, and grid impact
    multi_kpi_weights = PhaseWeights(w1=0.4, w2=0.3, w3=0.3, w4=0.0)  # Slightly higher weight on comfort

    # Ensure TensorBoard log directory exists
    os.makedirs(TENSORBOARD_LOG_PATH_MULTI_BASE, exist_ok=True)

    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for Multi-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_multi_kpi = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )

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
        'net_arch': dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Deeper network
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
            return 4e-4 * warmup_progress  # Slightly lower initial learning rate than 2-KPI
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 5e-6  # Slightly lower minimum learning rate for fine-tuning
        max_lr = 4e-4
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    # Initialize the model with optimized hyperparameters for multi-KPI training
    model_multi_kpi = PPO(
        "MlpPolicy",
        env_multi_kpi,
        learning_rate=lr_schedule,
        n_steps=4096,           # More steps per update for better gradient estimates
        batch_size=512,         # Larger batch size for more stable updates
        n_epochs=12,            # More epochs for better optimization
        gamma=0.995,            # Slightly higher discount factor for longer-term rewards
        gae_lambda=0.97,        # Adjusted for better bias-variance tradeoff
        clip_range=0.18,        # Slightly higher clip range for more exploration
        clip_range_vf=0.18,     # Add clipping for value function
        ent_coef=0.015,         # Slightly lower entropy for more focused learning
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
    model_multi_kpi.save(MODEL_SAVE_PATH_MULTI_BASE)

    env_multi_kpi.close() # Close training env
    print("Multi-KPI Training environment closed.")

    # --- Evaluation/Logging for Multi-KPI Model ---
    print("\n--- Evaluating Multi-KPI Trained Model ---")
    loaded_model_multi = PPO.load(model_save_path)

    # Use a fresh environment instance for evaluation
    eval_env_multi_kpi = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )

    eval_episodes = 5
    episode_rewards_log_multi = []
    all_episode_kpis_log_multi = []

    for i in range(eval_episodes):
        obs = eval_env_multi_kpi.reset()
        done = False
        current_episode_reward = 0
        print(f"Starting Multi-KPI Evaluation Episode {i+1}/{eval_episodes}")

        while not done:
            action, _states = loaded_model_multi.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env_multi_kpi.step(action)
            current_episode_reward += reward

        print(f"Multi-KPI Evaluation Episode {i+1} Reward: {current_episode_reward}")
        episode_rewards_log_multi.append(current_episode_reward)
        if 'episode_kpis' in info and info['episode_kpis']:
            all_episode_kpis_log_multi.append(info['episode_kpis'])
        else:
            all_episode_kpis_log_multi.append({'error': 'missing_kpis'})

    if episode_rewards_log_multi:
        avg_reward_multi = sum(episode_rewards_log_multi) / len(episode_rewards_log_multi) if episode_rewards_log_multi else 0
        print(f"Average Reward (Multi-KPI) over {eval_episodes} episodes: {avg_reward_multi:.4f}")

    if all_episode_kpis_log_multi:
        eval_df_multi = pd.DataFrame(all_episode_kpis_log_multi)
        eval_df_multi['episode_reward'] = episode_rewards_log_multi
        eval_df_multi.to_csv(EVAL_RESULTS_CSV_MULTI, index=False)
        print(f"Multi-KPI Evaluation results saved to {EVAL_RESULTS_CSV_MULTI}")

    eval_env_multi_kpi.close()
    print("Multi-KPI Evaluation environment closed.")
    print("--- Multi-KPI Training and Evaluation Phase Finished ---")

    # Plot training and evaluation results
    plot_training_results()

if __name__ == "__main__":
    NUM_BUILDINGS_MAIN = 2
    TIMESTEPS_PER_EPISODE_MAIN = 24 * 7 # One week of hourly data

    # Run 2-KPI Training Phase
    main_2kpi_training(num_buildings=NUM_BUILDINGS_MAIN, timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN)

    # Run Multi-KPI Training Phase, continuing from the 2-KPI model
    main_multi_kpi_training(num_buildings=NUM_BUILDINGS_MAIN, timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN)

    print("\nAll training phases complete.")
