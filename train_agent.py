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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch  # For neural network operations
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

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

def plot_training_results(window_size=10):
    """
    Plot training progress with smoothed curves and confidence intervals.
    
    Args:
        window_size: Size of the moving average window for smoothing
    """
    try:
        import glob
        import pandas as pd
        import seaborn as sns
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Training Curves ---
        print("\nGenerating training curves...")
        
        for log_dir, label, color in [
            (TENSORBOARD_LOG_PATH_2KPI_BASE, "2-KPI Training", "#1f77b4"),
            (TENSORBOARD_LOG_PATH_MULTI_BASE, "Multi-KPI Training", "#ff7f0e")
        ]:
            event_files = []
            event_files.extend(glob.glob(f"{log_dir}*/events.out.tfevents.*"))
            event_files.extend(glob.glob(f"{log_dir}events.out.tfevents.*"))
            
            if not event_files:
                print(f"No event files found for {label}")
                continue
                
            try:
                latest_event_file = max(event_files, key=os.path.getmtime)
                print(f"Processing {label} data from: {os.path.basename(latest_event_file)}")
                
                event_acc = EventAccumulator(os.path.dirname(latest_event_file) if os.path.isdir(os.path.dirname(latest_event_file)) else log_dir)
                event_acc.Reload()
                
                if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
                    events = event_acc.Scalars('rollout/ep_rew_mean')
                    steps = np.array([e.step for e in events])
                    values = np.array([e.value for e in events])
                    
                    if len(values) > 0:
                        # Apply smoothing
                        smoothed_values, rolling_std = smooth_data(values, window_size=window_size)
                        
                        # Plot main line with confidence interval
                        ax1.plot(steps, smoothed_values, label=label, color=color, linewidth=2)
                        ax1.fill_between(steps, 
                                       smoothed_values - rolling_std, 
                                       smoothed_values + rolling_std,
                                       color=color, alpha=0.15, linewidth=0)
                        
                        # Add start and end markers
                        ax1.scatter(steps[0], smoothed_values[0], color='green', marker='^', 
                                 s=100, zorder=5, label=f"{label} Start" if label == "2-KPI Training" else "")
                        ax1.scatter(steps[-1], smoothed_values[-1], color='red', marker='v', 
                                 s=100, zorder=5, label=f"{label} End" if label == "Multi-KPI Training" else "")
                        
            except Exception as e:
                print(f"Error processing {label} data: {str(e)}")
        
        # Format training plot
        ax1.set_title('Training Progress with Confidence Intervals', fontsize=14, pad=20)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.2)
        
        # --- KPI Metrics ---
        print("\nGenerating KPI metrics...")
        eval_dfs = []
        for csv_file, label, color in [
            (EVAL_RESULTS_CSV_2KPI, "2-KPI Eval", "#1f77b4"),
            (EVAL_RESULTS_CSV_MULTI, "Multi-KPI Eval", "#ff7f0e")
        ]:
            try:
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    if not df.empty and 'episode_reward' in df.columns:
                        eval_dfs.append((df, label, color))
                        print(f"Processed evaluation data from {os.path.basename(csv_file)}: {len(df)} episodes")
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
        
        if eval_dfs:
            kpi_data = []
            kpi_cols = [col for col in eval_dfs[0][0].columns if col.startswith('KPI_')]
            
            if kpi_cols:
                kpi_names = [k[4:].replace('_', ' ').title() for k in kpi_cols]
                x = np.arange(len(kpi_names))
                width = 0.35 if len(eval_dfs) > 1 else 0.7
                
                for i, (df, label, color) in enumerate(eval_dfs):
                    kpi_means = df[kpi_cols].mean().values
                    kpi_stds = df[kpi_cols].std().values
                    
                    # Plot bars with error bars
                    offset = width * (i - (len(eval_dfs)-1)/2)
                    ax2.bar(x + offset, kpi_means, width, yerr=kpi_stds,
                           label=label, color=color, alpha=0.7, capsize=5,
                           error_kw=dict(elinewidth=1.5, ecolor='black', capthick=1.5))
                    
                    # Add value labels
                    for j, v in enumerate(kpi_means):
                        ax2.text(x[j] + offset, v + 0.1, f"{v:.2f}", 
                               ha='center', va='bottom', fontsize=9)
                
                ax2.set_xticks(x)
                ax2.set_xticklabels(kpi_names, rotation=45, ha='right')
                ax2.set_ylabel('KPI Value', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.set_title('Key Performance Indicators', fontsize=14, pad=20)
                ax2.grid(True, alpha=0.2, axis='y')
        
        plt.tight_layout()
        
        # Save the combined plot
        plot_path = "training_results_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined training results plot saved to {os.path.abspath(plot_path)}")
        
        # Save individual plots
        fig1, ax = plt.subplots(figsize=(14, 6))
        for line in ax1.lines + ax1.collections:
            if hasattr(line, 'get_label') and line.get_label():
                if 'Start' in line.get_label() or 'End' in line.get_label():
                    ax.scatter([], [], color=line.get_color(), label=line.get_label(), 
                             marker='^' if 'Start' in line.get_label() else 'v', s=100)
                else:
                    ax.add_line(line)
                    if isinstance(line, plt.Line2D):
                        ax.fill_between(line.get_xdata(), 
                                      line.get_ydata() - rolling_std, 
                                      line.get_ydata() + rolling_std,
                                      color=line.get_color(), alpha=0.15, linewidth=0)
        
        ax.set_title('Training Progress with Confidence Intervals', fontsize=14, pad=20)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
        
        print("\nVisualization complete!")
        
    except ImportError as e:
        print(f"Could not import required libraries for plotting: {e}")
    except Exception as e:
        import traceback
        print(f"Error generating plots: {str(e)}\n{traceback.format_exc()}")

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
        'net_arch': dict(pi=[256, 256], vf=[256, 256]),
        'activation_fn': torch.nn.Tanh,
        'ortho_init': True,
        'log_std_init': -1.0,  # Adjusted for better exploration
        'optimizer_kwargs': {
            'eps': 1e-5,  # For numerical stability
            'weight_decay': 0.0  # Disabled for now to prevent interference
        }
    }
    
    # Learning rate schedule with warmup and cosine decay
    def lr_schedule(progress_remaining):
        """Cosine decay with warmup for learning rate"""
        # Linear warmup for first 10% of training
        warmup_frac = 0.1
        if progress_remaining > 1.0 - warmup_frac:
            warmup_progress = (1.0 - progress_remaining) / warmup_frac
            return 3e-4 * warmup_progress
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 1e-5
        max_lr = 3e-4
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    # Initialize the model with optimized hyperparameters
    model_2kpi = PPO(
        "MlpPolicy",
        env_2kpi,
        learning_rate=lr_schedule,
        n_steps=2048,           # Number of steps to run for each environment per update
        batch_size=256,         # Minibatch size for each epoch
        n_epochs=8,             # Number of epochs for optimization
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,         # Clipping parameter for the policy
        clip_range_vf=None,     # Clipping parameter for the value function
        ent_coef=0.01,          # Entropy coefficient for exploration
        vf_coef=0.5,            # Value function coefficient for the loss
        max_grad_norm=0.5,      # Maximum gradient norm for gradient clipping
        use_sde=False,          # Whether to use State Dependent Exploration
        sde_sample_freq=-1,     # Sample a new noise matrix every n steps when using gSDE
        target_kl=0.02,         # Target KL divergence threshold
        tensorboard_log=TENSORBOARD_LOG_PATH_2KPI_BASE,
        policy_kwargs=policy_kwargs,
        verbose=1,              # Verbosity level
        device='auto',          # Device to run on (auto, cpu, cuda, etc.)
        normalize_advantage=True,  # Whether to normalize the advantage
        seed=42                  # For reproducibility
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

    # Check if the 2-KPI model exists before trying to load its policy
    if not os.path.exists(path_to_2kpi_model):
        print(f"Error: 2-KPI model not found at {path_to_2kpi_model}. Skipping Multi-KPI training.")
        env_multi_kpi.close()
        return

    # First create the environment to get the correct observation space
    print(f"Setting up environment to get observation space...")
    temp_env = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    obs_shape = temp_env.observation_space.shape[0]
    temp_env.close()
    
    # Create a new model with the correct observation space
    print("Instantiating new PPO agent for Multi-KPI training "
          f"(TensorBoard logs: {TENSORBOARD_LOG_PATH_MULTI_BASE}/)...")
    
    # Initialize PPO agent with enhanced configuration for multi-KPI training
    policy_kwargs = {
        'net_arch': dict(pi=[512, 512], vf=[512, 512]),  # Larger network for more complex task
        'activation_fn': torch.nn.ReLU,  # ReLU for potentially better gradient flow
        'ortho_init': True,
        'log_std_init': -0.8,  # Balanced exploration/exploitation
        'optimizer_kwargs': {
            'eps': 1e-5,  # For numerical stability
            'weight_decay': 0.0  # No weight decay for stability
        }
    }
    
    # Learning rate schedule with warmup and cosine decay for multi-KPI
    def lr_schedule(progress_remaining):
        """Cosine decay with warmup for learning rate"""
        # Linear warmup for first 15% of training (longer warmup for more complex task)
        warmup_frac = 0.15
        if progress_remaining > 1.0 - warmup_frac:
            warmup_progress = (1.0 - progress_remaining) / warmup_frac
            return 2.5e-4 * warmup_progress
            
        # Cosine decay for the rest
        progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_frac)
        min_lr = 5e-6  # Lower minimum learning rate for fine-tuning
        max_lr = 2.5e-4
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    model_multi_kpi = PPO(
        "MlpPolicy",
        env_multi_kpi,
        learning_rate=lr_schedule,
        n_steps=4096,           # More steps per update for better advantage estimation
        batch_size=1024,        # Larger batch size for stability
        n_epochs=10,            # More epochs for better learning
        gamma=0.99,             # Standard discount factor
        gae_lambda=0.95,        # Standard GAE parameter for good bias-variance tradeoff
        clip_range=0.15,        # Slightly lower clip range for more stable updates
        clip_range_vf=None,     # No separate clip range for value function
        ent_coef=0.01,          # Lower entropy for more exploitation
        vf_coef=0.5,            # Standard value function coefficient
        max_grad_norm=0.5,      # Conservative gradient clipping
        use_sde=False,          # No state-dependent exploration
        sde_sample_freq=-1,     # Disabled
        target_kl=0.01,         # Tighter KL constraint for stable updates
        tensorboard_log=TENSORBOARD_LOG_PATH_MULTI_BASE,
        policy_kwargs=policy_kwargs,  # Use the policy_kwargs defined above
        verbose=1,              # Verbosity level
        device='auto',          # Device to run on (auto, cpu, cuda, etc.)
        normalize_advantage=True,  # Normalize advantages
        seed=42,                # For reproducibility
        _init_setup_model=True  # Initialize model immediately
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
