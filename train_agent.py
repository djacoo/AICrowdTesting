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
import numpy as np # Still needed for CustomCityEnv's dummy data generation, and potentially by SB3 or eval.
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Assuming AICrowdControl.py and CustomCityEnv.py are in the same directory or PYTHONPATH
from AICrowdControl import PhaseWeights, BASELINE_KPIS # BASELINE_KPIS is used by CustomCityEnv
from CustomCityEnv import CustomCityEnv

# --- Constants for training phases (can be overridden if needed) ---
TOTAL_TIMESTEPS_2KPI = 30000
TOTAL_TIMESTEPS_MULTI = 40000
MODEL_SAVE_PATH_2KPI_BASE = "ppo_2kpi_model"
MODEL_SAVE_PATH_MULTI_BASE = "ppo_multi_kpi_model"
TENSORBOARD_LOG_PATH_2KPI_BASE = "./ppo_tensorboard_logs_2kpi/"
TENSORBOARD_LOG_PATH_MULTI_BASE = "./ppo_tensorboard_logs_multi_kpi/"
EVAL_RESULTS_CSV_2KPI = 'evaluation_results_2kpi.csv'
EVAL_RESULTS_CSV_MULTI = 'evaluation_results_multi_kpi.csv'

def main_2kpi_training(num_buildings, timesteps_per_episode):
    """Trains a PPO agent focusing on 2 KPIs (comfort and emissions)."""
    print("\n--- Starting 2-KPI Training Phase ---")

    # --- Configuration ---
    comfort_emissions_weights = PhaseWeights(w1=0.5, w2=0.5, w3=0.0, w4=0.0)

    # Ensure TensorBoard log directory exists
    os.makedirs(TENSORBOARD_LOG_PATH_2KPI_BASE, exist_ok=True)

    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for 2-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_2kpi = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )

    # --- Agent Training (2-KPIs) ---
    print(f"Instantiating PPO agent for 2-KPI training (TensorBoard logs: {TENSORBOARD_LOG_PATH_2KPI_BASE})...")
    model_2kpi = PPO(
        "MlpPolicy",
        env_2kpi,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_PATH_2KPI_BASE
    )

    print(f"Starting 2-KPI training for {TOTAL_TIMESTEPS_2KPI} timesteps...")
    # Note: TOTAL_TIMESTEPS is set for demonstration. For robust convergence and
    # potentially higher scores, significantly more timesteps would likely be required.
    model_2kpi.learn(
        total_timesteps=TOTAL_TIMESTEPS_2KPI,
        progress_bar=True
    )

    model_save_path = MODEL_SAVE_PATH_2KPI_BASE + ".zip"
    print(f"2-KPI Training finished. Saving model to {model_save_path} ...")
    model_2kpi.save(MODEL_SAVE_PATH_2KPI_BASE)

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


def main_multi_kpi_training(num_buildings, timesteps_per_episode):
    """Continues training the agent with multiple KPIs, including grid impact."""
    print("\n--- Starting Multi-KPI Training Phase ---")

    # --- Configuration ---
    multi_kpi_weights = PhaseWeights(w1=0.3, w2=0.3, w3=0.4, w4=0.0) # Example: added grid impact

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

    policy_2kpi = PPO.load(path_to_2kpi_model).policy

    print(f"Instantiating new PPO agent for Multi-KPI training with loaded policy (TensorBoard logs: {TENSORBOARD_LOG_PATH_MULTI_BASE})...")
    model_multi_kpi = PPO(
        policy="MlpPolicy", # policy type
        env=env_multi_kpi,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_PATH_MULTI_BASE
    )
    model_multi_kpi.policy = policy_2kpi # Assign the loaded policy weights

    # --- Agent Training (Multi-KPIs) ---
    print(f"Starting Multi-KPI training for {TOTAL_TIMESTEPS_MULTI} timesteps...")
    # Note: TOTAL_TIMESTEPS is set for demonstration. For robust convergence and
    # potentially higher scores, significantly more timesteps would likely be required.
    # reset_num_timesteps=True is default and appropriate here since it's a new PPO instance for logging.
    model_multi_kpi.learn(
        total_timesteps=TOTAL_TIMESTEPS_MULTI,
        progress_bar=True
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


if __name__ == "__main__":
    NUM_BUILDINGS_MAIN = 2
    TIMESTEPS_PER_EPISODE_MAIN = 24 * 7 # One week of hourly data

    # Run 2-KPI Training Phase
    main_2kpi_training(num_buildings=NUM_BUILDINGS_MAIN, timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN)

    # Run Multi-KPI Training Phase, continuing from the 2-KPI model
    main_multi_kpi_training(num_buildings=NUM_BUILDINGS_MAIN, timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN)

    print("\nAll training phases complete.")
