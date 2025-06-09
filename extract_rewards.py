import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

# Ensure the subtask can install packages if they are missing
try:
    import tensorboard
except ImportError:
    print("Installing tensorboard...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "tensorboard"])
        print("Tensorboard installed.")
    except Exception as e:
        print(f"Pip install failed: {e}. Assuming tensorboard is already available.")
    # Re-import after potential installation
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_rewards_from_event_file(event_file_path, scalar_tag='rollout/ep_rew_mean'):
    '''Loads scalar data from a TensorBoard event file.'''
    if not os.path.exists(event_file_path):
        print(f"Event file not found: {event_file_path}")
        return None

    acc = EventAccumulator(event_file_path)
    acc.Reload()

    if scalar_tag not in acc.Tags()['scalars']:
        print(f"Tag '{scalar_tag}' not found in {event_file_path}. Available tags: {acc.Tags()['scalars']}")
        return None

    events = acc.Scalars(scalar_tag)
    steps = [e.step for e in events]
    rewards = [e.value for e in events]

    if not steps:
        print(f"No data found for tag '{scalar_tag}' in {event_file_path}")
        return None

    return list(zip(steps, rewards))

# Define file paths
event_file_2kpi = "ppo_tensorboard_logs_2kpi/PPO_2KPI_1/events.out.tfevents.1749480879.MacBookPro.home-life.hub.182.0"
event_file_multi_kpi = "ppo_tensorboard_logs_multi_kpi/PPO_MultiKPI_0/events.out.tfevents.1749481034.MacBookPro.home-life.hub.182.1"
output_txt_file = "extracted_rewards.txt"

rewards_2kpi = load_rewards_from_event_file(event_file_2kpi)
rewards_multi_kpi = load_rewards_from_event_file(event_file_multi_kpi)

with open(output_txt_file, "w") as f:
    f.write("2-KPI Training Rewards (rollout/ep_rew_mean):\n")
    if rewards_2kpi:
        f.write(f"Total data points: {len(rewards_2kpi)}\n")
        f.write("First 5 data points:\n")
        for step, reward in rewards_2kpi[:5]:
            f.write(f"Step: {step}, Reward: {reward:.4f}\n")
        f.write("Last 5 data points:\n")
        for step, reward in rewards_2kpi[-5:]:
            f.write(f"Step: {step}, Reward: {reward:.4f}\n")
        # For detailed analysis, write all points, but this could be long.
        # f.write("All data points:\n")
        # for step, reward in rewards_2kpi:
        #     f.write(f"Step: {step}, Reward: {reward:.4f}\n")
    else:
        f.write("No data found.\n")

    f.write("\nMulti-KPI Training Rewards (rollout/ep_rew_mean):\n")
    if rewards_multi_kpi:
        f.write(f"Total data points: {len(rewards_multi_kpi)}\n")
        f.write("First 5 data points:\n")
        for step, reward in rewards_multi_kpi[:5]:
            f.write(f"Step: {step}, Reward: {reward:.4f}\n")
        f.write("Last 5 data points:\n")
        for step, reward in rewards_multi_kpi[-5:]:
            f.write(f"Step: {step}, Reward: {reward:.4f}\n")
        # f.write("All data points:\n")
        # for step, reward in rewards_multi_kpi:
        #     f.write(f"Step: {step}, Reward: {reward:.4f}\n")
    else:
        f.write("No data found.\n")

print(f"Reward data extracted and saved to {output_txt_file}")
