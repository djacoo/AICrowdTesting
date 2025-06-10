#!/usr/bin/env python3
"""
Training Rewards Visualization Tool

This script helps visualize the training progress by plotting the episode rewards
from TensorBoard log files for both 2-KPI and multi-KPI training.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Default directories
DEFAULT_2KPI_LOG_DIR = "ppo_tensorboard_logs_2kpi"
DEFAULT_MULTI_KPI_LOG_DIR = "ppo_tensorboard_logs_multi_kpi"

def find_latest_event_file(log_dir):
    """Find the most recent event file in the given directory."""
    event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
    if not event_files:
        return None
    return max(event_files, key=lambda x: x.stat().st_mtime)

def load_training_data(log_dir, metric_tags=None):
    """
    Load training data from TensorBoard event files.
    
    Args:
        log_dir: Directory containing the TensorBoard logs
        metric_tags: List of metric tags to extract (None for all)
    """
    event_file = find_latest_event_file(log_dir)
    if not event_file:
        print(f"No event files found in {log_dir}")
        return None, {}
    
    print(f"\nLoading training data from: {event_file}")
    
    try:
        # Load the event file
        acc = EventAccumulator(str(event_file.parent))
        acc.Reload()
        
        # Get available tags
        available_tags = acc.Tags()['scalars']
        print(f"Available metrics: {', '.join(available_tags)}")
        
        # If specific tags are requested, filter them
        tags_to_load = metric_tags if metric_tags else available_tags
        
        data = {}
        for tag in tags_to_load:
            if tag in available_tags:
                events = acc.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[tag] = pd.DataFrame({'step': steps, tag: values})
                print(f"Loaded {len(values)} data points for {tag}")
            else:
                print(f"Warning: Metric '{tag}' not found in the event file")
        
        return event_file.parent, data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

def plot_training_comparison(log_dirs, labels, metric='rollout/ep_rew_mean'):
    """
    Plot comparison of training metrics across different runs.
    
    Args:
        log_dirs: List of log directories to compare
        labels: List of labels for each log directory
        metric: Metric to plot (default: 'rollout/ep_rew_mean')
    """
    plt.figure(figsize=(14, 8))
    
    for log_dir, label in zip(log_dirs, labels):
        _, data = load_training_data(log_dir, [metric])
        if not data or metric not in data:
            print(f"Skipping {log_dir} - no data for {metric}")
            continue
            
        df = data[metric]
        # Calculate moving average
        window = max(1, len(df) // 20)
        df[f'{metric}_ma'] = df[metric].rolling(window=window, min_periods=1).mean()
        
        # Plot
        plt.plot(df['step'], df[metric], alpha=0.2, label=f'{label} (raw)')
        plt.plot(df['step'], df[f'{metric}_ma'], linewidth=2, 
                label=f'{label} (MA {window} steps)')
    
    # Formatting
    plt.title(f'Training Comparison: {metric}', fontsize=16, pad=20)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Improve x-axis formatting
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else int(x))
    )
    
    plt.tight_layout()
    return plt.gcf()

def plot_kpi_metrics(log_dir, title_suffix=""):
    """
    Plot all available KPI metrics from the training run.
    """
    _, data = load_training_data(log_dir)
    if not data:
        return None
    
    # Filter for KPI-related metrics
    kpi_metrics = [k for k in data.keys() if any(x in k.lower() for x in ['kpi', 'reward', 'loss'])]
    if not kpi_metrics:
        print("No KPI metrics found")
        return None
    
    # Create subplots
    n_metrics = len(kpi_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(kpi_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        df = data[metric]
        
        # Calculate moving average
        window = max(1, len(df) // 20)
        df[f'{metric}_ma'] = df[metric].rolling(window=window, min_periods=1).mean()
        
        # Plot
        ax.plot(df['step'], df[metric], alpha=0.3, label='Raw')
        ax.plot(df['step'], df[f'{metric}_ma'], linewidth=2, label=f'MA {window} steps')
        
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else int(x))
        )
    
    # Remove any extra subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Training Metrics {title_suffix}'.strip(), fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def save_plot(fig, filename, output_dir='plots'):
    """Save the plot to a file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"\nPlot saved to: {os.path.abspath(filepath)}")
    return filepath
    
def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from TensorBoard logs')
    parser.add_argument('--log-dir-2kpi', type=str, default=DEFAULT_2KPI_LOG_DIR,
                       help=f'Directory containing 2-KPI TensorBoard logs (default: {DEFAULT_2KPI_LOG_DIR})')
    parser.add_argument('--log-dir-multi', type=str, default=DEFAULT_MULTI_KPI_LOG_DIR,
                       help=f'Directory containing multi-KPI TensorBoard logs (default: {DEFAULT_MULTI_KPI_LOG_DIR})')
    parser.add_argument('--output-dir', type=str, default='training_plots',
                       help='Directory to save output plots (default: training_plots)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison plots between 2-KPI and multi-KPI training')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot individual training runs
    if os.path.exists(args.log_dir_2kpi):
        print("\n" + "="*50)
        print("PLOTTING 2-KPI TRAINING METRICS")
        print("="*50)
        fig = plot_kpi_metrics(args.log_dir_2kpi, "(2-KPI Training)")
        if fig:
            save_plot(fig, f"2kpi_metrics_{timestamp}.png", args.output_dir)
    else:
        print(f"\n2-KPI log directory not found: {args.log_dir_2kpi}")
    
    if os.path.exists(args.log_dir_multi):
        print("\n" + "="*50)
        print("PLOTTING MULTI-KPI TRAINING METRICS")
        print("="*50)
        fig = plot_kpi_metrics(args.log_dir_multi, "(Multi-KPI Training)")
        if fig:
            save_plot(fig, f"multi_kpi_metrics_{timestamp}.png", args.output_dir)
    else:
        print(f"\nMulti-KPI log directory not found: {args.log_dir_multi}")
    
    # Generate comparison plots if requested and both log directories exist
    if args.compare and os.path.exists(args.log_dir_2kpi) and os.path.exists(args.log_dir_multi):
        print("\n" + "="*50)
        print("GENERATING COMPARISON PLOTS")
        print("="*50)
        
        # Compare episode rewards
        fig = plot_training_comparison(
            [args.log_dir_2kpi, args.log_dir_multi],
            ['2-KPI', 'Multi-KPI'],
            'rollout/ep_rew_mean'
        )
        save_plot(fig, f"reward_comparison_{timestamp}.png", args.output_dir)
        
        # Compare episode lengths if available
        try:
            fig = plot_training_comparison(
                [args.log_dir_2kpi, args.log_dir_multi],
                ['2-KPI', 'Multi-KPI'],
                'rollout/ep_len_mean'
            )
            save_plot(fig, f"episode_length_comparison_{timestamp}.png", args.output_dir)
        except:
            print("Could not generate episode length comparison - data not available")
    
    print("\n" + "="*50)
    print("PLOTTING COMPLETE")
    print("="*50)
    print(f"\nTo view the TensorBoard logs, run:")
    if os.path.exists(args.log_dir_2kpi):
        print(f"  2-KPI:    tensorboard --logdir={os.path.abspath(args.log_dir_2kpi)}")
    if os.path.exists(args.log_dir_multi):
        print(f"  Multi-KPI: tensorboard --logdir={os.path.abspath(args.log_dir_multi)}")
    print(f"\nPlots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
