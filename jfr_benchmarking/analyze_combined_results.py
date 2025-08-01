#!/usr/bin/env python3
"""
Combined JFR Queue Size Analysis

Analyzes CSV results and creates combined graphs showing:
- Loss percentage vs queue size (all intervals on one graph)
- P99 queue size vs configured queue size (all intervals on one graph)
- P99.9 queue size vs configured queue size (all intervals on one graph)

Generates separate graphs for each native duration/stack depth combination and Renaissance tests.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
RESULTS_DIR = Path("benchmark_results")
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"

def load_csv_data(test_type: str = 'both') -> tuple:
    """Load CSV data for analysis"""
    native_df = None
    renaissance_df = None

    if test_type in ['native', 'both']:
        # Look for latest native CSV
        native_files = list(DATA_DIR.glob("native_*.csv"))
        if native_files:
            latest_native = max(native_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading native data: {latest_native.name}")
            native_df = pd.read_csv(latest_native)
            print(f"   {len(native_df)} native test records")
        else:
            print("⚠️ No native CSV files found")

    if test_type in ['renaissance', 'both']:
        # Look for latest renaissance CSV
        renaissance_files = list(DATA_DIR.glob("renaissance_*.csv"))
        if renaissance_files:
            latest_renaissance = max(renaissance_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading renaissance data: {latest_renaissance.name}")
            renaissance_df = pd.read_csv(latest_renaissance)
            print(f"   {len(renaissance_df)} renaissance test records")
        else:
            print("⚠️ No renaissance CSV files found")

    return native_df, renaissance_df

def load_json_data_for_queue_stats(test_type: str = 'both') -> tuple:
    """Load JSON data to extract queue size percentiles"""
    native_queue_data = {}
    renaissance_queue_data = {}

    if test_type in ['native', 'both']:
        # Look for latest native JSON
        native_files = list(DATA_DIR.glob("native_*.json"))
        if native_files:
            latest_native = max(native_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading native JSON for queue stats: {latest_native.name}")
            with open(latest_native, 'r') as f:
                native_data = json.load(f)

            # Extract queue stats
            for record in native_data:
                if 'queue_stats' in record and record['queue_stats']:
                    key = (record['queue_size'], record['interval'], record.get('native_duration'))
                    native_queue_data[key] = record['queue_stats']

            print(f"   {len(native_queue_data)} native records with queue stats")

    if test_type in ['renaissance', 'both']:
        # Look for latest renaissance JSON
        renaissance_files = list(DATA_DIR.glob("renaissance_*.json"))
        if renaissance_files:
            latest_renaissance = max(renaissance_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading renaissance JSON for queue stats: {latest_renaissance.name}")
            with open(latest_renaissance, 'r') as f:
                renaissance_data = json.load(f)

            # Extract queue stats
            for record in renaissance_data:
                if 'queue_stats' in record and record['queue_stats']:
                    key = (record['queue_size'], record['interval'])
                    renaissance_queue_data[key] = record['queue_stats']

            print(f"   {len(renaissance_queue_data)} renaissance records with queue stats")

    return native_queue_data, renaissance_queue_data

def extract_percentiles_from_queue_stats(queue_stats: Dict, target_percentiles: List[str] = ['p99', 'p99_9']) -> Dict:
    """Extract specific percentiles from queue stats"""
    result = {}

    # Look for 'all' category first, then any available category
    categories_to_try = ['all', 'gc', 'vm', 'compiler']

    for category in categories_to_try:
        if category in queue_stats:
            stats = queue_stats[category]
            for percentile in target_percentiles:
                if percentile in stats:
                    result[percentile] = stats[percentile]

            # If we found data in this category, use it
            if result:
                result['category'] = category
                break

    return result

def create_combined_loss_graphs(native_df: pd.DataFrame, renaissance_df: pd.DataFrame):
    """Create combined loss percentage graphs with all intervals on one plot"""

    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Filter successful tests only
    if native_df is not None:
        native_success = native_df[native_df['success'] == True].copy()

        # Get unique combinations of native duration and stack depth
        if 'stack_depth' in native_success.columns and 'native_duration' in native_success.columns:
            native_combinations = native_success[['native_duration', 'stack_depth']].drop_duplicates().sort_values(['native_duration', 'stack_depth'])
        else:
            # Fallback for old data without stack_depth
            native_combinations = pd.DataFrame({'native_duration': sorted(native_success['native_duration'].unique()) if 'native_duration' in native_success.columns else [None], 'stack_depth': [None]})
    else:
        native_success = pd.DataFrame()
        native_combinations = pd.DataFrame()

    if renaissance_df is not None:
        renaissance_success = renaissance_df[renaissance_df['success'] == True].copy()

        # Get unique stack depths for Renaissance
        if 'stack_depth' in renaissance_success.columns:
            renaissance_stacks = sorted(renaissance_success['stack_depth'].unique())
        else:
            renaissance_stacks = [None]
    else:
        renaissance_success = pd.DataFrame()
        renaissance_stacks = []

    # Define interval order and colors
    interval_order = ["1ms", "5ms", "10ms", "20ms"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    # 1. Native tests - separate graph for each native duration/stack depth combination
    if not native_success.empty and not native_combinations.empty:
        n_combinations = len(native_combinations)
        if n_combinations > 0:
            # Create subplots for native duration/stack combinations
            cols = min(4, n_combinations)
            rows = (n_combinations + cols - 1) // cols
            fig_width = 6 * cols
            fig_height = 5 * rows

            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            fig.suptitle('Native Tests: Loss Percentage vs Queue Size (All Intervals Combined)', fontsize=16, y=0.98)

            for i, (_, combination) in enumerate(native_combinations.iterrows()):
                if i >= len(axes):
                    break

                duration = combination['native_duration']
                stack_depth = combination['stack_depth']

                # Filter data for this combination
                if stack_depth is not None and duration is not None:
                    combo_data = native_success[
                        (native_success['native_duration'] == duration) &
                        (native_success['stack_depth'] == stack_depth)
                    ]
                    title = f'Duration: {duration}s, Stack: {stack_depth}'
                elif duration is not None:
                    combo_data = native_success[native_success['native_duration'] == duration]
                    title = f'Duration: {duration}s'
                else:
                    combo_data = native_success
                    title = 'Native Tests'

                for j, interval in enumerate(interval_order):
                    interval_data = combo_data[combo_data['interval'] == interval]
                    if not interval_data.empty:
                        # Sort by queue size for proper line connection
                        interval_data = interval_data.sort_values('queue_size')
                        axes[i].plot(interval_data['queue_size'], interval_data['loss_percentage'],
                                   marker='o', linewidth=2, markersize=6,
                                   color=colors[j], label=f'{interval}',
                                   alpha=0.8)

                axes[i].set_title(title)
                axes[i].set_xlabel('Queue Size')
                axes[i].set_ylabel('Loss Percentage (%)')
                axes[i].set_xscale('log')
                axes[i].grid(True, alpha=0.3)

                # Only add legend if there are actually lines plotted
                lines, labels = axes[i].get_legend_handles_labels()
                if lines:
                    axes[i].legend()

                # Set consistent y-axis limits across all subplots
                if not combo_data.empty and combo_data['loss_percentage'].max() > 0:
                    axes[i].set_ylim(0, combo_data['loss_percentage'].max() * 1.1)

            # Hide any unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'combined_native_loss_all_intervals.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: combined_native_loss_all_intervals.png")

    # 2. Renaissance tests - separate graph for each stack depth
    if not renaissance_success.empty:
        n_stacks = len(renaissance_stacks)
        if n_stacks > 0:
            if n_stacks == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = [ax]
            else:
                cols = min(3, n_stacks)
                rows = (n_stacks + cols - 1) // cols
                fig_width = 8 * cols
                fig_height = 6 * rows
                fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
                if rows == 1:
                    axes = axes.flatten() if n_stacks > 1 else [axes]
                else:
                    axes = axes.flatten()

            fig.suptitle('Renaissance Tests: Loss Percentage vs Queue Size (All Intervals Combined)', fontsize=16, y=0.98)

            for i, stack_depth in enumerate(renaissance_stacks):
                if i >= len(axes):
                    break

                if stack_depth is not None:
                    stack_data = renaissance_success[renaissance_success['stack_depth'] == stack_depth]
                    title = f'Stack Depth: {stack_depth}'
                else:
                    stack_data = renaissance_success
                    title = 'Renaissance Tests'

                for j, interval in enumerate(interval_order):
                    interval_data = stack_data[stack_data['interval'] == interval]
                    if not interval_data.empty:
                        # Sort by queue size for proper line connection
                        interval_data = interval_data.sort_values('queue_size')
                        axes[i].plot(interval_data['queue_size'], interval_data['loss_percentage'],
                                   marker='o', linewidth=2, markersize=6,
                                   color=colors[j], label=f'{interval}',
                                   alpha=0.8)

                axes[i].set_title(title, fontsize=12)
                axes[i].set_xlabel('Queue Size')
                axes[i].set_ylabel('Loss Percentage (%)')
                axes[i].set_xscale('log')
                axes[i].grid(True, alpha=0.3)

                # Only add legend if there are actually lines plotted
                lines, labels = axes[i].get_legend_handles_labels()
                if lines:
                    axes[i].legend()

                if not stack_data.empty and stack_data['loss_percentage'].max() > 0:
                    axes[i].set_ylim(0, stack_data['loss_percentage'].max() * 1.1)

            # Hide any unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'combined_renaissance_loss_all_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: combined_renaissance_loss_all_intervals.png")

def create_combined_queue_size_graphs(native_df: pd.DataFrame, renaissance_df: pd.DataFrame,
                                    native_queue_data: Dict, renaissance_queue_data: Dict):
    """Create combined queue size percentile graphs"""

    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Define interval order and colors
    interval_order = ["1ms", "5ms", "10ms", "20ms"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Filter successful tests only
    if native_df is not None:
        native_success = native_df[native_df['success'] == True].copy()

        # Get unique combinations of native duration and stack depth
        if 'stack_depth' in native_success.columns and 'native_duration' in native_success.columns:
            native_combinations = native_success[['native_duration', 'stack_depth']].drop_duplicates().sort_values(['native_duration', 'stack_depth'])
        else:
            # Fallback for old data without stack_depth
            native_combinations = pd.DataFrame({'native_duration': sorted(native_success['native_duration'].unique()) if 'native_duration' in native_success.columns else [None], 'stack_depth': [None]})
    else:
        native_success = pd.DataFrame()
        native_combinations = pd.DataFrame()

    if renaissance_df is not None:
        renaissance_success = renaissance_df[renaissance_df['success'] == True].copy()

        # Get unique stack depths for Renaissance
        if 'stack_depth' in renaissance_success.columns:
            renaissance_stacks = sorted(renaissance_success['stack_depth'].unique())
        else:
            renaissance_stacks = [None]
    else:
        renaissance_success = pd.DataFrame()
        renaissance_stacks = []

    # 1. Native tests - P99 and P99.9 queue sizes
    if not native_success.empty and native_queue_data and not native_combinations.empty:
        n_combinations = len(native_combinations)

        for percentile, percentile_label in [('p99', 'P99'), ('p99_9', 'P99.9')]:
            if n_combinations > 0:
                cols = min(4, n_combinations)
                rows = (n_combinations + cols - 1) // cols
                fig_width = 6 * cols
                fig_height = 5 * rows

                fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
                if rows == 1 and cols == 1:
                    axes = [axes]
                elif rows == 1 or cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                fig.suptitle(f'Native Tests: {percentile_label} Queue Size vs Configured Size (All Intervals)', fontsize=16, y=0.98)

                for i, (_, combination) in enumerate(native_combinations.iterrows()):
                    if i >= len(axes):
                        break

                    duration = combination['native_duration']
                    stack_depth = combination['stack_depth']

                    # Filter data for this combination
                    if stack_depth is not None and duration is not None:
                        combo_data = native_success[
                            (native_success['native_duration'] == duration) &
                            (native_success['stack_depth'] == stack_depth)
                        ]
                        title = f'Duration: {duration}s, Stack: {stack_depth}'
                    elif duration is not None:
                        combo_data = native_success[native_success['native_duration'] == duration]
                        title = f'Duration: {duration}s'
                    else:
                        combo_data = native_success
                        title = 'Native Tests'

                    for j, interval in enumerate(interval_order):
                        interval_data = combo_data[combo_data['interval'] == interval]

                        # Extract queue size data for this interval and duration/stack combination
                        queue_sizes = []
                        percentile_values = []

                        for _, row in interval_data.iterrows():
                            if stack_depth is not None:
                                key = (row['queue_size'], row['interval'], row['native_duration'], row['stack_depth'])
                            else:
                                key = (row['queue_size'], row['interval'], row.get('native_duration'))

                            if key in native_queue_data:
                                queue_stats = native_queue_data[key]
                                percentiles = extract_percentiles_from_queue_stats(queue_stats, [percentile])
                                if percentile in percentiles:
                                    queue_sizes.append(row['queue_size'])
                                    percentile_values.append(percentiles[percentile])

                        if queue_sizes:
                            # Sort by queue size
                            sorted_data = sorted(zip(queue_sizes, percentile_values))
                            queue_sizes, percentile_values = zip(*sorted_data)

                            axes[i].plot(queue_sizes, percentile_values,
                                       marker='o', linewidth=2, markersize=6,
                                       color=colors[j], label=f'{interval}',
                                       alpha=0.8)

                    # Add diagonal line showing configured = actual
                    if not combo_data.empty:
                        min_queue = combo_data['queue_size'].min()
                        max_queue = combo_data['queue_size'].max()
                        axes[i].plot([min_queue, max_queue], [min_queue, max_queue],
                                   'k--', alpha=0.5, label='Configured = Actual')

                    axes[i].set_title(title)
                    axes[i].set_xlabel('Configured Queue Size')
                    axes[i].set_ylabel(f'{percentile_label} Actual Queue Size')
                    axes[i].set_xscale('log')
                    axes[i].set_yscale('log')
                    axes[i].grid(True, alpha=0.3)

                    # Only add legend if there are actually lines plotted
                    lines, labels = axes[i].get_legend_handles_labels()
                    if lines:
                        axes[i].legend()

                # Hide any unused subplots
                for j in range(i+1, len(axes)):
                    axes[j].axis('off')

                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f'combined_native_{percentile}_queue_size_all_intervals.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved: combined_native_{percentile}_queue_size_all_intervals.png")

    # 2. Renaissance tests - P99 and P99.9 queue sizes
    if not renaissance_success.empty and renaissance_queue_data:
        for percentile, percentile_label in [('p99', 'P99'), ('p99_9', 'P99.9')]:
            fig, ax = plt.subplots(figsize=(10, 6))

            for j, interval in enumerate(interval_order):
                interval_data = renaissance_success[renaissance_success['interval'] == interval]

                # Extract queue size data for this interval
                queue_sizes = []
                percentile_values = []

                for _, row in interval_data.iterrows():
                    key = (row['queue_size'], row['interval'])
                    if key in renaissance_queue_data:
                        queue_stats = renaissance_queue_data[key]
                        percentiles = extract_percentiles_from_queue_stats(queue_stats, [percentile])
                        if percentile in percentiles:
                            queue_sizes.append(row['queue_size'])
                            percentile_values.append(percentiles[percentile])

                if queue_sizes:
                    # Sort by queue size
                    sorted_data = sorted(zip(queue_sizes, percentile_values))
                    queue_sizes, percentile_values = zip(*sorted_data)

                    ax.plot(queue_sizes, percentile_values,
                           marker='o', linewidth=2, markersize=6,
                           color=colors[j], label=f'{interval}',
                           alpha=0.8)

            # Add diagonal line showing configured = actual
            if not renaissance_success.empty:
                min_queue = renaissance_success['queue_size'].min()
                max_queue = renaissance_success['queue_size'].max()
                ax.plot([min_queue, max_queue], [min_queue, max_queue],
                       'k--', alpha=0.5, label='Configured = Actual')

            ax.set_title(f'Renaissance Tests: {percentile_label} Queue Size vs Configured Size (All Intervals)', fontsize=14)
            ax.set_xlabel('Configured Queue Size')
            ax.set_ylabel(f'{percentile_label} Actual Queue Size')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            # Only add legend if there are actually lines plotted
            lines, labels = ax.get_legend_handles_labels()
            if lines:
                ax.legend()

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f'combined_renaissance_{percentile}_queue_size_all_intervals.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: combined_renaissance_{percentile}_queue_size_all_intervals.png")

def create_summary_table(native_df: pd.DataFrame, renaissance_df: pd.DataFrame):
    """Create summary statistics table"""

    print("\n📊 SUMMARY STATISTICS")
    print("=" * 60)

    # Native summary
    if native_df is not None and not native_df.empty:
        native_success = native_df[native_df['success'] == True]
        print(f"\n🔧 NATIVE TESTS:")
        print(f"   Total tests: {len(native_df)}")
        print(f"   Successful: {len(native_success)} ({len(native_success)/len(native_df)*100:.1f}%)")

        if not native_success.empty:
            for duration in sorted(native_success['native_duration'].unique()):
                duration_data = native_success[native_success['native_duration'] == duration]
                avg_loss = duration_data['loss_percentage'].mean()
                min_loss = duration_data['loss_percentage'].min()
                max_loss = duration_data['loss_percentage'].max()
                print(f"   Duration {duration}s: avg={avg_loss:.2f}%, min={min_loss:.2f}%, max={max_loss:.2f}%")

    # Renaissance summary
    if renaissance_df is not None and not renaissance_df.empty:
        renaissance_success = renaissance_df[renaissance_df['success'] == True]
        print(f"\n🏛️ RENAISSANCE TESTS:")
        print(f"   Total tests: {len(renaissance_df)}")
        print(f"   Successful: {len(renaissance_success)} ({len(renaissance_success)/len(renaissance_df)*100:.1f}%)")

        if not renaissance_success.empty:
            avg_loss = renaissance_success['loss_percentage'].mean()
            min_loss = renaissance_success['loss_percentage'].min()
            max_loss = renaissance_success['loss_percentage'].max()
            print(f"   Loss rate: avg={avg_loss:.2f}%, min={min_loss:.2f}%, max={max_loss:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Combined JFR Queue Size Analysis')
    parser.add_argument('--native-only', action='store_true',
                       help='Analyze only native test results')
    parser.add_argument('--renaissance-only', action='store_true',
                       help='Analyze only Renaissance test results')
    parser.add_argument('--loss-only', action='store_true',
                       help='Generate only loss percentage graphs (skip queue size percentiles)')
    parser.add_argument('--csv', type=str,
                       help='Analyze specific CSV file instead of latest')
    parser.add_argument('--json', type=str,
                       help='Use specific JSON file for queue stats')

    args = parser.parse_args()

    # Determine what to analyze
    if args.native_only:
        test_type = 'native'
    elif args.renaissance_only:
        test_type = 'renaissance'
    else:
        test_type = 'both'

    print(f"🔍 Combined JFR Queue Size Analysis")
    print(f"{'='*50}")

    # Create plots directory
    PLOTS_DIR.mkdir(exist_ok=True)

    # Load CSV data for loss percentages
    if args.csv:
        # Load specific CSV file
        csv_path = Path(args.csv)
        if not csv_path.exists():
            # Try in data directory
            csv_path = DATA_DIR / args.csv

        if csv_path.exists():
            print(f"📊 Loading specific CSV: {csv_path.name}")
            df = pd.read_csv(csv_path)

            # Determine if it's native or renaissance based on columns
            if 'native_duration' in df.columns:
                native_df = df
                renaissance_df = None
                print(f"   {len(native_df)} native test records")
            else:
                native_df = None
                renaissance_df = df
                print(f"   {len(renaissance_df)} renaissance test records")
        else:
            print(f"❌ CSV file not found: {args.csv}")
            return
    else:
        native_df, renaissance_df = load_csv_data(test_type)

    if native_df is None and renaissance_df is None:
        print("❌ No data found to analyze!")
        return

    # Create combined loss percentage graphs
    print(f"\n📊 Generating combined loss percentage graphs...")
    create_combined_loss_graphs(native_df, renaissance_df)

    # Load JSON data for queue size percentiles (unless loss-only)
    if not args.loss_only:
        print(f"\n📊 Loading queue size percentile data...")
        native_queue_data, renaissance_queue_data = load_json_data_for_queue_stats(test_type)

        if native_queue_data or renaissance_queue_data:
            print(f"\n📊 Generating combined queue size percentile graphs...")
            create_combined_queue_size_graphs(native_df, renaissance_df,
                                            native_queue_data, renaissance_queue_data)
        else:
            print(f"⚠️ No queue size percentile data found in JSON files")

    # Create summary
    create_summary_table(native_df, renaissance_df)

    print(f"\n✅ Analysis complete! Graphs saved to: {PLOTS_DIR}")
    print(f"🎯 Combined graphs show all sampling intervals on single plots for easy comparison")

if __name__ == "__main__":
    main()
