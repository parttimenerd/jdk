#!/usr/bin/env python3
"""
JFR Queue Size Benchmarking Script

Comprehensive testing of JFR queue sizes with different sampling intervals
and native durations. Supports data persistence and visualization regeneration.

Usage:
    python3 benchmark_queue_sizes.py --estimate
    python3 benchmark_queue_sizes.py --run-renaissance    # Run Renaissance tests
    python3 benchmark_queue_sizes.py --visualize          # Generate visualizations only
    python3 benchmark_queue_sizes.py --all                # Run all tests and visualize
    python3 benchmark_queue_sizes.py --threads 8          # Use 8 threads instead of CPU count

Runtime Estimates:
    Native benchmark: ~26 hours (108 tests × 4.5 minutes avg per test)
    Renaissance benchmark: ~2.4 hours (36 tests × 4 minutes avg per test)
    Combined: ~28.4 hours total
"""

import argparse
import csv
import glob
import json
import os
import psutil
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Import analysis functions from analyze_drain_categories
try:
    from analyze_drain_categories import extract_drain_stats_by_label, DRAIN_CATEGORIES, calculate_event_weighted_percentiles
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    ANALYSIS_AVAILABLE = False

# Import direct log parsing functions
try:
    from parse_logs_directly import parse_all_logs, create_percentile_dataframe, parse_drain_stats_from_log
    DIRECT_PARSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Direct log parsing not available: {e}")
    DIRECT_PARSING_AVAILABLE = False

# Process Management Utilities

def kill_lingering_java_processes(force=False, verbose=False):
    """Kill lingering Java processes that might interfere with benchmarks"""
    if verbose:
        print("    🧹 Cleaning up lingering Java processes...")

    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'java' in proc.info['name'].lower():
                # Filter out IDE/system Java processes
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(keyword in cmdline.lower() for keyword in ['idea', 'eclipse', 'vscode', 'netbeans']):
                    continue  # Skip IDE processes

                # Look for benchmark-related processes
                if any(keyword in cmdline.lower() for keyword in ['run.sh', 'renaissance', 'jfr', 'benchmark']):
                    if verbose:
                        print(f"      🔪 Killing Java process PID {proc.pid}")
                    if force:
                        proc.kill()  # SIGKILL
                    else:
                        proc.terminate()  # SIGTERM
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if killed_count > 0:
        if verbose:
            print(f"    🧹 Killed {killed_count} lingering Java processes")
        time.sleep(3)  # Give time for cleanup
    else:
        if verbose:
            print("    ✅ No lingering Java processes to clean up")

def has_json_parsing_errors(log_path: Path, verbose: bool = False) -> bool:
    """Check if the log file contains JSON parsing errors that indicate incomplete/corrupted output"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Look for JSON parsing error indicators
        json_error_indicators = [
            "Error parsing JSON:",
            "JSONDecodeError",
            "No DRAIN_STATS_JSON found",
            "Could not parse all required statistics",
            "Missing Successful Samples",
            "Missing Total Samples",
            "Missing Lost Samples"
        ]

        # Check if any error indicators are present
        for indicator in json_error_indicators:
            if indicator in content:
                if verbose:
                    print(f"    ⚠️ Found JSON parsing issue: {indicator}")
                return True

        # Also check if we have incomplete JSON (truncated output)
        if 'DRAIN_STATS_JSON:' in content:
            # Count complete vs incomplete JSON entries
            json_lines = [line for line in content.split('\n') if 'DRAIN_STATS_JSON:' in line]
            for line in json_lines:
                match = re.search(r'DRAIN_STATS_JSON:\s*(\{.*\})', line)
                if match:
                    try:
                        json.loads(match.group(1))
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"    ⚠️ Found malformed JSON in DRAIN_STATS_JSON")
                        return True

        return False

    except Exception as e:
        if verbose:
            print(f"    ⚠️ Error checking for JSON parsing errors: {e}")
        return True  # Assume there are errors if we can't check

# Configuration
QUEUE_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 750, 1000]
# restart threads after every native call
RESTART_THREADS_EVERY = 1
SAMPLING_INTERVALS = ["1ms", "10ms", "20ms"] #  ["1ms", "2ms", "5ms", "10ms", "20ms"]
NATIVE_DURATIONS = [30] # 250] # [5, 250]  # seconds
STACK_DEPTHS = [1200] #  [100, 1200]  # Different stack depths to test
TEST_DURATION = 250  # seconds
THREADS = os.cpu_count() or 4  # Use number of CPUs, fallback to 4

# Minimal configuration for quick testing
MINIMAL_QUEUE_SIZES = [100, 500]
MINIMAL_SAMPLING_INTERVALS = ["1ms", "10ms"]
MINIMAL_NATIVE_DURATIONS = [10]  #[10, 25]  # seconds
MINIMAL_STACK_DEPTHS = [1200] #[100, 1200]  # Test both stack depths
MINIMAL_TEST_DURATION = 30  # seconds

# Renaissance specific
RENAISSANCE_ITERATIONS = 3
MINIMAL_RENAISSANCE_ITERATIONS = 1  # Faster minimal testing

# Output directories
RESULTS_DIR = Path("benchmark_results")
LOGS_DIR = RESULTS_DIR / "logs"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"

class BenchmarkRunner:
    def __init__(self, minimal=False, threads=None, max_retries=2, verbose=False, dynamic_queue_size=False):
        self.setup_directories()
        self.results = {
            'native': [],
            'renaissance': []
        }
        self.minimal = minimal
        self.max_retries = max_retries
        self.verbose = verbose
        self.dynamic_queue_size = dynamic_queue_size

        # Track session start time to filter data from current run only
        self.session_start_time = time.time()

        # Always plot after every iteration for real-time feedback
        self.plot_frequency = 1  # Plot after every single test

        # Set thread count - use provided value, fallback to CPU count or 4
        self.threads = threads if threads is not None else (os.cpu_count() or 4)

        # Set configuration based on minimal flag
        if minimal:
            self.queue_sizes = MINIMAL_QUEUE_SIZES
            self.sampling_intervals = MINIMAL_SAMPLING_INTERVALS
            self.native_durations = MINIMAL_NATIVE_DURATIONS
            self.stack_depths = MINIMAL_STACK_DEPTHS
            self.test_duration = MINIMAL_TEST_DURATION
            self.renaissance_iterations = MINIMAL_RENAISSANCE_ITERATIONS
        else:
            self.queue_sizes = QUEUE_SIZES
            self.sampling_intervals = SAMPLING_INTERVALS
            self.native_durations = NATIVE_DURATIONS
            self.stack_depths = STACK_DEPTHS
            self.test_duration = TEST_DURATION
            self.renaissance_iterations = RENAISSANCE_ITERATIONS

    def _get_plot_filename(self, base_name: str, progress_mode: bool = False) -> str:
        """Generate appropriate filename based on progress mode"""
        if progress_mode:
            # Insert '_progress' before the file extension
            name_parts = base_name.split('.')
            if len(name_parts) > 1:
                return f"{'.'.join(name_parts[:-1])}_progress.{name_parts[-1]}"
            else:
                return f"{base_name}_progress.png"
        else:
            # Ensure .png extension if not present
            if not base_name.endswith('.png'):
                return f"{base_name}.png"
        return base_name

    def estimate_runtime(self, test_type: str) -> Tuple[int, str]:
        """Estimate total runtime for benchmark suite using real data from previous runs"""
        if test_type == 'native':
            # Try to get real timing data from previous CSV files
            actual_time_per_test = self._get_actual_test_duration('native')
            if actual_time_per_test is None:
                # Fallback to estimated timing if no real data available
                actual_time_per_test = self.test_duration + 30  # estimated setup/teardown
                self.vprint(f"⚠️ No historical data found, using estimated {actual_time_per_test:.1f}s per test")

            total_tests = len(self.queue_sizes) * len(self.sampling_intervals) * len(self.native_durations) * len(self.stack_depths)
            total_seconds = total_tests * actual_time_per_test

        elif test_type == 'renaissance':
            # Try to get real timing data from previous CSV files
            actual_time_per_test = self._get_actual_test_duration('renaissance')
            if actual_time_per_test is None:
                # Fallback: Renaissance tests take 3:30 minutes (210s) + setup/teardown (estimated 30s per test)
                actual_time_per_test = 210 + 30  # 4 minutes total per test
                self.vprint(f"⚠️ No historical data found, using estimated {actual_time_per_test:.1f}s per test")

            total_tests = len(self.queue_sizes) * len(self.sampling_intervals) * len(self.stack_depths)
            total_seconds = total_tests * actual_time_per_test * self.renaissance_iterations
        else:
            return 0, "Unknown test type"

        # Convert to human readable format
        hours = int(total_seconds) // 3600
        minutes = (int(total_seconds) % 3600) // 60
        remaining_seconds = int(total_seconds) % 60

        if hours > 0:
            time_str = f"{hours}h {minutes}m {remaining_seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {remaining_seconds}s"
        else:
            time_str = f"{remaining_seconds}s"

        return total_seconds, time_str

    def _get_actual_test_duration(self, test_type: str) -> Optional[float]:
        """Get average actual test duration from historical CSV data"""
        import glob
        import pandas as pd

        # Look for CSV files with actual timing data
        pattern = f"{DATA_DIR}/{test_type}_*.csv"
        csv_files = glob.glob(pattern)

        if not csv_files:
            return None

        all_durations = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'test_duration_actual' in df.columns:
                    # Filter out outliers (tests that finished way too early, likely due to errors)
                    durations = df['test_duration_actual'].dropna()
                    if test_type == 'native':
                        # For native tests, filter out tests that finished much faster than expected
                        # (likely due to errors or early termination)
                        min_expected = self.test_duration * 0.25  # At least 25% of test duration
                        durations = durations[durations >= min_expected]
                    elif test_type == 'renaissance':
                        # Renaissance tests should take at least 2 minutes (120s)
                        durations = durations[durations >= 120]

                    all_durations.extend(durations.tolist())
            except Exception as e:
                self.vprint(f"Warning: Could not read {csv_file} for timing data: {e}")
                continue

        if not all_durations:
            return None

        # Calculate median duration to avoid outliers affecting the estimate
        import statistics
        median_duration = statistics.median(all_durations)

        # Adjust estimate based on current test duration vs historical data
        if test_type == 'native':
            # Check if historical data is for a similar test duration
            historical_test_duration = median_duration - 30  # Rough estimate of historical test duration (minus overhead)
            if abs(historical_test_duration - self.test_duration) > 50:  # Significant difference (>50s)
                # Scale the overhead estimation based on test duration ratio
                overhead_ratio = median_duration / max(historical_test_duration, 1)
                estimated_duration = self.test_duration * overhead_ratio
                self.vprint(f"📊 Scaling estimate: Historical data from {historical_test_duration:.0f}s tests, current {self.test_duration}s tests")
                self.vprint(f"📊 Scaled estimate: {len(all_durations)} samples, historical median={median_duration:.1f}s, scaled estimate={estimated_duration:.1f}s")
            else:
                # Add some buffer for variance (10% extra)
                estimated_duration = median_duration * 1.10
                self.vprint(f"📊 Using real data: {len(all_durations)} samples, median={median_duration:.1f}s, estimate={estimated_duration:.1f}s")
        else:
            # For Renaissance tests, use median with buffer
            estimated_duration = median_duration * 1.10
            self.vprint(f"📊 Using real data: {len(all_durations)} samples, median={median_duration:.1f}s, estimate={estimated_duration:.1f}s")
        return estimated_duration

    def setup_directories(self):
        """Create necessary directories for results"""
        for dir_path in [RESULTS_DIR, LOGS_DIR, DATA_DIR, PLOTS_DIR, TABLES_DIR]:
            dir_path.mkdir(exist_ok=True)

    def vprint(self, *args, **kwargs):
        """Print only if verbose mode is enabled"""
        if self.verbose:
            print(*args, **kwargs)

    def save_ascii_table(self, df: pd.DataFrame, base_filename: str, title: str, progress_mode: bool = False):
        """Save an ASCII table representation of the plot data"""
        try:
            # Create tables subdirectory if needed
            tables_subdir = TABLES_DIR
            if progress_mode:
                tables_subdir = TABLES_DIR / "progress"
            tables_subdir.mkdir(exist_ok=True)

            # Generate filename
            table_filename = base_filename.replace('.png', '.txt')
            table_path = tables_subdir / table_filename

            # Create ASCII table content
            with open(table_path, 'w') as f:
                f.write(f"{title}\n")
                f.write("=" * len(title) + "\n\n")

                # Sort data by queue_size for consistent ordering
                df_sorted = df.sort_values(['queue_size'])

                # Format the table
                if 'interval' in df.columns:
                    # Group by interval for multi-interval data
                    intervals = sorted(df_sorted['interval'].unique())

                    for interval in intervals:
                        interval_data = df_sorted[df_sorted['interval'] == interval]
                        f.write(f"Sampling Interval: {interval}ms\n")
                        f.write("-" * 30 + "\n")

                        # Create table headers
                        headers = ["Queue Size", "Loss %"]
                        if 'out_of_thread_percentage' in interval_data.columns:
                            headers.append("Out-of-Thread %")

                        # Calculate column widths
                        col_widths = [max(len(str(h)), 12) for h in headers]

                        # Write headers
                        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
                        f.write(header_line + "\n")
                        f.write("-" * len(header_line) + "\n")

                        # Write data rows
                        for _, row in interval_data.iterrows():
                            values = [f"{int(row['queue_size']):,}", f"{row['loss_percentage']:.3f}"]
                            if 'out_of_thread_percentage' in row and pd.notna(row['out_of_thread_percentage']):
                                values.append(f"{row['out_of_thread_percentage']:.3f}")
                            elif len(headers) > 2:
                                values.append("N/A")

                            data_line = " | ".join(v.ljust(w) for v, w in zip(values, col_widths))
                            f.write(data_line + "\n")

                        f.write("\n")
                else:
                    # Single table for data without intervals
                    headers = ["Queue Size", "Loss %"]
                    if 'out_of_thread_percentage' in df_sorted.columns:
                        headers.append("Out-of-Thread %")

                    # Calculate column widths
                    col_widths = [max(len(str(h)), 12) for h in headers]

                    # Write headers
                    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
                    f.write(header_line + "\n")
                    f.write("-" * len(header_line) + "\n")

                    # Write data rows
                    for _, row in df_sorted.iterrows():
                        values = [f"{int(row['queue_size']):,}", f"{row['loss_percentage']:.3f}"]
                        if 'out_of_thread_percentage' in row and pd.notna(row['out_of_thread_percentage']):
                            values.append(f"{row['out_of_thread_percentage']:.3f}")
                        elif len(headers) > 2:
                            values.append("N/A")

                        data_line = " | ".join(v.ljust(w) for v, w in zip(values, col_widths))
                        f.write(data_line + "\n")

                # Add summary statistics
                f.write("\nSummary Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Min Loss Rate: {df['loss_percentage'].min():.3f}%\n")
                f.write(f"Max Loss Rate: {df['loss_percentage'].max():.3f}%\n")
                f.write(f"Mean Loss Rate: {df['loss_percentage'].mean():.3f}%\n")
                f.write(f"Median Loss Rate: {df['loss_percentage'].median():.3f}%\n")
                f.write(f"Total Configurations: {len(df)}\n")

                if 'out_of_thread_percentage' in df.columns:
                    out_of_thread_data = df.dropna(subset=['out_of_thread_percentage'])
                    if len(out_of_thread_data) > 0:
                        f.write(f"Mean Out-of-Thread: {out_of_thread_data['out_of_thread_percentage'].mean():.3f}%\n")

            self.vprint(f"    📋 Saved ASCII table: {table_path}")

        except Exception as e:
            self.vprint(f"    ⚠️ Error saving ASCII table {base_filename}: {e}")

    def save_loss_kinds_ascii_table(self, df: pd.DataFrame, base_filename: str, title: str, progress_mode: bool = False):
        """Save an ASCII table for loss kinds breakdown"""
        try:
            # Create tables subdirectory if needed
            tables_subdir = TABLES_DIR / "loss_kinds"
            if progress_mode:
                tables_subdir = TABLES_DIR / "progress" / "loss_kinds"
            tables_subdir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            table_filename = base_filename.replace('.png', '.txt')
            table_path = tables_subdir / table_filename

            # Loss categories as defined in the plotting function
            loss_categories = ['stw_gc', 'invalid_state', 'could_not_acquire_lock', 'enqueue_failed']

            # Process data to get loss kinds breakdown
            loss_data = []
            for _, row in df.iterrows():
                queue_size = row['queue_size']
                interval = row['interval']

                # Get loss kinds data (handle both raw dict and flattened CSV format)
                loss_kinds = {}
                total_lost_samples = 0

                # Try to get from raw dict format first
                if 'loss_kinds' in row and isinstance(row['loss_kinds'], dict):
                    loss_kinds = row['loss_kinds']
                    total_lost_samples = row.get('total_lost_samples', 0)
                else:
                    # Try to get from flattened CSV format
                    for category in loss_categories:
                        csv_col = f'loss_kind_{category}'
                        if csv_col in row:
                            loss_kinds[category] = row[csv_col] if pd.notna(row[csv_col]) else 0
                            total_lost_samples += loss_kinds[category]

                if loss_kinds and total_lost_samples > 0:
                    overall_loss_pct = row['loss_percentage']
                    for category in loss_categories:
                        category_count = loss_kinds.get(category, 0)
                        category_loss_pct = (category_count / total_lost_samples) * overall_loss_pct if total_lost_samples > 0 else 0.0

                        loss_data.append({
                            'queue_size': queue_size,
                            'interval': interval,
                            'loss_category': category,
                            'loss_percentage': category_loss_pct,
                            'count': category_count,
                            'total_lost': total_lost_samples
                        })

            if not loss_data:
                self.vprint(f"    ⚠️ No loss kinds data available for table {base_filename}")
                return

            loss_df = pd.DataFrame(loss_data)

            # Create ASCII table content
            with open(table_path, 'w') as f:
                f.write(f"{title}\n")
                f.write("=" * len(title) + "\n\n")

                # Group by interval
                intervals = sorted(loss_df['interval'].unique())

                for interval in intervals:
                    interval_data = loss_df[loss_df['interval'] == interval]
                    f.write(f"Sampling Interval: {interval}ms\n")
                    f.write("-" * 40 + "\n")

                    # Get unique queue sizes for this interval
                    queue_sizes = sorted(interval_data['queue_size'].unique())

                    for queue_size in queue_sizes:
                        queue_data = interval_data[interval_data['queue_size'] == queue_size]
                        f.write(f"\nQueue Size: {int(queue_size):,}\n")

                        # Create table headers
                        headers = ["Loss Category", "Count", "Loss %", "% of Total Loss"]
                        col_widths = [20, 10, 10, 16]

                        # Write headers
                        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
                        f.write(header_line + "\n")
                        f.write("-" * len(header_line) + "\n")

                        # Write data rows
                        total_lost = queue_data.iloc[0]['total_lost'] if len(queue_data) > 0 else 0
                        for _, row in queue_data.iterrows():
                            category = row['loss_category'].replace('_', ' ').title()
                            count = int(row['count'])
                            loss_pct = row['loss_percentage']
                            pct_of_total = (row['count'] / total_lost * 100) if total_lost > 0 else 0

                            values = [category, f"{count:,}", f"{loss_pct:.3f}%", f"{pct_of_total:.1f}%"]
                            data_line = " | ".join(v.ljust(w) for v, w in zip(values, col_widths))
                            f.write(data_line + "\n")

                        f.write(f"\nTotal Lost Samples: {total_lost:,}\n")

                    f.write("\n" + "=" * 60 + "\n\n")

                # Add overall summary
                f.write("Overall Summary:\n")
                f.write("-" * 16 + "\n")

                category_totals = loss_df.groupby('loss_category').agg({
                    'count': 'sum',
                    'loss_percentage': 'mean'
                }).round(3)

                for category, data in category_totals.iterrows():
                    category_name = category.replace('_', ' ').title()
                    f.write(f"{category_name}: {data['count']:,} samples (avg {data['loss_percentage']:.3f}% loss)\n")

            self.vprint(f"    📋 Saved loss kinds ASCII table: {table_path}")

        except Exception as e:
            self.vprint(f"    ⚠️ Error saving loss kinds ASCII table {base_filename}: {e}")

    def flatten_for_csv(self, results: List[Dict]) -> pd.DataFrame:
        """Flatten complex nested fields for CSV export"""
        flattened_results = []

        for result in results:
            flattened = result.copy()

            # Flatten out_of_thread_details (only keep ns timing entries)
            if result.get('out_of_thread_details'):
                details = result['out_of_thread_details']
                flattened['out_of_thread_requests'] = details.get('requests')
                flattened['out_of_thread_request_rate'] = details.get('request_rate')

                # Flatten time_ns statistics (only avg and max)
                if details.get('time_ns'):
                    time_stats = details['time_ns']
                    flattened['out_of_thread_time_avg_ns'] = time_stats.get('avg')
                    flattened['out_of_thread_time_max_ns'] = time_stats.get('max')

                # Flatten event statistics
                if details.get('events'):
                    events_stats = details['events']
                    flattened['out_of_thread_events_sum'] = events_stats.get('sum')
                    flattened['out_of_thread_events_avg'] = events_stats.get('avg')
                    flattened['out_of_thread_events_min'] = events_stats.get('min')
                    flattened['out_of_thread_events_max'] = events_stats.get('max')
                    flattened['out_of_thread_events_median'] = events_stats.get('median')
                    flattened['out_of_thread_events_p95'] = events_stats.get('p95')
                    flattened['out_of_thread_events_p99'] = events_stats.get('p99')
                    flattened['out_of_thread_events_p99_9'] = events_stats.get('p99_9')

                # Flatten queue size percentiles
                if details.get('queue_size_percentiles'):
                    queue_percs = details['queue_size_percentiles']
                    flattened['out_of_thread_queue_median'] = queue_percs.get('median')
                    flattened['out_of_thread_queue_p95'] = queue_percs.get('p95')
                    flattened['out_of_thread_queue_p99'] = queue_percs.get('p99')
                    flattened['out_of_thread_queue_p99_9'] = queue_percs.get('p99_9')
                    flattened['out_of_thread_queue_p99_99'] = queue_percs.get('p99_99')
                    flattened['out_of_thread_queue_p99_999'] = queue_percs.get('p99_999')

            # Flatten all_without_locks_details (only keep ns timing entries)
            if result.get('all_without_locks_details'):
                details = result['all_without_locks_details']
                flattened['all_without_locks_requests'] = details.get('requests')
                flattened['all_without_locks_request_rate'] = details.get('request_rate')

                # Flatten time_ns statistics (only avg and max)
                if details.get('time_ns'):
                    time_stats = details['time_ns']
                    flattened['all_without_locks_time_avg_ns'] = time_stats.get('avg')
                    flattened['all_without_locks_time_max_ns'] = time_stats.get('max')

                # Flatten events statistics
                if details.get('events'):
                    events_stats = details['events']
                    flattened['all_without_locks_events_sum'] = events_stats.get('sum')
                    flattened['all_without_locks_events_avg'] = events_stats.get('avg')
                    flattened['all_without_locks_events_min'] = events_stats.get('min')
                    flattened['all_without_locks_events_max'] = events_stats.get('max')
                    flattened['all_without_locks_events_median'] = events_stats.get('median')
                    flattened['all_without_locks_events_p95'] = events_stats.get('p95')
                    flattened['all_without_locks_events_p99'] = events_stats.get('p99')
                    flattened['all_without_locks_events_p99_9'] = events_stats.get('p99_9')

                # Flatten queue size percentiles
                if details.get('queue_size_percentiles'):
                    queue_percs = details['queue_size_percentiles']
                    flattened['all_without_locks_queue_median'] = queue_percs.get('median')
                    flattened['all_without_locks_queue_p95'] = queue_percs.get('p95')
                    flattened['all_without_locks_queue_p99'] = queue_percs.get('p99')
                    flattened['all_without_locks_queue_p99_9'] = queue_percs.get('p99_9')
                    flattened['all_without_locks_queue_p99_99'] = queue_percs.get('p99_99')
                    flattened['all_without_locks_queue_p99_999'] = queue_percs.get('p99_999')

            # Flatten drain_categories data for comprehensive drain statistics
            if result.get('drain_categories'):
                drain_cats = result['drain_categories']
                total_drain_events = 0
                total_drain_requests = 0

                for category_name, category_data in drain_cats.items():
                    # Clean category name for CSV columns
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', category_name.lower()).strip('_')

                    # Basic category statistics
                    flattened[f'drain_{clean_name}_requests'] = category_data.get('requests', 0)
                    flattened[f'drain_{clean_name}_runtime_seconds'] = category_data.get('runtime_seconds', 0)
                    flattened[f'drain_{clean_name}_request_rate'] = category_data.get('request_rate', 0)

                    # Events with full statistics including p99 and p99.9
                    if category_data.get('events'):
                        events_stats = category_data['events']
                        events_sum = events_stats.get('sum', 0)
                        flattened[f'drain_{clean_name}_events_sum'] = events_sum
                        flattened[f'drain_{clean_name}_events_avg'] = events_stats.get('avg', 0)
                        flattened[f'drain_{clean_name}_events_min'] = events_stats.get('min', 0)
                        flattened[f'drain_{clean_name}_events_max'] = events_stats.get('max', 0)
                        flattened[f'drain_{clean_name}_events_median'] = events_stats.get('median', 0)
                        flattened[f'drain_{clean_name}_events_p95'] = events_stats.get('p95', 0)
                        flattened[f'drain_{clean_name}_events_p99'] = events_stats.get('p99', 0)
                        flattened[f'drain_{clean_name}_events_p99_9'] = events_stats.get('p99_9', 0)

                        # Only count events from "without locks" categories for the total
                        if 'without_locks' in clean_name:
                            total_drain_events += events_sum

                    # Time statistics (ns) - only avg and max
                    if category_data.get('time_ns'):
                        time_stats = category_data['time_ns']
                        flattened[f'drain_{clean_name}_time_avg_ns'] = time_stats.get('avg', 0)
                        flattened[f'drain_{clean_name}_time_max_ns'] = time_stats.get('max', 0)

                    # Queue size percentiles (event-weighted)
                    if category_data.get('queue_size_percentiles'):
                        queue_percs = category_data['queue_size_percentiles']
                        flattened[f'drain_{clean_name}_queue_median'] = queue_percs.get('median', 0)
                        flattened[f'drain_{clean_name}_queue_p95'] = queue_percs.get('p95', 0)
                        flattened[f'drain_{clean_name}_queue_p99'] = queue_percs.get('p99', 0)
                        flattened[f'drain_{clean_name}_queue_p99_9'] = queue_percs.get('p99_9', 0)
                        flattened[f'drain_{clean_name}_queue_p99_99'] = queue_percs.get('p99_99', 0)
                        flattened[f'drain_{clean_name}_queue_p99_999'] = queue_percs.get('p99_999', 0)

                    total_drain_requests += category_data.get('requests', 0)

                # Add summary statistics
                flattened['total_drain_events'] = total_drain_events
                flattened['total_drain_requests'] = total_drain_requests
                flattened['drain_categories_count'] = len(drain_cats)

            # Flatten loss_kinds data for CSV
            if result.get('loss_kinds'):
                loss_kinds = result['loss_kinds']
                for loss_type, count in loss_kinds.items():
                    flattened[f'loss_kind_{loss_type}'] = count

            # Flatten loss_kind_percentages data for CSV
            if result.get('loss_kind_percentages'):
                loss_percentages = result['loss_kind_percentages']
                for loss_type, percentage in loss_percentages.items():
                    flattened[f'loss_kind_pct_{loss_type}'] = percentage

            # Flatten vm_operations data for CSV
            if result.get('vm_operations'):
                vm_ops = result['vm_operations']
                for vm_op_type, count in vm_ops.items():
                    flattened[f'vm_op_{vm_op_type}'] = count

            # Flatten vm_op_percentages data for CSV
            if result.get('vm_op_percentages'):
                vm_op_percentages = result['vm_op_percentages']
                for vm_op_type, percentage in vm_op_percentages.items():
                    flattened[f'vm_op_pct_{vm_op_type}'] = percentage

            # Flatten signal_handler_stats data for CSV
            if result.get('signal_handler_stats'):
                signal_handler_stats = result['signal_handler_stats']
                for signal_name, signal_data in signal_handler_stats.items():
                    # Clean signal name for CSV columns
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', signal_name.lower()).strip('_')

                    # Basic signal handler statistics
                    flattened[f'signal_{clean_name}_signals_processed'] = signal_data.get('signals_processed', 0)
                    flattened[f'signal_{clean_name}_runtime_seconds'] = signal_data.get('runtime_seconds', 0.0)

                    # Handler time statistics with all percentiles
                    if signal_data.get('handler_time'):
                        handler_time = signal_data['handler_time']
                        flattened[f'signal_{clean_name}_time_sum_ns'] = handler_time.get('sum', 0)
                        flattened[f'signal_{clean_name}_time_avg_ns'] = handler_time.get('avg', 0)
                        flattened[f'signal_{clean_name}_time_min_ns'] = handler_time.get('min', 0)
                        flattened[f'signal_{clean_name}_time_max_ns'] = handler_time.get('max', 0)
                        flattened[f'signal_{clean_name}_time_median_ns'] = handler_time.get('median', 0)
                        flattened[f'signal_{clean_name}_time_p90_ns'] = handler_time.get('p90', 0)
                        flattened[f'signal_{clean_name}_time_p95_ns'] = handler_time.get('p95', 0)
                        flattened[f'signal_{clean_name}_time_p99_ns'] = handler_time.get('p99', 0)
                        flattened[f'signal_{clean_name}_time_p99_9_ns'] = handler_time.get('p99_9', 0)

            # Remove the complex nested fields from CSV output
            flattened.pop('out_of_thread_details', None)
            flattened.pop('all_without_locks_details', None)
            flattened.pop('drain_categories', None)
            flattened.pop('loss_kinds', None)
            flattened.pop('loss_kind_percentages', None)
            flattened.pop('vm_operations', None)
            flattened.pop('vm_op_percentages', None)
            flattened.pop('signal_handler_stats', None)

            flattened_results.append(flattened)

        return pd.DataFrame(flattened_results)

    def update_csv_realtime(self, test_type: str, new_result: Dict):
        """Update CSV file in real-time as tests complete"""
        latest_csv = DATA_DIR / f"{test_type}_results_latest.csv"

        # Add the new result to our in-memory results
        self.results[test_type].append(new_result)

        # Convert all results to DataFrame with flattened complex fields and save immediately
        if self.results[test_type]:
            df = self.flatten_for_csv(self.results[test_type])
            df.to_csv(latest_csv, index=False)
            self.vprint(f"    💾 Updated {latest_csv.name} with {len(self.results[test_type])} results")

    def run_test_with_retry(self, test_func, *args, **kwargs) -> Dict:
        """Run a test function with retry logic for JSON parsing failures"""
        max_retries = self.max_retries
        for attempt in range(max_retries + 1):  # 0-based, so +1 for inclusive range
            if attempt > 0:
                print(f"    🔄 Retry attempt {attempt}/{max_retries}")
                # Clean up any lingering processes before retry
                kill_lingering_java_processes(force=False, verbose=self.verbose)
                time.sleep(5)  # Give some time between retries

            result = test_func(*args, **kwargs)

            # Check if test was successful
            if result.get('success', False):
                # Additional check: verify no JSON parsing errors in the log
                # But only retry if we didn't get the essential data (loss_percentage)
                log_file = result.get('log_file')
                if log_file and result.get('loss_percentage') is None:
                    log_path = LOGS_DIR / log_file
                    if log_path.exists() and has_json_parsing_errors(log_path, self.verbose):
                        print(f"    ⚠️ Test succeeded but found JSON parsing errors and missing loss data")
                        if attempt < max_retries:
                            print(f"    🔄 Will retry due to JSON parsing issues and missing data")
                            continue
                        else:
                            print(f"    ⚠️ Max retries reached, keeping result despite JSON issues")
                elif log_file:
                    log_path = LOGS_DIR / log_file
                    if log_path.exists() and has_json_parsing_errors(log_path, self.verbose):
                        print(f"    ⚠️ Test succeeded and extracted loss data, but found JSON parsing errors (not retrying)")

                print(f"    ✅ Test successful" + (f" (after {attempt} retries)" if attempt > 0 else ""))
                return result
            else:
                print(f"    ❌ Test failed" + (f" (attempt {attempt + 1}/{max_retries + 1})" if attempt < max_retries else ""))

        print(f"    ❌ Test failed after {max_retries + 1} attempts")
        return result  # Return the last failed result

    def run_native_test(self, queue_size: int, interval: str, stack_depth: int, native_duration: Optional[int] = None) -> Dict:
        """Run a single native test with retry logic for JSON parsing failures"""
        return self.run_test_with_retry(self._run_native_test_internal, queue_size, interval, stack_depth, native_duration)

    def _run_native_test_internal(self, queue_size: int, interval: str, stack_depth: int, native_duration: Optional[int] = None) -> Dict:
        """Run a single native test and extract loss percentage"""
        print(f"  Running: queue={queue_size}, interval={interval}, stack_depth={stack_depth}, native_duration={native_duration}")

        native_suffix = f"_native{native_duration}s" if native_duration else ""
        log_filename = f"native_q{queue_size}_{interval}_s{stack_depth}_{self.test_duration}s{native_suffix}.log"
        log_path = LOGS_DIR / log_filename

        # Build command
        cmd = [
            "./run.sh",
            "-d", str(self.test_duration),
            "-s", str(stack_depth),
            "-t", str(self.threads),
            "-i", interval,
            "-q", str(queue_size),
            "--no-analysis",  # Enable analysis tables but skip plots/visualizations
            "-f", str(log_path.absolute()) + ".raw"
        ]

        if self.dynamic_queue_size:
            cmd.append("--dynamic-queue-size")

        if native_duration:
            cmd.extend(["--native-duration", str(native_duration)])

        if RESTART_THREADS_EVERY > 0:
            cmd.extend(["--restart-frequency", str(RESTART_THREADS_EVERY)])

        # Generate log filename

        self.vprint(f"    Command: {' '.join(cmd)}")
        self.vprint(f"    Log file: {log_path}")

        # Run test
        try:
            # Set up environment variables
            env = os.environ.copy()
            if self.dynamic_queue_size:
                env['DYNAMIC_QUEUE_SIZE'] = 'true'
                self.vprint(f"    Environment: DYNAMIC_QUEUE_SIZE=true")

            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd="../run_in_native",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            self.vprint(f"    Return code: {result.returncode}")

            # Extract loss percentage and out-of-thread data from log
            extracted_data = self.extract_loss_percentage(log_path)

            # Handle the new dict format
            if isinstance(extracted_data, dict):
                loss_percentage = extracted_data.get('loss_percentage')
                out_of_thread_events = extracted_data.get('out_of_thread_events')
                out_of_thread_percentage = extracted_data.get('out_of_thread_percentage')
                out_of_thread_details = extracted_data.get('out_of_thread_details')
                all_without_locks_events = extracted_data.get('all_without_locks_events')
                all_without_locks_details = extracted_data.get('all_without_locks_details')
                final_max_queue_size = extracted_data.get('final_max_queue_size')
                final_queue_size_increase_count = extracted_data.get('final_queue_size_increase_count')
                print(f"    Loss: {loss_percentage}%")
                if out_of_thread_percentage is not None:
                    print(f"    Out-of-thread: {out_of_thread_events:,} events ({out_of_thread_percentage:.2f}%)")
                if all_without_locks_events is not None:
                    self.vprint(f"    All-without-locks: {all_without_locks_events:,} events")
                if final_max_queue_size is not None:
                    print(f"    Max Queue Size: {final_max_queue_size}")
                if final_queue_size_increase_count is not None:
                    print(f"    Queue Size Increases: {final_queue_size_increase_count}")

                # Check for missing queue statistics when dynamic queue sizing is enabled
                if self.dynamic_queue_size and (final_max_queue_size is None or final_queue_size_increase_count is None):
                    print(f"    ⚠️  WARNING: Dynamic queue sizing enabled but queue statistics missing!")
                    print(f"       MAX_QUEUE_SIZE_SUM found: {'✅' if final_max_queue_size is not None else '❌'}")
                    print(f"       QUEUE_SIZE_INCREASE_COUNT found: {'✅' if final_queue_size_increase_count is not None else '❌'}")
                    print(f"       This suggests the JFR output may not contain queue size statistics")

            else:
                # Fallback for old format
                loss_percentage = extracted_data
                out_of_thread_events = None
                out_of_thread_percentage = None
                out_of_thread_details = None
                all_without_locks_events = None
                all_without_locks_details = None
                final_max_queue_size = None
                final_queue_size_increase_count = None
                print(f"    Loss: {loss_percentage}%")

            # Extract additional loss kind data if available
            loss_kinds = extracted_data.get('loss_kinds', {})
            loss_kind_percentages = extracted_data.get('loss_kind_percentages', {})
            total_lost_samples = extracted_data.get('total_lost_samples', 0)
            main_loss_percentage = extracted_data.get('main_loss_percentage', 0)

            result_data = {
                'queue_size': queue_size,
                'interval': interval,
                'stack_depth': stack_depth,
                'native_duration': native_duration,
                'test_duration': self.test_duration,
                'loss_percentage': loss_percentage,
                'loss_kinds': loss_kinds,
                'loss_kind_percentages': loss_kind_percentages,
                'total_lost_samples': total_lost_samples,
                'main_loss_percentage': main_loss_percentage,
                'out_of_thread_events': out_of_thread_events,
                'out_of_thread_percentage': out_of_thread_percentage,
                'out_of_thread_details': out_of_thread_details,
                'all_without_locks_events': all_without_locks_events,
                'all_without_locks_details': all_without_locks_details,
                'final_max_queue_size': final_max_queue_size if final_max_queue_size is not None else queue_size,  # fallback to original queue size
                'final_queue_size_increase_count': final_queue_size_increase_count if final_queue_size_increase_count is not None else 0,  # fallback to 0
                'log_file': str(log_filename),
                'timestamp': datetime.now().isoformat(),
                'success': result.returncode == 0 and loss_percentage is not None
            }

            return result_data
        except Exception as e:
            print(f"    ERROR: {e}")
            return {
                'queue_size': queue_size,
                'interval': interval,
                'stack_depth': stack_depth,
                'native_duration': native_duration,
                'test_duration': self.test_duration,
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_events': None,
                'all_without_locks_details': None,
                'log_file': str(log_filename),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }

    def run_renaissance_test(self, queue_size: int, interval: str) -> Dict:
        """Run a single Renaissance test with retry logic for JSON parsing failures"""
        return self.run_test_with_retry(self._run_renaissance_test_internal, queue_size, interval)

    def _run_renaissance_test_internal(self, queue_size: int, interval: str) -> Dict:
        """Run a single Renaissance test and extract loss percentage"""
        print(f"  Running Renaissance: queue={queue_size}, interval={interval}")

        log_filename = f"renaissance_q{queue_size}_{interval}_n{self.renaissance_iterations}.log"
        log_path = LOGS_DIR / log_filename

        print(f"    📝 Log file: {log_path}")

        # Build command (no stack depth for Renaissance)
        cmd = [
            "./run.sh",
            "--mode", "renaissance",
            "-n", str(self.renaissance_iterations),
            "-t", str(self.threads),
            "-i", interval,
            "-q", str(queue_size),
            "--no-analysis",  # Enable analysis tables but skip plots/visualizations
            "-f", str(log_path.absolute()) + ".raw",
        ]

        if self.dynamic_queue_size:
            cmd.append("--dynamic-queue-size")

        # Generate log filename


        # Run test
        try:
            # Set up environment variables
            env = os.environ.copy()
            if self.dynamic_queue_size:
                env['DYNAMIC_QUEUE_SIZE'] = 'true'
                self.vprint(f"    Environment: DYNAMIC_QUEUE_SIZE=true")

            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd="../run_in_native",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            # Extract loss percentage from log
            extracted_data = self.extract_loss_percentage(log_path)

            # Handle both old (float) and new (dict) format
            if isinstance(extracted_data, dict):
                loss_percentage = extracted_data.get('loss_percentage')
                out_of_thread_events = extracted_data.get('out_of_thread_events')
                out_of_thread_percentage = extracted_data.get('out_of_thread_percentage')
                out_of_thread_details = extracted_data.get('out_of_thread_details')
                all_without_locks_events = extracted_data.get('all_without_locks_events')
                all_without_locks_details = extracted_data.get('all_without_locks_details')
                final_max_queue_size = extracted_data.get('final_max_queue_size')
                final_queue_size_increase_count = extracted_data.get('final_queue_size_increase_count')
                print(f"    Loss: {loss_percentage}%")
                if out_of_thread_percentage is not None:
                    print(f"    Out-of-thread: {out_of_thread_events:,} events ({out_of_thread_percentage:.2f}%)")
                if all_without_locks_events is not None:
                    self.vprint(f"    All-without-locks: {all_without_locks_events:,} events")
                if final_max_queue_size is not None:
                    print(f"    Max Queue Size: {final_max_queue_size}")
                if final_queue_size_increase_count is not None:
                    print(f"    Queue Size Increases: {final_queue_size_increase_count}")

                # Check for missing queue statistics when dynamic queue sizing is enabled
                if self.dynamic_queue_size and (final_max_queue_size is None or final_queue_size_increase_count is None):
                    print(f"    ⚠️  WARNING: Dynamic queue sizing enabled but queue statistics missing!")
                    print(f"       MAX_QUEUE_SIZE_SUM found: {'✅' if final_max_queue_size is not None else '❌'}")
                    print(f"       QUEUE_SIZE_INCREASE_COUNT found: {'✅' if final_queue_size_increase_count is not None else '❌'}")
                    print(f"       This suggests the JFR output may not contain queue size statistics")

            else:
                loss_percentage = extracted_data
                out_of_thread_events = None
                out_of_thread_percentage = None
                out_of_thread_details = None
                all_without_locks_events = None
                all_without_locks_details = None
                final_max_queue_size = None
                final_queue_size_increase_count = None
                print(f"    Loss: {loss_percentage}%")

            # Extract additional loss kind data if available (for Renaissance tests too)
            if isinstance(extracted_data, dict):
                loss_kinds = extracted_data.get('loss_kinds', {})
                loss_kind_percentages = extracted_data.get('loss_kind_percentages', {})
                total_lost_samples = extracted_data.get('total_lost_samples', 0)
                main_loss_percentage = extracted_data.get('main_loss_percentage', 0)
            else:
                loss_kinds = {}
                loss_kind_percentages = {}
                total_lost_samples = 0
                main_loss_percentage = 0

            result_data = {
                'queue_size': queue_size,
                'interval': interval,
                'iterations': self.renaissance_iterations,
                'loss_percentage': loss_percentage,
                'loss_kinds': loss_kinds,
                'loss_kind_percentages': loss_kind_percentages,
                'total_lost_samples': total_lost_samples,
                'main_loss_percentage': main_loss_percentage,
                'out_of_thread_events': out_of_thread_events,
                'out_of_thread_percentage': out_of_thread_percentage,
                'out_of_thread_details': out_of_thread_details,
                'all_without_locks_events': all_without_locks_events,
                'all_without_locks_details': all_without_locks_details,
                'final_max_queue_size': final_max_queue_size if final_max_queue_size is not None else queue_size,  # fallback to original queue size
                'final_queue_size_increase_count': final_queue_size_increase_count if final_queue_size_increase_count is not None else 0,  # fallback to 0
                'log_file': str(log_filename),
                'timestamp': datetime.now().isoformat(),
                'success': result.returncode == 0
            }

            return result_data
        except Exception as e:
            print(f"    ERROR: {e}")
            return {
                'queue_size': queue_size,
                'interval': interval,
                'iterations': self.renaissance_iterations,
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_events': None,
                'all_without_locks_details': None,
                'final_max_queue_size': queue_size,  # fallback to original queue size
                'final_queue_size_increase_count': 0,  # fallback to 0
                'log_file': str(log_filename),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }

    def extract_queue_size_percentiles(self, content):
        """Extract queue size percentile data from QUEUE SIZE ANALYSIS section"""
        try:
            # Look for the QUEUE SIZE ANALYSIS section (with or without emoji prefix)
            analysis_pattern = r'(?:🗂️\s*)?QUEUE SIZE ANALYSIS - Event-Weighted Percentiles\s*-*\s*(.*?)(?=\n\s*={10,}|\n\s*(?:🗂️\s*)?[A-Z][A-Z\s]+-+|\n\s*$)'
            section_match = re.search(analysis_pattern, content, re.DOTALL)

            if not section_match:
                self.vprint("    ⚠️ Could not find 'QUEUE SIZE ANALYSIS - Event-Weighted Percentiles' section")
                return {}

            analysis_text = section_match.group(1)
            self.vprint(f"    📊 Found queue size analysis section ({len(analysis_text)} chars)")

            queue_stats = {}

            # Look for category headers and their corresponding data
            # Pattern matches lines like "Category: gc" followed by percentile data
            category_pattern = r'Category:\s+(\w+)'
            percentile_pattern = r'(\d+(?:\.\d+)?)%:\s*([\d,]+)'
            events_pattern = r'Total Events:\s*([\d,]+)'

            # Split by lines and process
            lines = analysis_text.split('\n')
            current_category = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for category header
                cat_match = re.search(category_pattern, line)
                if cat_match:
                    current_category = cat_match.group(1)
                    queue_stats[current_category] = {}
                    continue

                # Check for total events
                events_match = re.search(events_pattern, line)
                if events_match and current_category:
                    total_events = int(events_match.group(1).replace(',', ''))
                    queue_stats[current_category]['total_events'] = total_events
                    continue

                # Check for percentile data
                perc_matches = re.findall(percentile_pattern, line)
                if perc_matches and current_category:
                    for perc, value in perc_matches:
                        percentile_key = f"p{perc.replace('.', '_')}"
                        queue_size = int(value.replace(',', ''))
                        queue_stats[current_category][percentile_key] = queue_size

            if queue_stats:
                self.vprint(f"    📊 Extracted percentiles for categories: {list(queue_stats.keys())}")
                for category, stats in queue_stats.items():
                    percentiles = [k for k in stats.keys() if k.startswith('p')]
                    events = stats.get('total_events', 'N/A')
                    self.vprint(f"      {category}: {len(percentiles)} percentiles, {events:,} total events")

            return queue_stats

        except Exception as e:
            self.vprint(f"    ⚠️ Error extracting queue size percentiles: {e}")
            return {}

    def parse_drain_stats_with_analysis_script(self, log_file_path: Path):
        """Parse drain statistics using the analysis script's parsing logic"""
        if not ANALYSIS_AVAILABLE:
            self.vprint("    ⚠️ Analysis script functions not available, falling back to basic parsing")
            # Still parse signal handler stats even without the analysis script
            signal_handler_stats = []
            try:
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if 'SIGNAL_HANDLER_STATS_JSON:' in line:
                            match = re.search(r'SIGNAL_HANDLER_STATS_JSON:\s*(\{.*\})', line)
                            if match:
                                try:
                                    data = json.loads(match.group(1))
                                    signal_handler_stats.append(data)
                                except json.JSONDecodeError as e:
                                    self.vprint(f"    ⚠️ Error parsing SIGNAL_HANDLER_STATS_JSON: {e}")

                result = {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None, 'signal_handler_stats': {}}

                # Process signal handler statistics
                for signal_stats in signal_handler_stats:
                    signal_name = signal_stats.get('name', 'unknown')
                    signals_processed = signal_stats.get('signals_processed', 0)
                    handler_time = signal_stats.get('handler_time', {})

                    signal_data = {
                        'signals_processed': signals_processed,
                        'runtime_ns': signal_stats.get('runtime_ns', 0),
                        'runtime_seconds': signal_stats.get('runtime_seconds', 0.0),
                        'handler_time': {
                            'sum': handler_time.get('sum', 0),
                            'avg': handler_time.get('avg', 0),
                            'min': handler_time.get('min', 0),
                            'max': handler_time.get('max', 0),
                            'median': handler_time.get('median', 0),
                            'p90': handler_time.get('p90', 0),
                            'p95': handler_time.get('p95', 0),
                            'p99': handler_time.get('p99', 0),
                            'p99_9': handler_time.get('p99_9', 0)
                        },
                        'handler_time_histogram': signal_stats.get('handler_time_histogram', [])
                    }

                    result['signal_handler_stats'][signal_name] = signal_data
                    self.vprint(f"    🎯 Signal Handler [{signal_name}]: {signals_processed:,} signals, avg={handler_time.get('avg', 0)}ns, p99={handler_time.get('p99', 0)}ns")

                return result
            except Exception as e:
                self.vprint(f"    ⚠️ Error in fallback parsing: {e}")
                return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

        try:
            self.vprint(f"    🔍 Using analysis script to parse: {log_file_path}")

            # Read all DRAIN_STATS_JSON and SIGNAL_HANDLER_STATS_JSON lines from the log file
            all_stats = []
            signal_handler_stats = []
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'DRAIN_STATS_JSON:' in line:
                        match = re.search(r'DRAIN_STATS_JSON:\s*(\{.*\})', line)
                        if match:
                            try:
                                data = json.loads(match.group(1))
                                all_stats.append(data)
                            except json.JSONDecodeError as e:
                                self.vprint(f"    ⚠️ Error parsing DRAIN_STATS_JSON: {e}")
                                continue
                    elif 'SIGNAL_HANDLER_STATS_JSON:' in line:
                        match = re.search(r'SIGNAL_HANDLER_STATS_JSON:\s*(\{.*\})', line)
                        if match:
                            try:
                                data = json.loads(match.group(1))
                                signal_handler_stats.append(data)
                            except json.JSONDecodeError as e:
                                self.vprint(f"    ⚠️ Error parsing SIGNAL_HANDLER_STATS_JSON: {e}")
                                continue

            if not all_stats:
                self.vprint(f"    ⚠️ No DRAIN_STATS_JSON found in {log_file_path}")
                return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

            self.vprint(f"    📊 Found {len(all_stats)} drain statistics entries")
            if signal_handler_stats:
                self.vprint(f"    🎯 Found {len(signal_handler_stats)} signal handler statistics entries")

            # Extract drain categories and calculate event counts
            result = {
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_events': None,
                'all_without_locks_details': None,
                'drain_categories': {},
                'signal_handler_stats': {}
            }

            total_events_without_locks = 0
            total_events_all = 0
            out_of_thread_events = 0
            all_without_locks_events = 0

            # Process each drain category from the JSON data
            for category_name in DRAIN_CATEGORIES:
                # Find matching entry in all_stats
                category_stats = None
                for stats in all_stats:
                    if stats.get('name') == category_name:
                        category_stats = stats
                        break

                if category_stats:
                    # Extract event information
                    events_data = category_stats.get('events', {})
                    events_sum = events_data.get('sum', 0)

                    # Extract time information
                    time_data = category_stats.get('time', {})

                    # Build category data structure - use time percentiles directly from JSON
                    category_data = {
                        'requests': category_stats.get('drains', 0),
                        'runtime_seconds': category_stats.get('runtime_ns', 0) / 1000000000.0 if 'runtime_ns' in category_stats else 0.0,  # Convert ns to seconds
                        'runtime_minutes': (category_stats.get('runtime_ns', 0) / 1000000000.0) / 60.0 if 'runtime_ns' in category_stats else 0.0,
                        'request_rate': category_stats.get('drains', 0) / (category_stats.get('runtime_ns', 1) / 1000000000.0) if category_stats.get('runtime_ns', 0) > 0 else 0.0,
                        'events': {
                            'sum': events_sum,
                            'avg': events_data.get('average', events_data.get('avg', 0)),
                            'min': events_data.get('min', 0),
                            'max': events_data.get('max', 0),
                            'median': events_data.get('50th', events_data.get('median', 0)),
                            'p95': events_data.get('95th', events_data.get('p95', 0)),
                            'p99': events_data.get('99th', events_data.get('p99', 0)),
                            'p99_9': events_data.get('p99_9', events_data.get('99.9th', events_data.get('p99.9', 0)))
                        },
                        'time_ns': {
                            'sum': time_data.get('sum', 0),
                            'avg': time_data.get('average', time_data.get('avg', 0)),
                            'min': time_data.get('min', 0),
                            'max': time_data.get('max', 0),
                            'median': time_data.get('50th', time_data.get('median', 0)),
                            'p95': time_data.get('95th', time_data.get('p95', 0)),
                            'p99': time_data.get('99th', time_data.get('p99', 0)),
                            'p99_9': time_data.get('p99_9', time_data.get('99.9th', time_data.get('p99.9', 0)))
                        },
                        'queue_size_percentiles': {
                            'median': time_data.get('median', 0),
                            'p90': time_data.get('p90', 0),
                            'p95': time_data.get('p95', 0),
                            'p99': time_data.get('p99', 0),
                            'p99_9': time_data.get('p99_9', 0),
                            'p99_99': time_data.get('p99_99', 0),
                            'p99_999': time_data.get('p99_999', 0)
                        }
                    }

                    result['drain_categories'][category_name] = category_data
                    total_events_all += events_sum

                    # Count events only from "without locks" categories
                    if 'without locks' in category_name.lower():
                        total_events_without_locks += events_sum
                        self.vprint(f"    📊 {category_name}: {events_sum:,} events (counted in without-locks total)")
                    else:
                        self.vprint(f"    📊 {category_name}: {events_sum:,} events (excluded from without-locks total)")

                    # Track specific categories
                    if 'out of thread' in category_name.lower():
                        out_of_thread_events += events_sum
                        result['out_of_thread_details'] = category_data
                    elif 'all without locks' in category_name.lower():
                        all_without_locks_events += events_sum
                        result['all_without_locks_details'] = category_data

            # Set extracted values
            result['out_of_thread_events'] = out_of_thread_events if out_of_thread_events > 0 else None
            result['all_without_locks_events'] = all_without_locks_events if all_without_locks_events > 0 else None

            # Calculate percentages using only "without locks" events for the denominator
            if total_events_without_locks > 0 and out_of_thread_events > 0:
                result['out_of_thread_percentage'] = (out_of_thread_events / total_events_without_locks) * 100
                print(f"    📊 Out-of-thread: {out_of_thread_events:,}/{total_events_without_locks:,} (without-locks only) = {result['out_of_thread_percentage']:.6f}%")

            print(f"    📋 Total events summary:")
            print(f"      Without-locks categories: {total_events_without_locks:,} events")
            print(f"      All categories: {total_events_all:,} events")
            print(f"    📋 Parsed {len(result['drain_categories'])} drain categories")

            # Parse JFR sample statistics for loss percentage (same logic as before)
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            successful_match = re.search(r'Successful Samples:\s*([\d,]+)', content)
            total_match = re.search(r'Total Samples:\s*([\d,]+)', content)
            lost_match = re.search(r'Lost Samples:\s*([\d,]+)', content)

            if successful_match and total_match and lost_match:
                successful = int(successful_match.group(1).replace(',', ''))
                total = int(total_match.group(1).replace(',', ''))
                lost = int(lost_match.group(1).replace(',', ''))

                if (total + lost) > 0:
                    loss_percentage = (lost / (total + lost)) * 100
                    result['loss_percentage'] = loss_percentage
                    self.vprint(f"    📊 JFR Loss: {lost:,}/{total + lost:,} = {loss_percentage:.2f}%")

            # Process signal handler statistics
            for signal_stats in signal_handler_stats:
                signal_name = signal_stats.get('name', 'unknown')
                signals_processed = signal_stats.get('signals_processed', 0)
                handler_time = signal_stats.get('handler_time', {})

                # Build signal handler data structure
                signal_data = {
                    'signals_processed': signals_processed,
                    'runtime_ns': signal_stats.get('runtime_ns', 0),
                    'runtime_seconds': signal_stats.get('runtime_seconds', 0.0),
                    'handler_time': {
                        'sum': handler_time.get('sum', 0),
                        'avg': handler_time.get('avg', 0),
                        'min': handler_time.get('min', 0),
                        'max': handler_time.get('max', 0),
                        'median': handler_time.get('median', 0),
                        'p90': handler_time.get('p90', 0),
                        'p95': handler_time.get('p95', 0),
                        'p99': handler_time.get('p99', 0),
                        'p99_9': handler_time.get('p99_9', 0)
                    },
                    'handler_time_histogram': signal_stats.get('handler_time_histogram', [])
                }

                result['signal_handler_stats'][signal_name] = signal_data
                self.vprint(f"    🎯 Signal Handler [{signal_name}]: {signals_processed:,} signals, avg={handler_time.get('avg', 0)}ns, p99={handler_time.get('p99', 0)}ns")

            return result

        except Exception as e:
            self.vprint(f"    ⚠️ Error parsing with analysis script: {e}")
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

    def parse_raw_drain_stats(self, raw_file_path: Path):
        """Parse comprehensive drain statistics from .raw file"""
        try:
            with open(raw_file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            self.vprint(f"    📊 Parsing drain statistics from .raw file ({len(content):,} chars)")

            result = {
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_events': None,
                'all_without_locks_details': None,
                'drain_categories': {}
            }

            # Parse all drain statistics sections
            drain_sections = re.findall(r'=== (.+?) Drain Statistics ===\s*(.*?)(?=\n===|\nJFR_SAMPLE_STATISTICS|\nDRAIN_STATS_JSON|\Z)', content, re.DOTALL)

            total_events_without_locks = 0  # Count only "without locks" categories
            total_events_all = 0  # Count all categories for reference
            out_of_thread_events = 0
            all_without_locks_events = 0

            for category_name, section_content in drain_sections:
                self.vprint(f"    📋 Processing drain category: {category_name}")

                # Parse basic statistics
                requests_match = re.search(r'Requests:\s*(\d+)', section_content)
                runtime_match = re.search(r'Runtime:\s*([\d.]+)\s*seconds\s*\(([\d.]+)\s*minutes\)', section_content)
                rate_match = re.search(r'Request Rate:\s*([\d.]+)\s*requests/second', section_content)

                # Parse time and event statistics
                time_match = re.search(r'Time \(ns\):\s*sum=(\d+),\s*avg=(\d+),\s*min=(\d+),\s*max=(\d+),\s*median=(\d+),\s*p95=(\d+),\s*p99=(\d+),\s*p99\.9=(\d+)', section_content)
                events_match = re.search(r'Events:\s*sum=(\d+),\s*avg=([\d.]+),\s*min=(\d+),\s*max=(\d+),\s*median=(\d+),\s*p95=(\d+),\s*p99=(\d+),\s*p99\.9=(\d+)', section_content)

                category_data = {
                    'requests': int(requests_match.group(1)) if requests_match else 0,
                    'runtime_seconds': float(runtime_match.group(1)) if runtime_match else 0.0,
                    'runtime_minutes': float(runtime_match.group(2)) if runtime_match else 0.0,
                    'request_rate': float(rate_match.group(1)) if rate_match else 0.0
                }

                if time_match:
                    category_data['time_ns'] = {
                        'sum': int(time_match.group(1)),
                        'avg': int(time_match.group(2)),
                        'min': int(time_match.group(3)),
                        'max': int(time_match.group(4)),
                        'median': int(time_match.group(5)),
                        'p95': int(time_match.group(6)),
                        'p99': int(time_match.group(7)),
                        'p99_9': int(time_match.group(8))
                    }

                if events_match:
                    events_sum = int(events_match.group(1))
                    category_data['events'] = {
                        'sum': events_sum,
                        'avg': float(events_match.group(2)),
                        'min': int(events_match.group(3)),
                        'max': int(events_match.group(4)),
                        'median': int(events_match.group(5)),
                        'p95': int(events_match.group(6)),
                        'p99': int(events_match.group(7)),
                        'p99_9': int(events_match.group(8))
                    }

                    total_events_all += events_sum

                    # Only count events from "without locks" categories
                    if 'without locks' in category_name.lower():
                        total_events_without_locks += events_sum
                        self.vprint(f"      {category_name}: {events_sum:,} events (counted in without-locks total)")
                    else:
                        self.vprint(f"      {category_name}: {events_sum:,} events (excluded from without-locks total)")

                    # Track specific categories
                    if 'out of thread' in category_name.lower():
                        out_of_thread_events += events_sum
                        result['out_of_thread_details'] = category_data
                    elif 'all without locks' in category_name.lower():
                        all_without_locks_events += events_sum
                        result['all_without_locks_details'] = category_data

                result['drain_categories'][category_name] = category_data

            # Set extracted values
            result['out_of_thread_events'] = out_of_thread_events if out_of_thread_events > 0 else None
            result['all_without_locks_events'] = all_without_locks_events if all_without_locks_events > 0 else None

            # Calculate percentages using only "without locks" events for the denominator
            if total_events_without_locks > 0 and out_of_thread_events > 0:
                result['out_of_thread_percentage'] = (out_of_thread_events / total_events_without_locks) * 100
                print(f"    📊 Out-of-thread: {out_of_thread_events:,}/{total_events_without_locks:,} (without-locks only) = {result['out_of_thread_percentage']:.6f}%")

            print(f"    📋 Total events summary:")
            print(f"      Without-locks categories: {total_events_without_locks:,} events")
            print(f"      All categories: {total_events_all:,} events")
            print(f"    📋 Parsed {len(result['drain_categories'])} drain categories")

            # Parse JFR sample statistics for loss percentage
            jfr_match = re.search(r'JFR_SAMPLE_STATISTICS:(.*?)JFR_SAMPLE_STATISTICS_END', content, re.DOTALL)
            if jfr_match:
                jfr_content = jfr_match.group(1)
                successful_match = re.search(r'Successful Samples:\s*([\d,]+)', jfr_content)
                total_match = re.search(r'Total Samples:\s*([\d,]+)', jfr_content)
                lost_match = re.search(r'Lost Samples:\s*([\d,]+)', jfr_content)

                if successful_match and total_match and lost_match:
                    successful = int(successful_match.group(1).replace(',', ''))
                    total = int(total_match.group(1).replace(',', ''))
                    lost = int(lost_match.group(1).replace(',', ''))

                    if (total + lost) > 0:
                        loss_percentage = (lost / (total + lost)) * 100
                        result['loss_percentage'] = loss_percentage
                        self.vprint(f"    📊 JFR Loss: {lost:,}/{total + lost:,} = {loss_percentage:.2f}%")

            return result

        except Exception as e:
            self.vprint(f"    ⚠️ Error parsing .raw drain statistics: {e}")
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

    def extract_loss_percentage(self, log_path: Path):
        """Extract loss percentage, out-of-thread statistics, and queue size statistics from log file"""
        try:
            # Initialize result to store all data we find
            result = {
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_details': None,
                'drain_categories': {},
                'total_events_all': None,
                'total_events_without_locks': None,
                'loss_kinds': {}
            }

            # First try to get drain statistics from .raw file if it exists
            raw_file_path = Path(str(log_path) + ".raw")
            if raw_file_path.exists():
                self.vprint(f"    🔍 Found .raw file: {raw_file_path}")

                # Prioritize using the analysis script on the .raw file since it contains DRAIN_STATS_JSON
                if ANALYSIS_AVAILABLE:
                    self.vprint(f"    🔍 Using analysis script on .raw file: {raw_file_path}")
                    raw_result = self.parse_drain_stats_with_analysis_script(raw_file_path)
                    if raw_result.get('out_of_thread_events') is not None or raw_result.get('loss_percentage') is not None:
                        # Update result with raw file data
                        result.update(raw_result)
                    else:
                        self.vprint(f"    ⚠️ Analysis script didn't find data in .raw file, falling back to raw parsing...")
                        raw_result = self.parse_raw_drain_stats(raw_file_path)
                        result.update(raw_result)
                else:
                    raw_result = self.parse_raw_drain_stats(raw_file_path)
                    result.update(raw_result)

            # If no .raw file, try using the analysis script on the main log file for drain stats
            elif ANALYSIS_AVAILABLE:
                self.vprint(f"    🔍 No .raw file found, trying analysis script on main log: {log_path}")
                analysis_result = self.parse_drain_stats_with_analysis_script(log_path)
                if analysis_result.get('out_of_thread_events') is not None or analysis_result.get('loss_percentage') is not None:
                    result.update(analysis_result)
                else:
                    self.vprint(f"    ⚠️ Analysis script didn't find data in main log, will continue with basic parsing...")

            # Now ALWAYS parse the main log file for LOST_SAMPLE_STATS and other data
            # Try different encodings to handle special characters
            content = None
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(log_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    self.vprint(f"    📝 Successfully read main log with {encoding} encoding: {log_path}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                # Fallback: read as binary and decode with error replacement
                with open(log_path, 'rb') as f:
                    raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace')
                self.vprint(f"    📝 Read main log with fallback binary mode: {log_path}")

            self.vprint(f"    📏 Main log size: {len(content):,} characters")

            # Extract comprehensive "out of thread" statistics from Drain Statistics sections
            # Look for the complete "=== out of thread Drain Statistics ===" section
            out_of_thread_section_pattern = r'=== out of thread Drain Statistics ===\s*(.*?)(?=\n===|\nAnalyzing|\nDRAIN_STATS_JSON|\n\[|\Z)'
            out_of_thread_match = re.search(out_of_thread_section_pattern, content, re.DOTALL)

            if out_of_thread_match:
                section_content = out_of_thread_match.group(1)
                self.vprint(f"    📊 Found out-of-thread section ({len(section_content)} chars)")

                # Parse detailed statistics from the section
                out_of_thread_details = {}

                # Parse Requests
                requests_match = re.search(r'Requests:\s*(\d+)', section_content)
                if requests_match:
                    out_of_thread_details['requests'] = int(requests_match.group(1))

                # Parse Runtime
                runtime_match = re.search(r'Runtime:\s*([\d.]+)\s*seconds\s*\(([\d.]+)\s*minutes\)', section_content)
                if runtime_match:
                    out_of_thread_details['runtime_seconds'] = float(runtime_match.group(1))
                    out_of_thread_details['runtime_minutes'] = float(runtime_match.group(2))

                # Parse Request Rate
                rate_match = re.search(r'Request Rate:\s*([\d.]+)\s*requests/second', section_content)
                if rate_match:
                    out_of_thread_details['request_rate'] = float(rate_match.group(1))

                # Parse Time statistics (ns)
                time_match = re.search(r'Time \(ns\):\s*sum=(\d+),\s*avg=(\d+),\s*min=(\d+),\s*max=(\d+),\s*median=(\d+),\s*p95=(\d+),\s*p99=(\d+),\s*p99\.9=(\d+)', section_content)
                if time_match:
                    out_of_thread_details['time_ns'] = {
                        'sum': int(time_match.group(1)),
                        'avg': int(time_match.group(2)),
                        'min': int(time_match.group(3)),
                        'max': int(time_match.group(4)),
                        'median': int(time_match.group(5)),
                        'p95': int(time_match.group(6)),
                        'p99': int(time_match.group(7)),
                        'p99_9': int(time_match.group(8))
                    }

                # Parse Events
                events_match = re.search(r'Events:\s*sum=(\d+)', section_content)
                if events_match:
                    out_of_thread_events = int(events_match.group(1))
                    out_of_thread_details['events'] = out_of_thread_events
                    result['out_of_thread_events'] = out_of_thread_events

                    self.vprint(f"    📊 Parsed out-of-thread details:")
                    self.vprint(f"       Requests: {out_of_thread_details.get('requests', 'N/A')}")
                    self.vprint(f"       Runtime: {out_of_thread_details.get('runtime_seconds', 'N/A')}s ({out_of_thread_details.get('runtime_minutes', 'N/A')} min)")
                    self.vprint(f"       Request Rate: {out_of_thread_details.get('request_rate', 'N/A')} req/s")
                    self.vprint(f"       Events: {out_of_thread_events:,}")
                    if 'time_ns' in out_of_thread_details:
                        time_stats = out_of_thread_details['time_ns']
                        self.vprint(f"       Time (ns): avg={time_stats['avg']:,}, median={time_stats['median']:,}, p95={time_stats['p95']:,}")

                result['out_of_thread_details'] = out_of_thread_details
            else:
                self.vprint(f"    ⚠️ Could not find 'out of thread Drain Statistics' section")

            # Extract "all without locks" statistics from Drain Statistics sections
            # Look for the complete "=== all without locks Drain Statistics ===" section
            all_without_locks_section_pattern = r'=== all without locks Drain Statistics ===\s*(.*?)(?=\n===|\nAnalyzing|\nDRAIN_STATS_JSON|\n\[|\Z)'
            all_without_locks_match = re.search(all_without_locks_section_pattern, content, re.DOTALL)

            if all_without_locks_match:
                section_content = all_without_locks_match.group(1)
                self.vprint(f"    📊 Found all-without-locks section ({len(section_content)} chars)")

                # Parse detailed statistics from the section
                all_without_locks_details = {}

                # Parse Requests
                requests_match = re.search(r'Requests:\s*(\d+)', section_content)
                if requests_match:
                    all_without_locks_details['requests'] = int(requests_match.group(1))

                # Parse Runtime
                runtime_match = re.search(r'Runtime:\s*([\d.]+)\s*seconds\s*\(([\d.]+)\s*minutes\)', section_content)
                if runtime_match:
                    all_without_locks_details['runtime_seconds'] = float(runtime_match.group(1))
                    all_without_locks_details['runtime_minutes'] = float(runtime_match.group(2))

                # Parse Request Rate
                rate_match = re.search(r'Request Rate:\s*([\d.]+)\s*requests/second', section_content)
                if rate_match:
                    all_without_locks_details['request_rate'] = float(rate_match.group(1))

                # Parse Time statistics (ns)
                time_match = re.search(r'Time \(ns\):\s*sum=(\d+),\s*avg=(\d+),\s*min=(\d+),\s*max=(\d+),\s*median=(\d+),\s*p95=(\d+),\s*p99=(\d+),\s*p99\.9=(\d+)', section_content)
                if time_match:
                    all_without_locks_details['time_ns'] = {
                        'sum': int(time_match.group(1)),
                        'avg': int(time_match.group(2)),
                        'min': int(time_match.group(3)),
                        'max': int(time_match.group(4)),
                        'median': int(time_match.group(5)),
                        'p95': int(time_match.group(6)),
                        'p99': int(time_match.group(7)),
                        'p99_9': int(time_match.group(8))
                    }

                # Parse Events
                events_match = re.search(r'Events:\s*sum=(\d+)', section_content)
                if events_match:
                    all_without_locks_events = int(events_match.group(1))
                    all_without_locks_details['events'] = all_without_locks_events
                    result['all_without_locks_events'] = all_without_locks_events

                    self.vprint(f"    📊 Parsed all-without-locks details:")
                    self.vprint(f"       Requests: {all_without_locks_details.get('requests', 'N/A'):,}")
                    self.vprint(f"       Runtime: {all_without_locks_details.get('runtime_seconds', 'N/A')}s ({all_without_locks_details.get('runtime_minutes', 'N/A')} min)")
                    self.vprint(f"       Request Rate: {all_without_locks_details.get('request_rate', 'N/A'):,} req/s")
                    self.vprint(f"       Events: {all_without_locks_events:,}")
                    if 'time_ns' in all_without_locks_details:
                        time_stats = all_without_locks_details['time_ns']
                        self.vprint(f"       Time (ns): sum={time_stats['sum']:,}, avg={time_stats['avg']:,}, median={time_stats['median']:,}")

                result['all_without_locks_details'] = all_without_locks_details
            else:
                self.vprint(f"    ⚠️ Could not find 'all without locks Drain Statistics' section")

            # Calculate out-of-thread percentage using all drain statistics sections
            all_events_pattern = r'Events:\s*sum=(\d+)'
            all_events_matches = re.findall(all_events_pattern, content)

            if len(all_events_matches) >= 1:
                total_events = sum(int(match) for match in all_events_matches)
                self.vprint(f"    📊 Found {len(all_events_matches)} drain statistics sections with total events: {total_events:,}")

                if result.get('out_of_thread_events') and total_events > 0:
                    out_of_thread_percentage = (result['out_of_thread_events'] / total_events) * 100
                    result['out_of_thread_percentage'] = out_of_thread_percentage
                    self.vprint(f"    📊 Calculated out-of-thread percentage: {out_of_thread_percentage:.6f}% ({result['out_of_thread_events']:,}/{total_events:,})")
                elif total_events == 0:
                    self.vprint(f"    ⚠️ Total events is 0, cannot calculate percentage")
            else:
                self.vprint(f"    ⚠️ Could not find any drain statistics sections with events")

            # Parse LOSS_PERCENTAGE output from run.sh (if available)
            print(f"    🔍 Searching for LOSS_PERCENTAGE from run.sh...")
            loss_percentage_pattern = r'LOSS_PERCENTAGE:\s*([\d.]+)'
            loss_percentage_match = re.search(loss_percentage_pattern, content)
            if loss_percentage_match:
                result['loss_percentage'] = float(loss_percentage_match.group(1))
                print(f"    ✅ Found LOSS_PERCENTAGE from run.sh: {result['loss_percentage']:.3f}%")
            else:
                print(f"    ⚠️ No LOSS_PERCENTAGE found from run.sh, will calculate from JFR statistics")

            # Parse LOST_SAMPLE_STATS for detailed loss kind breakdown
            print(f"    🔍 Searching for LOST_SAMPLE_STATS lines...")
            lost_sample_stats_pattern = r'LOST_SAMPLE_STATS:\s*(.+)'
            lost_stats_matches = re.findall(lost_sample_stats_pattern, content)

            if lost_stats_matches:
                print(f"    📊 Found {len(lost_stats_matches)} LOST_SAMPLE_STATS entries")

                # Parse the last entry (most recent/complete statistics)
                last_stats_line = lost_stats_matches[-1]

                # Initialize loss kind statistics
                loss_kinds = {
                    'stw_gc': 0,
                    'invalid_state': 0,
                    'could_not_acquire_lock': 0,
                    'enqueue_failed': 0,
                    'other': 0  # sum of all other loss types (excluding VM ops)
                }

                # Initialize VM operations tracking
                vm_operations = {
                    'no_vm_ops': 0,
                    'in_jfr_safepoint': 0,
                    'vm_op_handshakeallthreads': 0,
                    'vm_op_g1collectforfullgc': 0,
                    'vm_op_g1incremental': 0,
                    'vm_op_cleanup': 0,
                    'vm_op_other': 0  # sum of all other VM operations
                }

                # Parse each key=value pair
                pairs = last_stats_line.split()
                total_lost_samples = 0

                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        try:
                            value = int(value)
                            total_lost_samples += value

                            if key in loss_kinds:
                                loss_kinds[key] = value
                            elif key in ['no_vm_ops', 'in_jfr_safepoint']:
                                vm_operations[key] = value
                            elif key.startswith('vm_op_'):
                                # Convert to lowercase for matching
                                key_lower = key.lower()
                                if key_lower in vm_operations:
                                    vm_operations[key_lower] = value
                                else:
                                    vm_operations['vm_op_other'] += value
                            elif key in ['state_thread_uninitialized', 'state_thread_new',
                                        'state_thread_new_trans', 'state_thread_in_native_trans',
                                        'state_thread_in_vm', 'state_thread_in_vm_trans',
                                        'state_thread_in_java_trans', 'state_thread_blocked',
                                        'state_thread_blocked_trans']:
                                loss_kinds['other'] += value
                        except ValueError:
                            continue

                if total_lost_samples > 0:
                    self.vprint(f"    📊 Loss breakdown: STW_GC={loss_kinds['stw_gc']}, Invalid_State={loss_kinds['invalid_state']}, "
                              f"Lock={loss_kinds['could_not_acquire_lock']}, Enqueue={loss_kinds['enqueue_failed']}, Other={loss_kinds['other']}")
                    self.vprint(f"    📊 VM operations: no_vm_ops={vm_operations['no_vm_ops']}, in_jfr_safepoint={vm_operations['in_jfr_safepoint']}, "
                              f"handshake={vm_operations['vm_op_handshakeallthreads']}, other_vm_ops={vm_operations['vm_op_other']}")

                    # Calculate percentages for loss kinds
                    loss_kind_percentages = {}
                    for kind, count in loss_kinds.items():
                        if total_lost_samples > 0:
                            loss_kind_percentages[kind] = (count / total_lost_samples) * 100
                        else:
                            loss_kind_percentages[kind] = 0.0

                    # Calculate percentages for VM operations
                    vm_op_percentages = {}
                    for kind, count in vm_operations.items():
                        if total_lost_samples > 0:
                            vm_op_percentages[kind] = (count / total_lost_samples) * 100
                        else:
                            vm_op_percentages[kind] = 0.0

                    result['loss_kinds'] = loss_kinds
                    result['loss_kind_percentages'] = loss_kind_percentages
                    result['vm_operations'] = vm_operations
                    result['vm_op_percentages'] = vm_op_percentages
                    result['total_lost_samples'] = total_lost_samples

                    # Calculate the main loss categories as requested (stw_gc + invalid_state + could_not_acquire_lock + enqueue_failed)
                    main_loss_categories = loss_kinds['stw_gc'] + loss_kinds['invalid_state'] + loss_kinds['could_not_acquire_lock'] + loss_kinds['enqueue_failed']
                    if total_lost_samples > 0:
                        main_loss_percentage = (main_loss_categories / total_lost_samples) * 100
                        result['main_loss_percentage'] = main_loss_percentage
                        self.vprint(f"    📊 Main loss categories: {main_loss_percentage:.2f}% ({main_loss_categories}/{total_lost_samples})")
            else:
                print(f"    ⚠️ No LOST_SAMPLE_STATS found in log file")

            # First try direct search for all statistics in entire content
            self.vprint(f"    🔍 Searching entire log for JFR statistics...")
            successful_match = re.search(r'Successful Samples:\s*([\d,]+)', content)
            total_match = re.search(r'Total Samples:\s*([\d,]+)', content)
            lost_match = re.search(r'Lost Samples:\s*([\d,]+)', content)

            self.vprint(f"    Direct search results:")
            self.vprint(f"       Successful: {'✅' if successful_match else '❌'} {successful_match.group(1) if successful_match else 'Not found'}")
            self.vprint(f"       Total: {'✅' if total_match else '❌'} {total_match.group(1) if total_match else 'Not found'}")
            self.vprint(f"       Lost: {'✅' if lost_match else '❌'} {lost_match.group(1) if lost_match else 'Not found'}")

            if successful_match and total_match and lost_match:
                # Remove commas and convert to int
                successful = int(successful_match.group(1).replace(',', ''))
                total = int(total_match.group(1).replace(',', ''))
                lost = int(lost_match.group(1).replace(',', ''))

                # Calculate loss percentage: lost / (total + lost) * 100
                if (total + lost) > 0:
                    loss_percentage = (lost / (total + lost)) * 100
                    self.vprint(f"    📊 Parsed: Successful={successful:,}, Total={total:,}, Lost={lost:,}")
                    self.vprint(f"    📊 Total attempted: {total + lost:,}")
                    self.vprint(f"    📊 Calculated loss rate: {loss_percentage:.2f}%")

                    # Set loss percentage but continue parsing for queue statistics
                    # Only set loss_percentage if we didn't get it from run.sh LOSS_PERCENTAGE
                    if 'loss_percentage' not in result:
                        result['loss_percentage'] = loss_percentage
                        self.vprint(f"    📊 Using calculated loss percentage: {loss_percentage:.2f}%")
                    else:
                        self.vprint(f"    📊 Keeping LOSS_PERCENTAGE from run.sh: {result['loss_percentage']:.2f}%")
                    # Continue to parse queue statistics instead of returning early
                else:
                    self.vprint(f"    ⚠️ Total attempted samples is 0")
                    if 'loss_percentage' not in result:
                        result['loss_percentage'] = 0.0
                    # Continue to parse queue statistics instead of returning early

            # If direct search didn't work, try the original section-based approach

            # Look for the loss rate in the final summary
            loss_pattern = r'Loss Rate:\s+([\d.]+)%'
            match = re.search(loss_pattern, content)

            if match:
                loss_rate = float(match.group(1))
                self.vprint(f"    Found loss rate pattern: {loss_rate}%")
                if 'loss_percentage' not in result:
                    result['loss_percentage'] = loss_rate
                    self.vprint(f"    📊 Using pattern-matched loss percentage: {loss_rate:.2f}%")
                else:
                    self.vprint(f"    📊 Keeping LOSS_PERCENTAGE from run.sh: {result['loss_percentage']:.2f}%")
                # Continue to parse queue statistics instead of returning early

            # Look for JFR CPU Time Sample Statistics section
            # Look for the section and capture everything until we hit a line that starts with a special character or emoji
            stats_section = r'CPU Time Sample Statistics\s*-+\s*(.*?)(?=\n\s*[⚠️🔍📊]|\n\s*={10,}|\n\s*ANALYZING|\n\s*$)'
            section_match = re.search(stats_section, content, re.DOTALL)

            if section_match:
                stats_text = section_match.group(1)
                print(f"    Found JFR statistics section ({len(stats_text)} chars)")
                print(f"    📝 Full stats text:")
                print(f"       {repr(stats_text)}")

                # Extract numbers from the statistics (handle commas in numbers)
                successful_match = re.search(r'Successful Samples:\s*([\d,]+)', stats_text)
                total_match = re.search(r'Total Samples:\s*([\d,]+)', stats_text)
                lost_match = re.search(r'Lost Samples:\s*([\d,]+)', stats_text)

                if successful_match and total_match and lost_match:
                    # Remove commas and convert to int
                    successful = int(successful_match.group(1).replace(',', ''))
                    total = int(total_match.group(1).replace(',', ''))
                    lost = int(lost_match.group(1).replace(',', ''))

                    # Calculate loss percentage: lost / (total + lost) * 100
                    if (total + lost) > 0:
                        loss_percentage = (lost / (total + lost)) * 100
                        print(f"    📊 Parsed: Successful={successful:,}, Total={total:,}, Lost={lost:,}")
                        print(f"    📊 Total attempted: {total + lost:,}")
                        print(f"    📊 Calculated loss rate: {loss_percentage:.2f}%")
                        if 'loss_percentage' not in result:
                            result['loss_percentage'] = loss_percentage
                            print(f"    📊 Using section-parsed loss percentage: {loss_percentage:.2f}%")
                        else:
                            print(f"    📊 Keeping LOSS_PERCENTAGE from run.sh: {result['loss_percentage']:.2f}%")
                        # Continue to parse queue statistics instead of returning early
                    else:
                        print(f"    ⚠️ Total attempted samples is 0")
                        if 'loss_percentage' not in result:
                            result['loss_percentage'] = 0.0
                        # Continue to parse queue statistics instead of returning early
                else:
                    print(f"    ⚠️ Could not parse all required statistics from JFR section")
                    if successful_match:
                        print(f"       ✅ Found Successful: {successful_match.group(1)}")
                    else:
                        print(f"       ❌ Missing Successful Samples")
                    if total_match:
                        print(f"       ✅ Found Total: {total_match.group(1)}")
                    else:
                        print(f"       ❌ Missing Total Samples")
                    if lost_match:
                        print(f"       ✅ Found Lost: {lost_match.group(1)}")
                    else:
                        print(f"       ❌ Missing Lost Samples")

                    # Show the extracted stats text for debugging
                    print(f"    📝 Stats text (first 500 chars):")
                    print(f"       {repr(stats_text[:500])}")
            else:
                print(f"    ⚠️ Could not find 'CPU Time Sample Statistics' section in log")

            # Parse queue size statistics from JFR output (get the LAST occurrence)
            print(f"    🔍 Searching for queue size statistics...")
            max_queue_size_pattern = r'MAX_QUEUE_SIZE_SUM:\s*(\d+)'
            queue_increase_pattern = r'QUEUE_SIZE_INCREASE_COUNT:\s*(\d+)'

            # Find ALL matches and take the last one
            max_queue_matches = re.findall(max_queue_size_pattern, content)
            queue_increase_matches = re.findall(queue_increase_pattern, content)

            if max_queue_matches:
                result['final_max_queue_size'] = int(max_queue_matches[-1])  # Take the LAST match
                print(f"    ✅ Found MAX_QUEUE_SIZE_SUM: {result['final_max_queue_size']} (from {len(max_queue_matches)} total entries)")
            else:
                print(f"    ⚠️ No MAX_QUEUE_SIZE_SUM found in log")
                result['final_max_queue_size'] = None

            if queue_increase_matches:
                result['final_queue_size_increase_count'] = int(queue_increase_matches[-1])  # Take the LAST match
                print(f"    ✅ Found QUEUE_SIZE_INCREASE_COUNT: {result['final_queue_size_increase_count']} (from {len(queue_increase_matches)} total entries)")
            else:
                print(f"    ⚠️ No QUEUE_SIZE_INCREASE_COUNT found in log")
                result['final_queue_size_increase_count'] = None            # Show last few lines of log for debugging
            lines = content.split('\n')
            print(f"    Last 5 lines of log:")
            for line in lines[-5:]:
                if line.strip():
                    print(f"      {line}")

            return result

        except Exception as e:
            print(f"    Error extracting data from {log_path}: {e}")
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

    def run_native_benchmark(self):
        """Run comprehensive native benchmark"""
        print("� Starting Native Benchmark Suite")

        # Calculate and display runtime estimate
        total_seconds, time_str = self.estimate_runtime('native')
        total_tests = len(self.queue_sizes) * len(self.sampling_intervals) * len(self.native_durations) * len(self.stack_depths)

        print(f"📊 Benchmark Overview:")
        print(f"  Total tests: {total_tests}")
        print(f"  Estimated runtime: {time_str}")
        print(f"  Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n⚙️ Configuration:")
        print(f"  Queue sizes: {self.queue_sizes}")
        print(f"  Intervals: {self.sampling_intervals}")
        print(f"  Durations: {self.native_durations}")
        print(f"  Stack depths: {self.stack_depths}")
        print(f"  Threads: {self.threads}")

        current_test = 0
        start_time = time.time()

        for queue_size in self.queue_sizes:
            for interval in self.sampling_intervals:
                for duration in self.native_durations:
                    for stack_depth in self.stack_depths:
                        current_test += 1

                        # Calculate progress and ETA
                        elapsed = time.time() - start_time
                        if current_test > 1:
                            avg_time_per_test = elapsed / (current_test - 1)
                            remaining_tests = total_tests - current_test + 1
                            eta_seconds = remaining_tests * avg_time_per_test
                            eta_time = datetime.fromtimestamp(time.time() + eta_seconds).strftime('%H:%M:%S')
                            progress = (current_test - 1) / total_tests * 100
                            print(f"\n[{current_test}/{total_tests}] Progress: {progress:.1f}% | ETA: {eta_time}")

                        print(f"🧪 Test {current_test}: Queue={queue_size}, Interval={interval}, Duration={duration}s, Stack={stack_depth}")

                        # Run test
                        test_start = time.time()
                        result = self.run_native_test(queue_size, interval, stack_depth, duration)
                        test_time = time.time() - test_start

                        if result and result.get('success', False) and result.get('loss_percentage') is not None:
                            print(f"    ✅ Success: {result['loss_percentage']:.2f}% loss rate (took {test_time:.1f}s)")
                            result['test_number'] = current_test
                            result['test_duration_actual'] = test_time
                            self.update_csv_realtime('native', result)

                            # Generate real-time plots after every successful test
                            self.plot_realtime_progress('native')

                            # Generate Native-specific loss plots after every successful test
                            native_df = self.load_results('native')
                            if native_df is not None and len(native_df) >= 2:
                                print(f"    📊 Generating Native loss plots...")
                                self.plot_native_results(native_df, progress_mode=True)
                                self.plot_loss_kinds(native_df, progress_mode=True)
                                self.plot_queue_size_percentiles(native_df, 'native', progress_mode=True)
                                # New grid plots for progress monitoring
                                self.plot_signal_handler_duration_grid(native_df, progress_mode=True)
                                self.plot_drainage_duration_grid(native_df, progress_mode=True)
                                self.plot_vm_ops_loss_grid(native_df, progress_mode=True)

                                # Generate queue memory consumption plots using main results (contains proper loss_percentage)
                                try:
                                    if native_df is not None and len(native_df) >= 2:
                                        print(f"    💾 Generating queue memory consumption plots...")
                                        self.plot_queue_memory_consumption(native_df, progress_mode=True)
                                except Exception as e:
                                    print(f"    ⚠️ Queue memory plot generation failed: {e}")
                        else:
                            print(f"    ❌ Failed or no data (took {test_time:.1f}s)")
                            if result:
                                result['test_number'] = current_test
                                result['test_duration_actual'] = test_time
                                self.update_csv_realtime('native', result)

                        # Small delay between tests
                        time.sleep(2)

        total_time = time.time() - start_time
        print(f"\n✅ Native benchmark complete!")
        print(f"   Total runtime: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        print(f"   Results saved to {DATA_DIR.absolute()}")

        # Generate final comprehensive real-time plots
        print(f"   📊 Generating final real-time plots...")
        self.plot_realtime_progress('native')

        # Load and generate specific Native plots
        native_df = self.load_results('native')
        if native_df is not None:
            print(f"   📊 Generating Native-specific plots...")
            self.plot_native_results(native_df)
            self.plot_loss_kinds(native_df)
            self.plot_queue_size_percentiles(native_df, 'native')
            # New grid plots for signal handler and drainage duration
            self.plot_signal_handler_duration_grid(native_df)
            self.plot_drainage_duration_grid(native_df)
            self.plot_vm_ops_loss_grid(native_df)

        self.save_results('native')

    def run_renaissance_benchmark(self):
        """Run comprehensive Renaissance benchmark"""
        print("🏛️ Starting Renaissance Benchmark Suite")

        # Calculate and display runtime estimate
        total_seconds, time_str = self.estimate_runtime('renaissance')
        total_tests = len(self.queue_sizes) * len(self.sampling_intervals)  # No stack depths for Renaissance

        print(f"📊 Benchmark Overview:")
        print(f"  Total tests: {total_tests}")
        print(f"  Estimated runtime: {time_str}")
        print(f"  Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n⚙️ Configuration:")
        print(f"  Queue sizes: {self.queue_sizes}")
        print(f"  Intervals: {self.sampling_intervals}")
        print(f"  Iterations: {self.renaissance_iterations}")
        print(f"  Threads: {self.threads}")

        current_test = 0
        start_time = time.time()

        for queue_size in self.queue_sizes:
            for interval in self.sampling_intervals:
                current_test += 1

                # Calculate progress and ETA
                elapsed = time.time() - start_time
                if current_test > 1:
                    avg_time_per_test = elapsed / (current_test - 1)
                    remaining_tests = total_tests - current_test + 1
                    eta_seconds = remaining_tests * avg_time_per_test
                    eta_time = datetime.fromtimestamp(time.time() + eta_seconds).strftime('%H:%M:%S')
                    progress = (current_test - 1) / total_tests * 100
                    print(f"\n[{current_test}/{total_tests}] Progress: {progress:.1f}% | ETA: {eta_time}")

                print(f"🏛️ Test {current_test}: Queue={queue_size}, Interval={interval}")

                # Run test (no stack_depth for Renaissance)
                test_start = time.time()
                result = self.run_renaissance_test(queue_size, interval)
                test_time = time.time() - test_start

                if result and result.get('success', False) and result.get('loss_percentage') is not None:
                    print(f"    ✅ Success: {result['loss_percentage']:.2f}% loss rate (took {test_time:.1f}s)")
                    result['test_number'] = current_test
                    result['test_duration_actual'] = test_time
                    self.update_csv_realtime('renaissance', result)

                    # Generate real-time plots after every successful test
                    self.plot_realtime_progress('renaissance')

                    # Generate Renaissance-specific loss plots after every successful test
                    renaissance_df = self.load_results('renaissance')
                    if renaissance_df is not None and len(renaissance_df) >= 2:
                        print(f"    📊 Generating Renaissance loss plots...")
                        self.plot_renaissance_results(renaissance_df, progress_mode=True)
                        self.plot_loss_kinds(renaissance_df, progress_mode=True)
                        self.plot_vm_operations(renaissance_df, progress_mode=True)
                        self.plot_queue_size_percentiles(renaissance_df, 'renaissance', progress_mode=True)
                        # New grid plots for progress monitoring
                        self.plot_signal_handler_duration_grid(renaissance_df, progress_mode=True)
                        self.plot_drainage_duration_grid(renaissance_df, progress_mode=True)
                        self.plot_vm_ops_loss_grid(renaissance_df, progress_mode=True)

                        # Generate queue memory consumption plots using main results (contains proper loss_percentage)
                        try:
                            if renaissance_df is not None and len(renaissance_df) >= 2:
                                print(f"    💾 Generating queue memory consumption plots...")
                                self.plot_queue_memory_consumption(renaissance_df, progress_mode=True)
                        except Exception as e:
                            print(f"    ⚠️ Queue memory plot generation failed: {e}")
                else:
                    print(f"    ❌ Failed or no data (took {test_time:.1f}s)")
                    if result:
                        result['test_number'] = current_test
                        result['test_duration_actual'] = test_time
                        self.update_csv_realtime('renaissance', result)

                # Small delay between tests
                time.sleep(2)

        total_time = time.time() - start_time
        print(f"\n✅ Renaissance benchmark complete!")
        print(f"   Total runtime: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        print(f"   Results saved to {DATA_DIR.absolute()}")

        # Generate final comprehensive real-time plots
        print(f"   📊 Generating final real-time plots...")
        self.plot_realtime_progress('renaissance')

        # Load and generate specific Renaissance plots
        renaissance_df = self.load_results('renaissance')
        if renaissance_df is not None:
            print(f"   📊 Generating Renaissance-specific plots...")
            self.plot_renaissance_results(renaissance_df)
            self.plot_loss_kinds(renaissance_df)
            self.plot_vm_operations(renaissance_df)
            self.plot_queue_size_percentiles(renaissance_df, 'renaissance')
            # New grid plots for signal handler and drainage duration
            self.plot_signal_handler_duration_grid(renaissance_df)
            self.plot_drainage_duration_grid(renaissance_df)
            self.plot_vm_ops_loss_grid(renaissance_df)

        self.save_results('renaissance')

    def save_results(self, test_type: str):
        """Save results to JSON and CSV with descriptive filenames"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create descriptive filename based on configuration
        if self.minimal:
            config_desc = f"minimal_q{min(self.queue_sizes)}-{max(self.queue_sizes)}_i{len(self.sampling_intervals)}_d{self.test_duration}s"
            if test_type == 'native':
                config_desc += f"_native{min(self.native_durations)}-{max(self.native_durations)}s"
        else:
            config_desc = f"full_q{min(self.queue_sizes)}-{max(self.queue_sizes)}_i{len(self.sampling_intervals)}_d{self.test_duration}s"
            if test_type == 'native':
                config_desc += f"_native{min(self.native_durations)}-{max(self.native_durations)}s"

        # JSON format (complete data) with descriptive filename
        json_file = DATA_DIR / f"{test_type}_{config_desc}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results[test_type], f, indent=2)

        # CSV format (for easy analysis) - use flattened data
        csv_file = DATA_DIR / f"{test_type}_{config_desc}_{timestamp}.csv"
        if self.results[test_type]:
            df = self.flatten_for_csv(self.results[test_type])
            df.to_csv(csv_file, index=False)

        # Also save as "latest" for easy access
        latest_json = DATA_DIR / f"{test_type}_results_latest.json"
        latest_csv = DATA_DIR / f"{test_type}_results_latest.csv"

        with open(latest_json, 'w') as f:
            json.dump(self.results[test_type], f, indent=2)

        if self.results[test_type]:
            df = self.flatten_for_csv(self.results[test_type])
            df.to_csv(latest_csv, index=False)

        print(f"   📁 Results saved: {json_file.name}")
        print(f"   📊 CSV saved: {csv_file.name}")

        # Display summary of results
        successful_tests = [r for r in self.results[test_type] if r.get('success', False)]
        failed_tests = [r for r in self.results[test_type] if not r.get('success', False)]

        print(f"   ✅ Successful tests: {len(successful_tests)}")
        print(f"   ❌ Failed tests: {len(failed_tests)}")

        if successful_tests:
            valid_losses = [r['loss_percentage'] for r in successful_tests if r.get('loss_percentage') is not None]
            if valid_losses:
                avg_loss = sum(valid_losses) / len(valid_losses)
                min_loss = min(valid_losses)
                max_loss = max(valid_losses)
                print(f"   📊 Loss rate stats: avg={avg_loss:.2f}%, min={min_loss:.2f}%, max={max_loss:.2f}%")

    def load_results(self, test_type: str) -> Optional[pd.DataFrame]:
        """Load latest results for visualization"""
        csv_file = DATA_DIR / f"{test_type}_results_latest.csv"

        if csv_file.exists():
            return pd.read_csv(csv_file)
        else:
            print(f"No results found for {test_type}. Run benchmark first.")
            return None

    def load_drain_statistics(self, session_only: bool = False) -> Optional[pd.DataFrame]:
        """Load drain statistics using direct log parsing

        Args:
            session_only: If True, only load logs created after session start time
        """
        if not DIRECT_PARSING_AVAILABLE:
            print("⚠️ Direct log parsing not available")
            return None

        try:
            print("📊 Loading drain statistics from log files...")
            logs_dir = Path("benchmark_results/logs")
            if not logs_dir.exists():
                print(f"❌ Logs directory not found: {logs_dir}")
                return None

            # Filter log files by session time if requested
            if session_only and hasattr(self, 'session_start_time'):
                # Use a 60-second buffer before session start to account for timing differences
                session_start_with_buffer = self.session_start_time - 60
                print(f"   🔍 Filtering logs to current session only (started {datetime.fromtimestamp(self.session_start_time).strftime('%H:%M:%S')})")
                log_files = []
                for log_file in logs_dir.glob("*.txt"):
                    file_mtime = log_file.stat().st_mtime
                    if file_mtime >= session_start_with_buffer:
                        log_files.append(log_file)

                if not log_files:
                    print("   ⚠️ No log files from current session found, falling back to all logs")
                    # Fallback to all logs if no session-specific logs found
                    results = parse_all_logs(logs_dir)
                else:
                    print(f"   ✅ Found {len(log_files)} log files from current session")

                    # Parse only current session logs
                    results = []
                    for log_file in log_files:
                        try:
                            file_results = parse_drain_stats_from_log(log_file)
                            if file_results:
                                results.append(file_results)
                        except Exception as e:
                            print(f"   ⚠️ Error parsing {log_file}: {e}")
            else:
                # Parse all logs (original behavior)
                results = parse_all_logs(logs_dir)

            if not results:
                print("❌ No drain statistics found in log files")
                return None

            # Create DataFrame
            drain_df = create_percentile_dataframe(results)
            session_desc = " from current session" if session_only else ""
            print(f"✅ Loaded drain statistics{session_desc}: {len(drain_df)} records across {len(drain_df['drain_category'].unique())} categories")
            return drain_df

        except Exception as e:
            print(f"❌ Error loading drain statistics: {e}")
            return None

    def plot_realtime_progress(self, test_type: str = 'native'):
        """Generate real-time plots after each test iteration"""
        try:
            # Only generate plots if we have sufficient data
            if len(self.results[test_type]) < 2:
                return

            print(f"    📊 Generating real-time plots ({len(self.results[test_type])} results so far)...")

            # Convert current results to DataFrame
            df = self.flatten_for_csv(self.results[test_type])

            # Only include successful tests with loss percentage data
            df = df[df['success'] == True]
            df = df.dropna(subset=['loss_percentage'])

            if len(df) < 2:
                print(f"    ⚠️ Not enough successful results yet ({len(df)} successful)")
                return

            # Set plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Calculate global y-range for all data to ensure consistency across plots
            all_loss_percentages = df['loss_percentage'].dropna()
            if len(all_loss_percentages) == 0:
                print(f"    ⚠️ No valid loss percentage data")
                return

            # For normal scale: minimum should be 0, maximum with some padding but capped at 100%
            y_min_normal = 0  # Always start from 0 for normal scale
            y_max_normal = min(100.0, all_loss_percentages.max() * 1.1)  # Cap at 100% loss

            # Create combined grid plot for all intervals
            # Order intervals by numeric value (1ms, 2ms, 5ms, 10ms, 20ms)
            def interval_sort_key(interval):
                """Sort intervals by numeric value"""
                return int(interval.replace('ms', ''))

            intervals = sorted(df['interval'].unique(), key=interval_sort_key)
            n_intervals = len(intervals)

            if n_intervals == 0:
                print(f"    ⚠️ No intervals found in data")
                return

            # Calculate grid dimensions
            if n_intervals <= 2:
                cols = n_intervals
                rows = 1
            elif n_intervals <= 4:
                cols = 2
                rows = 2
            elif n_intervals <= 6:
                cols = 3
                rows = 2
            else:
                cols = 3
                rows = (n_intervals + 2) // 3

            # Create figure with subplots
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

            # Handle single subplot case
            if n_intervals == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()

            for i, interval in enumerate(intervals):
                if i >= len(axes):
                    break

                ax = axes[i]
                interval_data = df[df['interval'] == interval]

                if len(interval_data) < 1:
                    ax.set_title(f'{interval} (No Data)')
                    ax.axis('off')
                    continue

                if test_type == 'native':
                    # Plot points for each combination of stack depth and native duration
                    for stack_depth in sorted(interval_data['stack_depth'].unique()):
                        for native_duration in sorted(interval_data['native_duration'].unique()):
                            subset = interval_data[
                                (interval_data['stack_depth'] == stack_depth) &
                                (interval_data['native_duration'] == native_duration)
                            ]

                            if len(subset) < 1:
                                continue

                            # Sort by queue size for proper plotting
                            subset = subset.sort_values('queue_size')

                            # Create label for this combination
                            label = f"S{stack_depth}, D{native_duration}s"

                            # Use scatter plot for realtime progress
                            ax.scatter(subset['queue_size'], subset['loss_percentage'],
                                      s=30, alpha=0.7, label=label)

                elif test_type == 'renaissance':
                    # Renaissance tests don't have stack_depth, just plot all points
                    subset = interval_data.sort_values('queue_size')

                    if len(subset) < 1:
                        continue

                    # Create single line for all Renaissance data
                    label = "Renaissance"

                    # Use scatter plot for realtime progress
                    ax.scatter(subset['queue_size'], subset['loss_percentage'],
                              s=30, alpha=0.7, label=label)

                # Customize the subplot
                ax.set_xlabel('Queue Size', fontsize=10)
                ax.set_ylabel('Loss %', fontsize=10)
                ax.set_title(f'{interval} ({len(interval_data)} tests)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Set shared y-axis range for consistency across all subplots
                ax.set_ylim(y_min_normal, y_max_normal)

                # Add legend if there are multiple lines
                lines, labels = ax.get_legend_handles_labels()
                if lines:
                    ax.legend(fontsize=8, loc='best')

                # Set x-axis to log scale if we have a wide range of queue sizes
                if len(interval_data) > 1 and interval_data['queue_size'].max() / interval_data['queue_size'].min() > 10:
                    ax.set_xscale('log')

                    # Add more minor ticks for log scale with finer subdivisions
                    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
                    # Show selective minor tick labels to prevent scientific notation
                    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
                    ax.grid(True, which='minor', alpha=0.25)
                    ax.grid(True, which='major', alpha=0.5)
                else:
                    # Add more minor ticks for normal scale
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

                # Add more minor ticks for y-axis (always normal scale for these plots)
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                ax.grid(True, which='minor', alpha=0.15)

                # Prevent scientific notation on axes
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

            # Hide unused subplots
            for i in range(n_intervals, len(axes)):
                axes[i].set_visible(False)

            # Add overall title (minimal)
            fig.suptitle(f'{test_type.title()} Progress',
                        fontsize=16, fontweight='bold', y=0.99)

            # Save the combined plot
            realtime_plots_dir = PLOTS_DIR / "realtime"
            realtime_plots_dir.mkdir(exist_ok=True)

            plot_filename = f"{test_type}_realtime_combined_progress.png"
            plot_path = realtime_plots_dir / plot_filename

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, right=0.95)  # Move title higher and add right margin for x-labels
            plt.savefig(plot_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            print(f"    📊 Saved combined real-time plot: {plot_path.absolute()}")

            # Also create log-scaled version with unified y-range and 1% reference line
            self._create_log_scaled_combined_plot(df, test_type, realtime_plots_dir)

            # Generate high DPI individual plots for each interval
            self._create_individual_interval_plots(df, test_type, realtime_plots_dir, intervals)

        except Exception as e:
            print(f"    ⚠️ Error generating real-time plots: {e}")

    def _create_log_scaled_combined_plot(self, df, test_type: str, realtime_plots_dir):
        """Create a log-scaled combined plot with unified y-range and 1% reference line"""
        try:
            # Order intervals by numeric value (1ms, 2ms, 5ms, 10ms, 20ms)
            def interval_sort_key(interval):
                """Sort intervals by numeric value"""
                return int(interval.replace('ms', ''))

            intervals = sorted(df['interval'].unique(), key=interval_sort_key)
            n_intervals = len(intervals)

            if n_intervals == 0:
                return

            # Calculate global y-range for all data
            all_loss_percentages = df['loss_percentage'].dropna()
            if len(all_loss_percentages) == 0:
                return

            # Set y-range with some padding, ensuring we capture the 1% line
            y_min = max(0.001, all_loss_percentages.min() * 0.5)  # Don't go below 0.001%
            y_max = min(100.0, max(1.0, all_loss_percentages.max() * 2.0))   # Cap at 100% loss, ensure 1% line is visible

            # Calculate grid dimensions
            if n_intervals <= 2:
                cols = n_intervals
                rows = 1
            elif n_intervals <= 4:
                cols = 2
                rows = 2
            elif n_intervals <= 6:
                cols = 3
                rows = 2
            else:
                cols = 3
                rows = (n_intervals + 2) // 3

            # Create figure with subplots
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

            # Handle single subplot case
            if n_intervals == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()

            for i, interval in enumerate(intervals):
                if i >= len(axes):
                    break

                ax = axes[i]
                interval_data = df[df['interval'] == interval]

                if len(interval_data) < 1:
                    ax.set_title(f'{interval} (No Data)')
                    ax.axis('off')
                    continue

                if test_type == 'native':
                    # Plot points for each combination of stack depth and native duration
                    for stack_depth in sorted(interval_data['stack_depth'].unique()):
                        for native_duration in sorted(interval_data['native_duration'].unique()):
                            subset = interval_data[
                                (interval_data['stack_depth'] == stack_depth) &
                                (interval_data['native_duration'] == native_duration)
                            ]

                            if len(subset) < 1:
                                continue

                            # Sort by queue size for proper plotting
                            subset = subset.sort_values('queue_size')

                            # Create label for this combination
                            label = f"S{stack_depth}, D{native_duration}s"

                            # Use scatter plot for log-scale realtime progress
                            ax.scatter(subset['queue_size'], subset['loss_percentage'],
                                      s=30, alpha=0.7, label=label)

                elif test_type == 'renaissance':
                    # Renaissance tests don't have stack_depth, just plot all points
                    subset = interval_data.sort_values('queue_size')

                    if len(subset) < 1:
                        continue

                    # Create single line for all Renaissance data
                    label = "Renaissance"

                    # Use scatter plot for log-scale realtime progress
                    ax.scatter(subset['queue_size'], subset['loss_percentage'],
                              s=30, alpha=0.7, label=label)

                # Add the psychologically important 1% reference line
                ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1, alpha=0.7, label='1% threshold')

                # Customize the subplot
                ax.set_xlabel('Queue Size (log scale)', fontsize=10)
                ax.set_ylabel('Loss % (log scale)', fontsize=10)
                ax.set_title(f'{interval} ({len(interval_data)} tests)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Set unified y-range and log scale
                ax.set_ylim(ymax=y_max)
                ax.set_yscale('log')

                # Always set x-axis to log scale for the log-scaled plot
                ax.set_xscale('log')

                # Prevent scientific notation on axes even with log scale
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.4f}' if y < 0.01 else f'{y:,.2f}'))

                # Set major tick locators first to ensure they're always visible
                ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))

                # Add minor ticks with conservative subdivision to avoid overcrowding
                ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.2)))  # Less crowded subdivisions
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.2)))  # Less crowded subdivisions

                # Selective minor tick labeling - only show key values to avoid overcrowding
                ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000] else ''))
                # Reduce y-axis minor labels to prevent disappearing labels
                key_y_values = [0.002, 0.005, 0.02, 0.05, 0.2, 0.5, 2.0, 5.0, 20.0, 50.0]
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:.3f}' if y in key_y_values else ''))

                ax.grid(True, which='minor', alpha=0.3)   # Make minor grid more visible
                ax.grid(True, which='major', alpha=0.6)   # Ensure major grid is visible

                # Add legend if there are multiple lines
                lines, labels = ax.get_legend_handles_labels()
                if lines:
                    ax.legend(fontsize=8, loc='best')

            # Hide unused subplots
            for i in range(n_intervals, len(axes)):
                axes[i].set_visible(False)

            # Add overall title (minimal)
            fig.suptitle(f'{test_type.title()} (Log Scale)',
                        fontsize=16, fontweight='bold', y=0.99)

            # Save the log-scaled combined plot (without test count in the filename)
            log_plot_filename = f"{test_type}_realtime_combined_logscale.png"
            log_plot_path = realtime_plots_dir / log_plot_filename

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, right=0.95)  # Move title higher and add right margin for x-labels
            plt.savefig(log_plot_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            print(f"    📊 Saved log-scale combined plot: {log_plot_path.absolute()}")

        except Exception as e:
            print(f"    ⚠️ Error generating log-scale plot: {e}")

    def _create_individual_interval_plots(self, df, test_type: str, realtime_plots_dir, intervals):
        """Create high DPI individual plots for each interval"""
        try:
            individual_plots_dir = realtime_plots_dir / "individual"
            individual_plots_dir.mkdir(exist_ok=True)

            for interval in intervals:
                interval_data = df[df['interval'] == interval]

                if len(interval_data) < 1:
                    continue

                # Create individual plot for this interval
                fig, ax = plt.subplots(figsize=(10, 8))

                if test_type == 'native':
                    # Plot points for each combination of stack depth and native duration
                    for stack_depth in sorted(interval_data['stack_depth'].unique()):
                        for native_duration in sorted(interval_data['native_duration'].unique()):
                            subset = interval_data[
                                (interval_data['stack_depth'] == stack_depth) &
                                (interval_data['native_duration'] == native_duration)
                            ]

                            if len(subset) < 1:
                                continue

                            # Sort by queue size for proper plotting
                            subset = subset.sort_values('queue_size')

                            # Create label for this combination
                            label = f"Stack {stack_depth}, Duration {native_duration}s"

                            # Use scatter plot for individual interval plots
                            ax.scatter(subset['queue_size'], subset['loss_percentage'],
                                      s=50, alpha=0.7, label=label)

                elif test_type == 'renaissance':
                    # Renaissance tests don't have stack_depth, just plot all points
                    subset = interval_data.sort_values('queue_size')

                    if len(subset) < 1:
                        continue

                    # Create single line for all Renaissance data
                    label = "Renaissance"

                    # Use scatter plot for individual interval plots
                    ax.scatter(subset['queue_size'], subset['loss_percentage'],
                              s=50, alpha=0.7, label=label)

                # Customize the plot
                ax.set_xlabel('Queue Size', fontsize=14)
                ax.set_ylabel('Loss Percentage (%)', fontsize=14)
                ax.set_title(f'{test_type.title()} Test - {interval} Interval ({len(interval_data)} tests)',
                            fontsize=16, fontweight='bold', pad=15)  # Move title higher
                ax.grid(True, alpha=0.3)

                # Add legend if there are multiple lines
                lines, labels = ax.get_legend_handles_labels()
                if lines:
                    ax.legend(fontsize=12, loc='best')

                # Set x-axis to log scale if we have a wide range of queue sizes
                if len(interval_data) > 1 and interval_data['queue_size'].max() / interval_data['queue_size'].min() > 10:
                    ax.set_xscale('log')

                    # Add more minor ticks for log scale with finer subdivisions
                    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
                    # Show selective minor tick labels to prevent scientific notation
                    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
                    ax.grid(True, which='minor', alpha=0.25)
                    ax.grid(True, which='major', alpha=0.5)
                else:
                    # Add more minor ticks for normal scale
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

                # Add more minor ticks for y-axis (always normal scale for individual plots)
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                ax.grid(True, which='minor', alpha=0.15)

                # Prevent scientific notation on axes
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

                # Save the individual plot with high DPI (without test count in the filename)
                plot_filename = f"{test_type}_{interval}_individual.png"
                plot_path = individual_plots_dir / plot_filename

                plt.tight_layout()
                plt.savefig(plot_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                self.vprint(f"    📊 Saved individual plot: {plot_path.absolute()}")

            print(f"    📊 Saved {len(intervals)} individual interval plots to: {individual_plots_dir.absolute()}")

            # Create additional Renaissance plots with all intervals combined
            if test_type == 'renaissance':
                self._create_renaissance_combined_interval_plots(df, realtime_plots_dir)

        except Exception as e:
            print(f"    ⚠️ Error generating individual interval plots: {e}")

    def _create_renaissance_combined_interval_plots(self, df, output_dir):
        """Create two Renaissance plots with all intervals combined into one plot"""
        try:
            # Plot 1: Normal scale
            self._create_renaissance_all_intervals_plot(df, output_dir, log_scale=False)

            # Plot 2: Log scale
            self._create_renaissance_all_intervals_plot(df, output_dir, log_scale=True)

        except Exception as e:
            print(f"    ⚠️ Error generating Renaissance combined interval plots: {e}")

    def _create_renaissance_all_intervals_plot(self, df, output_dir, log_scale=False):
        """Create a Renaissance plot with all intervals in one plot"""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 9))

            # Order intervals by numeric value
            def interval_sort_key(interval):
                return int(interval.replace('ms', ''))

            intervals = sorted(df['interval'].unique(), key=interval_sort_key)

            # Plot each interval as a separate line
            for i, interval in enumerate(intervals):
                interval_data = df[df['interval'] == interval]

                if len(interval_data) < 1:
                    continue

                # Sort by queue size for proper plotting
                interval_data = interval_data.sort_values('queue_size')

                # Plot with default round dots
                ax.plot(interval_data['queue_size'], interval_data['loss_percentage'],
                       'o', markersize=10, label=f'{interval}')

            # Set axes
            ax.set_xlabel('Queue Size', fontsize=14)
            ax.set_ylabel('Loss Percentage (%)', fontsize=14)

            # Apply log scales if requested
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
                title = 'Renaissance (Log Scale)'

                # Add 1% reference line for log plot
                ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='1% threshold')

                # Set y-range with some padding
                all_loss_percentages = df['loss_percentage'].dropna()
                if len(all_loss_percentages) > 0:
                    y_min = max(0.001, all_loss_percentages.min() * 0.5)
                    y_max = min(100.0, max(1.0, all_loss_percentages.max() * 2.0))
                    ax.set_ylim(ymax=y_max)

                # Prevent scientific notation on axes even with log scale
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.4f}' if y < 0.01 else f'{y:,.2f}'))

                # Set major tick locators first to ensure they're always visible
                ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))

                # Add minor ticks with conservative subdivision to avoid overcrowding
                ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.2)))  # Less crowded subdivisions
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.2)))  # Less crowded subdivisions

                # Selective minor tick labeling - only show key values to avoid overcrowding
                ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000] else ''))
                # Reduce y-axis minor labels to prevent disappearing labels
                key_y_values = [0.002, 0.005, 0.02, 0.05, 0.2, 0.5, 2.0, 5.0, 20.0, 50.0]
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:.3f}' if y in key_y_values else ''))

                # Add more minor ticks for log scale with finer subdivisions
                ax.grid(True, which='minor', alpha=0.3)   # Make minor grid more visible
                ax.grid(True, which='major', alpha=0.6)   # Ensure major grid is visible

                # Force tick label visibility - this prevents matplotlib from hiding labels
                ax.tick_params(axis='both', which='major', labelsize=8, labelbottom=True, labelleft=True)
                ax.tick_params(axis='both', which='minor', labelsize=6)

                # Force tick label visibility - this prevents matplotlib from hiding labels
                ax.tick_params(axis='both', which='major', labelsize=8, labelbottom=True, labelleft=True)
                ax.tick_params(axis='both', which='minor', labelsize=6)
            else:
                title = 'Renaissance'

                # For normal scale: set shared y-axis range
                all_loss_percentages = df['loss_percentage'].dropna()
                if len(all_loss_percentages) > 0:
                    y_min = 0  # Always start from 0 for normal scale
                    y_max = min(100.0, all_loss_percentages.max() * 1.1)  # Cap at 100% loss with 10% padding
                    ax.set_ylim(y_min, y_max)

                # Prevent scientific notation on axes for normal scale
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

                # Add more minor ticks for normal scale
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # 5 minor ticks between major ticks
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                ax.grid(True, which='minor', alpha=0.2)

            # Set title with more padding to move it higher (minimal title)
            ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, loc='best')

            # Save the plot with high DPI
            filename_suffix = 'logscale' if log_scale else 'normal'
            plot_filename = f"renaissance_all_intervals_{filename_suffix}.png"
            plot_path = output_dir / plot_filename

            plt.tight_layout()
            plt.savefig(plot_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            print(f"    📊 Saved Renaissance combined intervals plot ({filename_suffix}): {plot_path.absolute()}")

        except Exception as e:
            print(f"    ⚠️ Error generating Renaissance all intervals plot: {e}")

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("📊 Generating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Load data
        native_df = self.load_results('native')
        renaissance_df = self.load_results('renaissance')

        if native_df is not None:
            self.plot_native_results(native_df)
            self.plot_loss_kinds(native_df)
            self.plot_queue_size_percentiles(native_df, 'native')
            self.plot_queue_size_distribution_percentiles(native_df, 'native')
            # New grid plots for comprehensive visualization
            self.plot_signal_handler_duration_grid(native_df)
            self.plot_drainage_duration_grid(native_df)
            self.plot_vm_ops_loss_grid(native_df)
            # Queue memory consumption plots
            self.plot_queue_memory_consumption(native_df)

        if renaissance_df is not None:
            self.plot_renaissance_results(renaissance_df)
            self.plot_loss_kinds(renaissance_df)
            self.plot_vm_operations(renaissance_df)
            self.plot_queue_size_percentiles(renaissance_df, 'renaissance')
            self.plot_queue_size_distribution_percentiles(renaissance_df, 'renaissance')
            # New grid plots for comprehensive visualization
            self.plot_signal_handler_duration_grid(renaissance_df)
            self.plot_drainage_duration_grid(renaissance_df)
            self.plot_vm_ops_loss_grid(renaissance_df)
            # Queue memory consumption plots
            self.plot_queue_memory_consumption(renaissance_df)

            # Load drain statistics and create Renaissance-specific plots
            drain_df = self.load_drain_statistics()
            if drain_df is not None:
                self.plot_renaissance_out_of_thread_percentage(drain_df)

        if native_df is not None and renaissance_df is not None:
            self.plot_comparison(native_df, renaissance_df)

        print(f"📊 Visualizations saved to {PLOTS_DIR.absolute()}")

    def create_progress_visualizations(self, csv_file: Optional[str] = None):
        """Create visualizations from live CSV data with 'progress' suffix"""
        print("📊 Generating progress visualizations from live CSV data...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Determine CSV file to use
        if csv_file:
            csv_path = Path(csv_file)
            if not csv_path.exists():
                print(f"❌ Specified CSV file not found: {csv_file}")
                return
        else:
            # Look for the most recent CSV file
            csv_files = list(DATA_DIR.glob("*_results_latest.csv"))
            if not csv_files:
                print("❌ No CSV files found. Run benchmark first or specify --csv-file")
                return
            csv_path = max(csv_files, key=lambda x: x.stat().st_mtime)

        print(f"📂 Loading data from: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return

        if df.empty:
            print("⚠️ CSV file is empty")
            return

        # Check if we need drain statistics data and if CSV doesn't have it, try loading from JSON
        needs_drain_stats = False
        if 'drain_categories' not in df.columns:
            # Try to load from corresponding JSON file for drain statistics
            json_path = csv_path.with_suffix('.json')

            if json_path.exists():
                print(f"📂 CSV missing drain statistics, loading from JSON: {json_path}")
                try:
                    import json
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)

                    # Add drain_categories back to DataFrame if available
                    for idx, row in df.iterrows():
                        # Find matching entry in JSON data
                        for json_entry in json_data:
                            if (json_entry.get('queue_size') == row['queue_size'] and
                                json_entry.get('interval') == row['interval'] and
                                json_entry.get('stack_depth') == row.get('stack_depth')):
                                # Store the drain_categories as an object in the DataFrame
                                df.loc[idx, 'drain_categories'] = json_entry.get('drain_categories', {})
                                break

                    print(f"📊 Enhanced CSV data with drain statistics from JSON")
                except Exception as e:
                    print(f"⚠️ Could not load drain statistics from JSON: {e}")
            else:
                print(f"⚠️ No corresponding JSON file found: {json_path}")

        if df.empty:
            print("⚠️ CSV file is empty")
            return

        # Determine test type from filename or data
        if 'native' in csv_path.name:
            test_type = 'native'
        elif 'renaissance' in csv_path.name:
            test_type = 'renaissance'
        else:
            # Try to determine from data - be more specific about column detection
            has_native_cols = 'native_duration' in df.columns
            has_renaissance_cols = 'interval' in df.columns and 'stack_depth' in df.columns and 'native_duration' not in df.columns

            if has_native_cols:
                test_type = 'native'
            elif has_renaissance_cols:
                test_type = 'renaissance'
            else:
                # Default fallback - check if there are more Renaissance-like patterns
                test_type = 'renaissance'  # More common pattern

        print(f"🔍 Detected test type: {test_type}")
        print(f"📊 Data contains {len(df)} records")

        # Create progress visualizations (same as regular but with different filenames)
        if test_type == 'native':
            self.plot_native_results(df, progress_mode=True)
            self.plot_queue_size_percentiles(df, 'native', progress_mode=True)
            # New grid plots for progress monitoring
            self.plot_signal_handler_duration_grid(df, progress_mode=True)
            self.plot_drainage_duration_grid(df, progress_mode=True)
            self.plot_vm_ops_loss_grid(df, progress_mode=True)
            # Queue memory consumption plots
            self.plot_queue_memory_consumption(df, progress_mode=True)
        else:
            self.plot_renaissance_results(df, progress_mode=True)
            self.plot_vm_operations(df, progress_mode=True)
            self.plot_queue_size_percentiles(df, 'renaissance', progress_mode=True)
            # New grid plots for progress monitoring
            self.plot_signal_handler_duration_grid(df, progress_mode=True)
            self.plot_drainage_duration_grid(df, progress_mode=True)
            self.plot_vm_ops_loss_grid(df, progress_mode=True)
            # Queue memory consumption plots
            self.plot_queue_memory_consumption(df, progress_mode=True)

        print(f"📊 Progress visualizations saved to {PLOTS_DIR.absolute()} with '_progress' suffix")

    def watch_progress_visualizations(self, csv_file: Optional[str] = None):
        """Watch mode: continuously monitor and update progress visualizations"""
        print("👁️ Starting watch mode for progress visualizations...")
        print("Press Ctrl+C to stop watching")

        import time

        last_modified = 0
        csv_path = None

        try:
            while True:
                # Determine CSV file to monitor
                if csv_file:
                    csv_path = Path(csv_file)
                    if not csv_path.exists():
                        print(f"❌ Specified CSV file not found: {csv_file}")
                        print("Waiting for file to be created...")
                        time.sleep(5)
                        continue
                else:
                    # Look for the most recent CSV file
                    csv_files = list(DATA_DIR.glob("*_results_latest.csv"))
                    if not csv_files:
                        print("⏳ No CSV files found. Waiting for benchmark data...")
                        time.sleep(10)
                        continue
                    csv_path = max(csv_files, key=lambda x: x.stat().st_mtime)

                # Check if file has been modified
                current_modified = csv_path.stat().st_mtime
                if current_modified > last_modified:
                    last_modified = current_modified
                    print(f"\n🔄 CSV updated: {csv_path.name} - Regenerating plots...")

                    try:
                        self.create_progress_visualizations(str(csv_path))
                        print(f"✅ Plots updated at {datetime.now().strftime('%H:%M:%S')}")
                    except Exception as e:
                        print(f"⚠️ Error updating plots: {e}")

                # Wait before checking again
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n👋 Watch mode stopped by user")
        except Exception as e:
            print(f"\n❌ Watch mode error: {e}")

    def plot_native_results(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create native test visualizations"""
        # Filter successful tests if 'success' column exists
        if 'success' in df.columns:
            df_success = df[df['success'] == True].copy()
        else:
            df_success = df.copy()  # Use all data if no success column

        if df_success.empty:
            print("No successful native tests to visualize")
            return

        # Check if we have the required columns for native plotting
        required_cols = ['native_duration', 'stack_depth', 'interval']
        missing_cols = [col for col in required_cols if col not in df_success.columns]

        if missing_cols:
            print(f"⚠️ Missing required columns for native plotting: {missing_cols}")
            print("   This might be Renaissance data. Use --visualize instead.")
            return

        # Convert interval to numeric for sorting
        interval_order = ["1ms", "5ms", "10ms", "20ms"]
        df_success['interval_num'] = df_success['interval'].map({
            "1ms": 1, "5ms": 5, "10ms": 10, "20ms": 20
        })

        # Get unique combinations of native duration and stack depth
        duration_stack_combinations = df_success[['native_duration', 'stack_depth']].drop_duplicates().sort_values(['native_duration', 'stack_depth'])

        # 1. Create separate heatmaps for each native duration/stack depth combination
        n_combinations = len(duration_stack_combinations)
        if n_combinations <= 6:
            cols = 3
            rows = (n_combinations + cols - 1) // cols
        else:
            cols = 4
            rows = (n_combinations + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        fig.suptitle('Native Test: Loss Rate Heatmaps by Duration/Stack Combination', fontsize=16)

        for i, (_, row) in enumerate(duration_stack_combinations.iterrows()):
            if i >= len(axes):
                break

            native_dur = row['native_duration']
            stack_depth = row['stack_depth']

            df_combo = df_success[
                (df_success['native_duration'] == native_dur) &
                (df_success['stack_depth'] == stack_depth)
            ]

            if not df_combo.empty:
                pivot = df_combo.pivot(index='queue_size', columns='interval', values='loss_percentage')
                pivot = pivot.reindex(columns=interval_order)

                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                           ax=axes[i], cbar_kws={'label': 'Loss Rate (%)'})
                axes[i].set_title(f'Duration: {native_dur}s, Stack: {stack_depth}')
                axes[i].set_xlabel('Sampling Interval')
                axes[i].set_ylabel('Queue Size')

                # Prevent scientific notation on heatmap axes
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}' if y % 1 == 0 else f'{y:,.1f}'))
            else:
                axes[i].set_title(f'Duration: {native_dur}s, Stack: {stack_depth} (No Data)')
                axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_heatmaps_by_duration_stack.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2. Line plots: Loss Rate vs Queue Size for each Interval, showing all duration/stack combinations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.suptitle('Native Test: Loss Rate vs Queue Size by Interval (All Duration/Stack Combinations)', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            for _, row in duration_stack_combinations.iterrows():
                native_dur = row['native_duration']
                stack_depth = row['stack_depth']

                df_combo = df_int[
                    (df_int['native_duration'] == native_dur) &
                    (df_int['stack_depth'] == stack_depth)
                ]

                if not df_combo.empty:
                    axes[i].plot(df_combo['queue_size'], df_combo['loss_percentage'],
                               marker='o', label=f'{native_dur}s/S{stack_depth}', linestyle='None')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

            # Add minor ticks for log scale with finer subdivisions
            axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            # Show selective minor tick labels to prevent scientific notation
            axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
            axes[i].grid(True, which='minor', alpha=0.25)
            axes[i].grid(True, which='major', alpha=0.5)

            # Prevent scientific notation on axes
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_loss_vs_queue_size_all_combinations.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2b. Same plot but with log scale for y-axis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.suptitle('Native Test: Loss Rate vs Queue Size by Interval (All Duration/Stack Combinations, Log Y-Scale)', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            for _, row in duration_stack_combinations.iterrows():
                native_dur = row['native_duration']
                stack_depth = row['stack_depth']

                df_combo = df_int[
                    (df_int['native_duration'] == native_dur) &
                    (df_int['stack_depth'] == stack_depth)
                ]

                if not df_combo.empty:
                    axes[i].plot(df_combo['queue_size'], df_combo['loss_percentage'],
                               marker='o', label=f'{native_dur}s/S{stack_depth}', linestyle='None')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Set log scale for y-axis
            axes[i].set_yscale('log')

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

            # Add minor ticks for log scale with finer subdivisions
            axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            axes[i].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            # Show selective minor tick labels to prevent scientific notation
            axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
            axes[i].grid(True, which='minor', alpha=0.25)
            axes[i].grid(True, which='major', alpha=0.5)

            # Prevent scientific notation on axes
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_loss_vs_queue_size_all_combinations_log.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2c. Same plot but with y-axis starting from 0
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.suptitle('Native Test: Loss Rate vs Queue Size by Interval (All Duration/Stack Combinations, Y-Min=0)', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            for _, row in duration_stack_combinations.iterrows():
                native_dur = row['native_duration']
                stack_depth = row['stack_depth']

                df_combo = df_int[
                    (df_int['native_duration'] == native_dur) &
                    (df_int['stack_depth'] == stack_depth)
                ]

                if not df_combo.empty:
                    axes[i].plot(df_combo['queue_size'], df_combo['loss_percentage'],
                               marker='o', label=f'{native_dur}s/S{stack_depth}', linestyle='None')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Set y-axis to start from 0
            axes[i].set_ylim(bottom=0)

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

            # Add minor ticks for log scale with finer subdivisions
            axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            # Show selective minor tick labels to prevent scientific notation
            axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
            axes[i].grid(True, which='minor', alpha=0.25)
            axes[i].grid(True, which='major', alpha=0.5)

            # Prevent scientific notation on axes
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_loss_vs_queue_size_all_combinations_y0.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 3. Separate plots for each stack depth showing native duration effects
        for stack_depth in df_success['stack_depth'].unique():
            df_stack = df_success[df_success['stack_depth'] == stack_depth]

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            fig.suptitle(f'Native Test: Loss Rate vs Queue Size (Stack Depth: {stack_depth})', fontsize=16)

            for i, interval in enumerate(interval_order):
                df_int = df_stack[df_stack['interval'] == interval]

                for native_dur in df_stack['native_duration'].unique():
                    df_dur = df_int[df_int['native_duration'] == native_dur]
                    if not df_dur.empty:
                        axes[i].plot(df_dur['queue_size'], df_dur['loss_percentage'],
                                   marker='o', label=f'{native_dur}s native', linestyle='None')

                axes[i].set_title(f'Interval: {interval}')
                axes[i].set_xlabel('Queue Size')
                axes[i].set_ylabel('Loss Rate (%)')

                # Only add legend if there are actually lines plotted
                lines, labels = axes[i].get_legend_handles_labels()
                if lines:
                    axes[i].legend()

                axes[i].grid(True, alpha=0.3)
                axes[i].set_xscale('log')

                # Add minor ticks for log scale with finer subdivisions
                axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
                # Show selective minor tick labels to prevent scientific notation
                axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
                axes[i].grid(True, which='minor', alpha=0.25)
                axes[i].grid(True, which='major', alpha=0.5)

                # Prevent scientific notation on axes
                axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / self._get_plot_filename(f'native_loss_vs_queue_size_stack{stack_depth}.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        # 4. 3D surface plot for most interesting interval (1ms) with one stack depth
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use 1ms interval and first stack depth as it's likely most interesting
        first_stack = df_success['stack_depth'].iloc[0]
        df_1ms = df_success[
            (df_success['interval'] == '1ms') &
            (df_success['stack_depth'] == first_stack)
        ]

        if not df_1ms.empty:
            X = df_1ms['queue_size'].values
            Y = df_1ms['native_duration'].values
            Z = df_1ms['loss_percentage'].values

            scatter = ax.scatter(X, Y, Z, c=Z, cmap='RdYlBu_r', s=100)
            ax.set_xlabel('Queue Size')
            ax.set_ylabel('Native Duration (s)')
            ax.set_zlabel('Loss Rate (%)')
            ax.set_title(f'Native Test: 3D Loss Rate Surface (1ms interval, Stack: {first_stack})')
            plt.colorbar(scatter)

            # Prevent scientific notation on 3D plot axes
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}' if y % 1 == 0 else f'{y:,.1f}'))
            ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, _: f'{z:,.2f}'))

        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_3d_surface.png', progress_mode), dpi=600, bbox_inches='tight')
        plt.close()

        # 5. Individual plots for each combination of interval, stack depth, and native duration
        self._create_individual_native_plots(df_success, progress_mode)

    def _create_individual_native_plots(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create individual plots for each native test combination"""
        print("📊 Creating individual native plots...")

        for interval in df['interval'].unique():
            for stack_depth in df['stack_depth'].unique():
                for native_duration in df['native_duration'].unique():
                    subset = df[(df['interval'] == interval) &
                               (df['stack_depth'] == stack_depth) &
                               (df['native_duration'] == native_duration)]
                    if subset.empty:
                        continue

                    # Create both normal and log scale versions
                    self._create_individual_plot(subset,
                                               f'native_{interval}_stack{stack_depth}_dur{native_duration}s',
                                               f'Native - {interval}, Stack {stack_depth}, {native_duration}s',
                                               progress_mode)

    def plot_renaissance_results(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create Renaissance test visualizations"""
        # Filter successful tests if 'success' column exists
        if 'success' in df.columns:
            df_success = df[df['success'] == True].copy()
        else:
            df_success = df.copy()  # Use all data if no success column

        if df_success.empty:
            print("No successful Renaissance tests to visualize")
            return

        # Convert interval to numeric for sorting
        interval_order = ["1ms", "5ms", "10ms", "20ms"]
        df_success['interval_num'] = df_success['interval'].map({
            "1ms": 1, "5ms": 5, "10ms": 10, "20ms": 20
        })

        # 1. Create single heatmap (Renaissance doesn't have stack depth)
        fig, ax = plt.subplots(figsize=(10, 8))

        pivot = df_success.pivot(index='queue_size', columns='interval', values='loss_percentage')
        pivot = pivot.reindex(columns=interval_order)

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   ax=ax, cbar_kws={'label': 'Loss Rate (%)'})
        ax.set_title(f'Renaissance Test: Loss Rate Heatmap', fontsize=14)
        ax.set_xlabel('Sampling Interval')
        ax.set_ylabel('Queue Size')

        # Prevent scientific notation on heatmap axes
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}' if y % 1 == 0 else f'{y:,.1f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename(f'renaissance_heatmap.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2. Line plots: Loss Rate vs Queue Size for each Interval
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('Renaissance Test: Loss Rate vs Queue Size by Interval', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            if not df_int.empty:
                axes[i].plot(df_int['queue_size'], df_int['loss_percentage'],
                           marker='o', linestyle='-', markersize=8, label='Renaissance')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend()

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

            # Add minor ticks for log scale with finer subdivisions
            axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            # Show selective minor tick labels to prevent scientific notation
            axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
            axes[i].grid(True, which='minor', alpha=0.25)
            axes[i].grid(True, which='major', alpha=0.5)

            # Prevent scientific notation on axes
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('renaissance_loss_vs_queue_size.png', progress_mode), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2b. Same plot but with log scale for y-axis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('Renaissance Test: Loss Rate vs Queue Size by Interval (Log Y-Scale)', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            if not df_int.empty:
                axes[i].plot(df_int['queue_size'], df_int['loss_percentage'],
                           marker='o', linestyle='-', markersize=8, label='Renaissance')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Set log scale for y-axis
            axes[i].set_yscale('log')

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend()

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

            # Add minor ticks for log scale with finer subdivisions
            axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            axes[i].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            # Show selective minor tick labels to prevent scientific notation
            axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
            axes[i].grid(True, which='minor', alpha=0.25)
            axes[i].grid(True, which='major', alpha=0.5)

            # Prevent scientific notation on axes
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('renaissance_loss_vs_queue_size_log.png', progress_mode), dpi=600, bbox_inches='tight')
        plt.close()

        # 2c. Same plot but with y-axis starting from 0
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('Renaissance Test: Loss Rate vs Queue Size by Interval (Y-Min=0)', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            if not df_int.empty:
                axes[i].plot(df_int['queue_size'], df_int['loss_percentage'],
                           marker='o', linestyle='-', markersize=8, label='Renaissance')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Set y-axis to start from 0
            axes[i].set_ylim(bottom=0)

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend()

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

            # Add minor ticks for log scale with finer subdivisions
            axes[i].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
            # Show selective minor tick labels to prevent scientific notation
            axes[i].xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000, 5000] else ''))
            axes[i].grid(True, which='minor', alpha=0.25)
            axes[i].grid(True, which='major', alpha=0.5)

            # Prevent scientific notation on axes
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename('renaissance_loss_vs_queue_size_y0.png', progress_mode), dpi=600, bbox_inches='tight')
        plt.close()

        # 3. Individual plots for each interval
        self._create_individual_renaissance_plots(df_success, progress_mode)

    def _create_individual_renaissance_plots(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create individual plots for each interval"""
        print("📊 Creating individual Renaissance plots...")

        for interval in df['interval'].unique():
            subset = df[df['interval'] == interval]
            if subset.empty:
                continue

            # Create both normal and log scale versions
            self._create_individual_plot(subset, f'renaissance_{interval}',
                                       f'Renaissance - {interval}', progress_mode)

    def _create_individual_plot(self, df: pd.DataFrame, base_filename: str, title: str, progress_mode: bool = False):
        """Create both normal and log scale individual plots"""

        # 1. Normal scale with zero min y and 1% dotted line
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(df['queue_size'], df['loss_percentage'], s=80, marker='o', label='Loss Rate', alpha=0.8)

        # Set y-axis to start from 0 and cap at 100%
        y_max = min(100.0, max(5.0, df['loss_percentage'].max() * 1.1))  # At least 5% visible, cap at 100%
        ax.set_ylim(0, y_max)

        # Add 1% reference line if it's within the visible range
        if y_max >= 1.0:
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1, label='1% Reference')

        ax.set_title(f'{title} (Normal Scale)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Queue Size')
        ax.set_ylabel('Loss Rate (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Prevent scientific notation
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename(f'{base_filename}_normal.png', progress_mode), dpi=600, bbox_inches='tight')
        plt.close()

        # 2. Log scale version
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(df['queue_size'], df['loss_percentage'], s=80, marker='o', label='Loss Rate', alpha=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Set y-axis limits for log scale
        y_min = max(0.001, df['loss_percentage'].min() * 0.5)
        y_max = min(100.0, max(1.0, df['loss_percentage'].max() * 2.0))
        ax.set_ylim(y_min, y_max)

        # Add 1% reference line
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1, label='1% Reference')

        ax.set_title(f'{title} (Log Scale)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Queue Size (log scale)')
        ax.set_ylabel('Loss Rate (%) (log scale)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set up log scale formatting
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.4f}' if y < 0.01 else f'{y:,.2f}'))

        # Set major tick locators to ensure labels are visible
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))

        # Add conservative minor ticks
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.2)))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.2, 1.0, 0.2)))

        # Minor tick formatting
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}' if x in [20, 50, 200, 500, 2000] else ''))
        key_y_values = [0.002, 0.005, 0.02, 0.05, 0.2, 0.5, 2.0, 5.0, 20.0, 50.0]
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:.3f}' if y in key_y_values else ''))

        # Force tick label visibility
        ax.tick_params(axis='both', which='major', labelsize=8, labelbottom=True, labelleft=True)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / self._get_plot_filename(f'{base_filename}_log.png', progress_mode), dpi=600, bbox_inches='tight')
        plt.close()

    def plot_comparison(self, native_df: pd.DataFrame, renaissance_df: pd.DataFrame):
        """Create comparison visualizations between native and Renaissance"""
        # Filter successful tests
        native_success = native_df[native_df['success'] == True].copy()
        renaissance_success = renaissance_df[renaissance_df['success'] == True].copy()

        if native_success.empty or renaissance_success.empty:
            print("Insufficient data for comparison plots")
            return

        # For native, average across native durations for comparison
        native_avg = native_success.groupby(['queue_size', 'interval'])['loss_percentage'].mean().reset_index()
        native_avg['test_type'] = 'Native'

        renaissance_success['test_type'] = 'Renaissance'

        # Combine data
        comparison_data = []

        for _, row in native_avg.iterrows():
            comparison_data.append({
                'queue_size': row['queue_size'],
                'interval': row['interval'],
                'loss_percentage': row['loss_percentage'],
                'test_type': 'Native (avg)'
            })

        for _, row in renaissance_success.iterrows():
            comparison_data.append({
                'queue_size': row['queue_size'],
                'interval': row['interval'],
                'loss_percentage': row['loss_percentage'],
                'test_type': 'Renaissance'
            })

        comp_df = pd.DataFrame(comparison_data)

        # Comparison plot: Side-by-side bar charts
        interval_order = ["1ms", "5ms", "10ms", "20ms"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.suptitle('Comparison: Native vs Renaissance Loss Rates', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = comp_df[comp_df['interval'] == interval]

            if not df_int.empty:
                sns.barplot(data=df_int, x='queue_size', y='loss_percentage',
                           hue='test_type', ax=axes[i])
                axes[i].set_title(f'Interval: {interval}')
                axes[i].set_xlabel('Queue Size')
                axes[i].set_ylabel('Loss Rate (%)')
                axes[i].tick_params(axis='x', rotation=45)

                # Prevent scientific notation on axes
                axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.2f}'))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'comparison_native_vs_renaissance.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_loss_kinds(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create scatter plots for loss kinds breakdown by queue size and interval"""
        print("📊 Creating loss kinds plots...")

        # Filter out rows where we don't have loss kind data
        df_with_loss_kinds = df.dropna(subset=['loss_percentage'])

        if df_with_loss_kinds.empty:
            print("⚠️ No data with loss kinds information found")
            return

        # Use only the latest entry for each unique configuration to avoid duplicates
        print(f"    📊 Deduplicating data: {len(df_with_loss_kinds)} rows before deduplication")

        # Group by configuration columns and take the last entry (most recent) for each group
        config_columns = ['queue_size', 'interval']
        if 'stack_depth' in df_with_loss_kinds.columns:
            config_columns.append('stack_depth')
        if 'native_duration' in df_with_loss_kinds.columns:
            config_columns.append('native_duration')

        df_deduplicated = df_with_loss_kinds.groupby(config_columns).last().reset_index()
        print(f"    📊 After deduplication: {len(df_deduplicated)} rows")

        # Create the output directory for loss kind plots
        loss_plots_dir = PLOTS_DIR / "loss_kinds"
        loss_plots_dir.mkdir(exist_ok=True)

        # Get unique intervals
        intervals = sorted(df_deduplicated['interval'].unique())

        print(f"📊 Found {len(intervals)} intervals: {intervals}")

        # Define loss categories
        loss_categories = ['stw_gc', 'invalid_state', 'could_not_acquire_lock', 'enqueue_failed']

        # Calculate loss percentages for each category
        loss_data = []

        for _, row in df_deduplicated.iterrows():
            queue_size = row['queue_size']
            interval = row['interval']

            # Get loss kinds data from the results (could be raw dict or flattened CSV format)
            loss_kinds = {}
            total_lost_samples = 0

            # Try to get from raw dict format first
            if 'loss_kinds' in row and isinstance(row['loss_kinds'], dict):
                loss_kinds = row['loss_kinds']
                total_lost_samples = row.get('total_lost_samples', 0)
            else:
                # Try to get from flattened CSV format
                for category in loss_categories:
                    csv_col = f'loss_kind_{category}'
                    if csv_col in row:
                        loss_kinds[category] = row[csv_col] if pd.notna(row[csv_col]) else 0
                        total_lost_samples += loss_kinds[category]

            if not loss_kinds or total_lost_samples == 0:
                # If no loss kinds data, skip this row
                continue

            # Calculate absolute loss percentages for each category
            # Loss percentage = (category_count / total_lost_samples) * overall_loss_percentage
            overall_loss_pct = row['loss_percentage']

            for category in loss_categories:
                category_count = loss_kinds.get(category, 0)
                if total_lost_samples > 0:
                    # This gives the percentage of total samples lost to this specific category
                    category_loss_pct = (category_count / total_lost_samples) * overall_loss_pct
                else:
                    category_loss_pct = 0.0

                loss_data.append({
                    'queue_size': queue_size,
                    'interval': interval,
                    'loss_category': category,
                    'loss_percentage': category_loss_pct
                })


        loss_df = pd.DataFrame(loss_data)

        if loss_df.empty:
            print("⚠️ No loss kinds data found - plots will be skipped")
            return        # Create individual plots for each interval
        for interval in intervals:
            interval_data = loss_df[loss_df['interval'] == interval]

            if interval_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create scatter plot for each loss category
            for category in loss_categories:
                cat_data = interval_data[interval_data['loss_category'] == category]
                if not cat_data.empty:
                    ax.scatter(cat_data['queue_size'], cat_data['loss_percentage'],
                             label=category, alpha=0.7, s=60)

            ax.set_title(f'Loss Kinds by Queue Size - {interval}ms Interval', fontsize=14, fontweight='bold')
            ax.set_xlabel('Queue Size', fontsize=12)
            ax.set_ylabel('Loss Percentage (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Loss Category')

            # Prevent scientific notation
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

            plt.tight_layout()
            plt.savefig(loss_plots_dir / f'loss_kinds_{interval}ms.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        # Create combined plot with all intervals in a grid
        if len(intervals) > 1:
            # Calculate grid dimensions
            n_intervals = len(intervals)
            cols = min(3, n_intervals)  # Max 3 columns
            rows = (n_intervals + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Find global y-axis limits for consistent scaling
            all_y_values = loss_df['loss_percentage'].values
            y_min = 0
            y_max = max(all_y_values) * 1.1 if len(all_y_values) > 0 else 1.0

            for idx, interval in enumerate(intervals):
                if idx >= len(axes):
                    break

                ax = axes[idx]
                interval_data = loss_df[loss_df['interval'] == interval]

                # Create scatter plot for each loss category
                for category in loss_categories:
                    cat_data = interval_data[interval_data['loss_category'] == category]
                    if not cat_data.empty:
                        ax.scatter(cat_data['queue_size'], cat_data['loss_percentage'],
                                 label=category, alpha=0.7, s=40)

                ax.set_title(f'{interval}ms Interval', fontsize=12, fontweight='bold')
                ax.set_xlabel('Queue Size', fontsize=10)
                ax.set_ylabel('Loss %', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(y_min, y_max)

                # Prevent scientific notation
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

                # Add legend only to first subplot
                if idx == 0:
                    ax.legend(title='Loss Category', fontsize=8, title_fontsize=9)

            # Hide unused subplots
            for idx in range(len(intervals), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle('Loss Kinds by Queue Size - All Intervals', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(loss_plots_dir / 'loss_kinds_all_intervals.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        print(f"📊 Loss kinds plots saved to {loss_plots_dir.absolute()}")

    def plot_vm_operations(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create plots for VM operations breakdown by queue size and interval"""
        print("📊 Creating VM operations plots...")

        # Filter out rows where we don't have loss kind data
        df_with_loss_kinds = df.dropna(subset=['loss_percentage'])

        if df_with_loss_kinds.empty:
            print("⚠️ No data with loss kinds information found")
            return

        # Use only the latest entry for each unique configuration to avoid duplicates
        print(f"    📊 Deduplicating VM operations data: {len(df_with_loss_kinds)} rows before deduplication")

        # Group by configuration columns and take the last entry (most recent) for each group
        config_columns = ['queue_size', 'interval']
        if 'stack_depth' in df_with_loss_kinds.columns:
            config_columns.append('stack_depth')
        if 'native_duration' in df_with_loss_kinds.columns:
            config_columns.append('native_duration')

        df_deduplicated = df_with_loss_kinds.groupby(config_columns).last().reset_index()
        print(f"    📊 After deduplication: {len(df_deduplicated)} rows")

        # Create the output directory for VM operations plots
        vm_ops_plots_dir = PLOTS_DIR / "vm_operations"
        vm_ops_plots_dir.mkdir(exist_ok=True)

        # Get unique intervals
        intervals = sorted(df_deduplicated['interval'].unique())

        print(f"📊 Found {len(intervals)} intervals: {intervals}")

        # Define VM operation categories to track (these should match what we parse in LOST_SAMPLE_STATS)
        vm_op_categories = [
            'no_vm_ops',                    # When no VM operations are causing loss
            'in_jfr_safepoint',            # JFR safepoint operations
            'vm_op_handshakeallthreads',   # Most common VM operation
            'vm_op_g1collectforfullgc',    # GC-related VM operations
            'vm_op_g1incremental',
            'vm_op_cleanup',
            'vm_op_other'                  # Sum of all other VM operations
        ]

        # Calculate VM operation percentages for each category
        vm_ops_data = []

        for _, row in df_deduplicated.iterrows():
            queue_size = row['queue_size']
            interval = row['interval']

            # Get VM operations data from the results (could be raw dict or flattened CSV format)
            vm_operations = {}
            total_lost_samples = 0

            # Try to get from raw dict format first
            if 'vm_operations' in row and isinstance(row['vm_operations'], dict):
                vm_operations = row['vm_operations']
                total_lost_samples = row.get('total_lost_samples', 0)
            else:
                # Try to get from flattened CSV format
                for category in vm_op_categories:
                    csv_col = f'vm_op_{category}'
                    if csv_col in row:
                        vm_operations[category] = row[csv_col] if pd.notna(row[csv_col]) else 0
                        total_lost_samples += vm_operations[category]

                # If we didn't find VM operations in CSV format, try to get total from loss data
                if total_lost_samples == 0:
                    total_lost_samples = row.get('total_lost_samples', 0)

            if not vm_operations or total_lost_samples == 0:
                # If no VM operations data, skip this row
                continue

            # Calculate absolute loss percentages for each VM operation category
            # Loss percentage = (category_count / total_lost_samples) * overall_loss_percentage
            overall_loss_pct = row['loss_percentage']

            for category in vm_op_categories:
                category_count = vm_operations.get(category, 0)
                if total_lost_samples > 0:
                    # This gives the percentage of total samples lost to this specific VM operation
                    category_loss_pct = (category_count / total_lost_samples) * overall_loss_pct
                else:
                    category_loss_pct = 0.0

                vm_ops_data.append({
                    'queue_size': queue_size,
                    'interval': interval,
                    'vm_op_category': category,
                    'loss_percentage': category_loss_pct
                })

        vm_ops_df = pd.DataFrame(vm_ops_data)

        if vm_ops_df.empty:
            print("⚠️ No VM operations data found - plots will be skipped")
            return

        # Create individual plots for each interval
        for interval in intervals:
            interval_data = vm_ops_df[vm_ops_df['interval'] == interval]

            if interval_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create scatter plot for each VM operation category
            for category in vm_op_categories:
                cat_data = interval_data[interval_data['vm_op_category'] == category]
                if not cat_data.empty and cat_data['loss_percentage'].sum() > 0:  # Only plot if there's actual data
                    # Create user-friendly labels
                    label = category.replace('vm_op_', 'VM: ').replace('_', ' ').title()
                    if category == 'no_vm_ops':
                        label = 'No VM Ops'
                    elif category == 'in_jfr_safepoint':
                        label = 'JFR Safepoint'

                    # Use round dots for all markers as requested
                    ax.scatter(cat_data['queue_size'], cat_data['loss_percentage'],
                             label=label, alpha=0.7, s=60, marker='o')

            ax.set_title(f'VM Operations by Queue Size - {interval} Interval', fontsize=14, fontweight='bold')
            ax.set_xlabel('Queue Size', fontsize=12)
            ax.set_ylabel('Loss Percentage (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(title='VM Operation Category')

            # Prevent scientific notation
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

            plt.tight_layout()
            plt.savefig(vm_ops_plots_dir / f'vm_operations_{interval}.png', dpi=600, bbox_inches='tight')
            plt.close()

        # Create combined plot with all intervals in a grid
        if len(intervals) > 1:
            # Calculate grid dimensions
            if len(intervals) <= 2:
                cols = len(intervals)
                rows = 1
            elif len(intervals) <= 4:
                cols = 2
                rows = 2
            else:
                cols = 3
                rows = (len(intervals) + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

            # Handle single subplot case
            if len(intervals) == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()

            for i, interval in enumerate(intervals):
                if i >= len(axes):
                    break

                ax = axes[i]
                interval_data = vm_ops_df[vm_ops_df['interval'] == interval]

                # Create scatter plot for each VM operation category
                for category in vm_op_categories:
                    cat_data = interval_data[interval_data['vm_op_category'] == category]
                    if not cat_data.empty and cat_data['loss_percentage'].sum() > 0:  # Only plot if there's actual data
                        # Create user-friendly labels
                        label = category.replace('vm_op_', 'VM: ').replace('_', ' ').title()
                        if category == 'no_vm_ops':
                            label = 'No VM Ops'
                        elif category == 'in_jfr_safepoint':
                            label = 'JFR Safepoint'

                        # Use round dots for all markers
                        ax.scatter(cat_data['queue_size'], cat_data['loss_percentage'],
                                 label=label, alpha=0.7, s=60, marker='o')

                ax.set_title(f'{interval} Interval', fontsize=12, fontweight='bold')
                ax.set_xlabel('Queue Size', fontsize=10)
                ax.set_ylabel('Loss Percentage (%)', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add legend only to first subplot
                if i == 0:
                    ax.legend(title='VM Operation', fontsize=8, title_fontsize=9)

                # Prevent scientific notation
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

            # Hide unused subplots
            for i in range(len(intervals), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle('VM Operations by Queue Size and Interval', fontsize=16, fontweight='bold')
            plt.tight_layout()

        if progress_mode:
            plt.savefig(vm_ops_plots_dir / f'vm_operations_combined_progress.png', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(vm_ops_plots_dir / f'vm_operations_combined.png', dpi=600, bbox_inches='tight')
        plt.close()

        print(f"📊 VM operations plots saved to {vm_ops_plots_dir.absolute()}")

    def plot_renaissance_out_of_thread_percentage(self, drain_df: pd.DataFrame, progress_mode: bool = False):
        """Create plots showing percentage of 'out of thread' drainages for Renaissance tests"""
        print("📊 Creating Renaissance 'out of thread' percentage plots...")

        if drain_df is None or drain_df.empty:
            print("⚠️ No drain statistics data available")
            return

        # Filter for Renaissance tests only
        renaissance_data = drain_df[drain_df['test_type'] == 'renaissance'].copy()
        if renaissance_data.empty:
            print("⚠️ No Renaissance drain statistics found")
            return

        # Create output directory
        out_of_thread_plots_dir = PLOTS_DIR / "renaissance_out_of_thread"
        out_of_thread_plots_dir.mkdir(exist_ok=True)

        # Calculate total drains per test configuration
        # Group by test configuration and sum all drains
        total_drains = renaissance_data.groupby(['queue_size', 'interval'])['drains'].sum().reset_index()
        total_drains.rename(columns={'drains': 'total_drains'}, inplace=True)

        # Get 'out of thread' drains only
        out_of_thread_data = renaissance_data[renaissance_data['drain_category'] == 'out of thread'].copy()

        # Merge to get percentages
        merged_data = out_of_thread_data.merge(total_drains, on=['queue_size', 'interval'], how='left')
        merged_data['out_of_thread_percentage'] = (merged_data['drains'] / merged_data['total_drains']) * 100

        # Get unique intervals for subplots
        intervals = sorted(merged_data['interval'].unique())

        # Create figure with subplots for each interval
        n_intervals = len(intervals)
        if n_intervals == 0:
            print("⚠️ No interval data found")
            return

        cols = min(3, n_intervals)
        rows = (n_intervals + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_intervals == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Color palette for different queue sizes
        queue_sizes = sorted(merged_data['queue_size'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(queue_sizes)))

        for i, interval in enumerate(intervals):
            if i >= len(axes):
                break

            ax = axes[i]
            interval_data = merged_data[merged_data['interval'] == interval]

            # Create scatter plot
            for j, queue_size in enumerate(queue_sizes):
                queue_data = interval_data[interval_data['queue_size'] == queue_size]
                if not queue_data.empty:
                    # Ensure we have consistent data sizes
                    x_values = [queue_size] * len(queue_data)
                    y_values = queue_data['out_of_thread_percentage'].tolist()
                    ax.scatter(x_values, y_values,
                             color=colors[j], s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

            ax.set_xlabel('Queue Size', fontweight='bold')
            ax.set_ylabel('Out of Thread Drains (%)', fontweight='bold')
            ax.set_title(f'Interval: {interval}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')

            # Format x-axis to show actual queue sizes
            ax.set_xticks(queue_sizes)
            ax.set_xticklabels([f'{int(qs):,}' if qs % 1 == 0 else f'{qs:,.1f}' for qs in queue_sizes])

            # Prevent scientific notation on axes
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

        # Remove empty subplots
        for i in range(n_intervals, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle('Renaissance: Out of Thread Drainage Percentage by Queue Size and Interval',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        if progress_mode:
            plt.savefig(out_of_thread_plots_dir / 'renaissance_out_of_thread_percentage_progress.png',
                       dpi=600, bbox_inches='tight')
        else:
            plt.savefig(out_of_thread_plots_dir / 'renaissance_out_of_thread_percentage.png',
                       dpi=600, bbox_inches='tight')
        plt.close()

        # Create combined plot with all intervals
        fig, ax = plt.subplots(figsize=(12, 8))

        # Different markers for different intervals
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

        for i, interval in enumerate(intervals):
            interval_data = merged_data[merged_data['interval'] == interval]
            marker = markers[i % len(markers)]

            ax.scatter(interval_data['queue_size'], interval_data['out_of_thread_percentage'],
                      label=f'{interval}', marker=marker, s=100, alpha=0.7,
                      edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Queue Size', fontweight='bold', fontsize=12)
        ax.set_ylabel('Out of Thread Drains (%)', fontweight='bold', fontsize=12)
        ax.set_title('Renaissance: Out of Thread Drainage Percentage\nAcross All Intervals',
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.legend(title='Sampling Interval', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format x-axis
        ax.set_xticks(queue_sizes)
        ax.set_xticklabels([f'{int(qs):,}' if qs % 1 == 0 else f'{qs:,.1f}' for qs in queue_sizes])

        # Prevent scientific notation on axes
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

        plt.tight_layout()

        if progress_mode:
            plt.savefig(out_of_thread_plots_dir / 'renaissance_out_of_thread_combined_progress.png',
                       dpi=600, bbox_inches='tight')
        else:
            plt.savefig(out_of_thread_plots_dir / 'renaissance_out_of_thread_combined.png',
                       dpi=600, bbox_inches='tight')
        plt.close()

        print(f"📊 Renaissance 'out of thread' percentage plots saved to {out_of_thread_plots_dir}")

    def plot_queue_size_distribution_percentiles(self, df: pd.DataFrame, test_type: str, progress_mode: bool = False):
        """Create plots showing queue size distribution percentiles (P95, P99, P99.9 of queue sizes used)"""
        print(f"📊 Creating queue size distribution percentile plots for {test_type}...")

        # Load drain statistics using direct parsing
        drain_df = self.load_drain_statistics()
        if drain_df is None or drain_df.empty:
            print(f"⚠️ No drain statistics available for {test_type} queue size distribution plots")
            return

        # Filter for the specific test type
        test_drain_data = drain_df[drain_df['test_type'] == test_type].copy()
        if test_drain_data.empty:
            print(f"⚠️ No drain statistics found for {test_type} tests")
            return

        # Create output directory
        queue_dist_plots_dir = PLOTS_DIR / "queue_size_distributions"
        queue_dist_plots_dir.mkdir(exist_ok=True)

        # Group by drain category and interval to calculate queue size distributions
        intervals = sorted(test_drain_data['interval'].unique())
        drain_categories = sorted(test_drain_data['drain_category'].unique())

        # Calculate percentiles of queue sizes for each category and interval
        percentiles = [95, 99, 99.9]
        percentile_labels = ['P95', 'P99', 'P99.9']

        # Create plots for each drain category
        for category in drain_categories:
            category_data = test_drain_data[test_drain_data['drain_category'] == category]

            if category_data.empty:
                continue

            fig, axes = plt.subplots(1, len(intervals), figsize=(5*len(intervals), 5))
            if len(intervals) == 1:
                axes = [axes]

            for i, interval in enumerate(intervals):
                ax = axes[i]
                interval_data = category_data[category_data['interval'] == interval]

                if interval_data.empty:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                           ha='center', va='center', fontsize=14, fontweight='bold')
                    ax.set_title(f'{category} - {interval}', fontweight='bold')
                    continue

                # Calculate percentiles of queue sizes weighted by drain counts
                queue_sizes = []
                drain_counts = []
                for _, row in interval_data.iterrows():
                    queue_sizes.extend([row['queue_size']] * int(row['drains']))
                    drain_counts.append(row['drains'])

                if queue_sizes:
                    queue_percentiles = np.percentile(queue_sizes, percentiles)
                    colors = ['blue', 'red', 'green']

                    # Create bar plot for the percentiles
                    bars = ax.bar(percentile_labels, queue_percentiles, color=colors, alpha=0.7,
                                 edgecolor='black', linewidth=1)

                    # Add value labels on bars
                    for bar, value in zip(bars, queue_percentiles):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

                ax.set_xlabel('Percentile', fontweight='bold')
                ax.set_ylabel('Queue Size', fontweight='bold')
                ax.set_title(f'{category} - {interval}', fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Prevent scientific notation on y-axis
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}' if y % 1 == 0 else f'{y:,.1f}'))

            plt.suptitle(f'{test_type.title()}: Queue Size Distribution Percentiles\n({category})',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save plot
            safe_category = category.replace(' ', '_').replace('/', '_')
            if progress_mode:
                plt.savefig(queue_dist_plots_dir / f'{test_type}_{safe_category}_queue_distribution_progress.png',
                           dpi=600, bbox_inches='tight')
            else:
                plt.savefig(queue_dist_plots_dir / f'{test_type}_{safe_category}_queue_distribution.png',
                           dpi=600, bbox_inches='tight')
            plt.close()

        # Create combined plot showing all categories
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grouped bar chart
        width = 0.25
        x_positions = np.arange(len(percentile_labels))

        colors = plt.cm.Set1(np.linspace(0, 1, len(drain_categories)))

        for i, category in enumerate(drain_categories):
            category_data = test_drain_data[test_drain_data['drain_category'] == category]

            if category_data.empty:
                continue

            # Aggregate all queue sizes across intervals for this category
            all_queue_sizes = []
            for _, row in category_data.iterrows():
                all_queue_sizes.extend([row['queue_size']] * int(row['drains']))

            if all_queue_sizes:
                category_percentiles = np.percentile(all_queue_sizes, percentiles)

                # Plot bars for this category
                bars = ax.bar(x_positions + i * width, category_percentiles, width,
                             label=category, color=colors[i], alpha=0.7,
                             edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Percentile', fontweight='bold', fontsize=12)
        ax.set_ylabel('Queue Size', fontweight='bold', fontsize=12)
        ax.set_title(f'{test_type.title()}: Queue Size Distribution Percentiles\n(All Categories Combined)',
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set x-axis labels
        ax.set_xticks(x_positions + width * (len(drain_categories) - 1) / 2)
        ax.set_xticklabels(percentile_labels)

        # Prevent scientific notation on y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}' if y % 1 == 0 else f'{y:,.1f}'))

        plt.tight_layout()

        if progress_mode:
            plt.savefig(queue_dist_plots_dir / f'{test_type}_queue_distribution_combined_progress.png',
                       dpi=600, bbox_inches='tight')
        else:
            plt.savefig(queue_dist_plots_dir / f'{test_type}_queue_distribution_combined.png',
                       dpi=600, bbox_inches='tight')
        plt.close()

        print(f"📊 Queue size distribution plots saved to {queue_dist_plots_dir.absolute()}")

    def plot_queue_size_percentiles(self, df: pd.DataFrame, test_type: str, progress_mode: bool = False):
        """Create plots for queue size percentiles (p95, p99, p99.9) from direct drain statistics"""
        print(f"📊 Creating queue size percentile plots for {test_type} using direct drain statistics...")

        # Load drain statistics using direct parsing
        drain_df = self.load_drain_statistics()
        if drain_df is None or drain_df.empty:
            print(f"⚠️ No drain statistics available for {test_type} percentile plots")
            return

        # Filter for the specific test type
        test_drain_data = drain_df[drain_df['test_type'] == test_type].copy()
        if test_drain_data.empty:
            print(f"⚠️ No drain statistics found for {test_type} tests")
            return

        # Focus on 'all without locks' category for percentile data (as discussed)
        percentile_data = test_drain_data[test_drain_data['drain_category'] == 'all without locks'].copy()

        if percentile_data.empty:
            print(f"⚠️ No 'all without locks' drain data found for {test_type}")
            return

        # Create output directory
        percentile_plots_dir = PLOTS_DIR / "queue_size_percentiles"
        percentile_plots_dir.mkdir(exist_ok=True)

        # Get unique intervals
        intervals = sorted(percentile_data['interval'].unique())

        # Create separate plots for each percentile (P95, P99, P99.9)
        percentiles = ['p95', 'p99', 'p99_9']
        percentile_labels = ['P95', 'P99', 'P99.9']

        for percentile, label in zip(percentiles, percentile_labels):
            fig, axes = plt.subplots(1, len(intervals), figsize=(5*len(intervals), 5))
            if len(intervals) == 1:
                axes = [axes]

            for i, interval in enumerate(intervals):
                ax = axes[i]
                interval_data = percentile_data[percentile_data['interval'] == interval]

                # Filter out zero values (missing data)
                valid_data = interval_data[interval_data[f'time_{percentile}'] > 0]

                if not valid_data.empty:
                    ax.scatter(valid_data['queue_size'], valid_data[f'time_{percentile}'],
                             s=80, alpha=0.7, edgecolors='black', linewidth=0.5, color='darkblue')

                ax.set_xlabel('Queue Size', fontweight='bold')
                ax.set_ylabel(f'{label} Latency (ns)', fontweight='bold')
                ax.set_title(f'{test_type.title()}: {label} - {interval}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                ax.set_yscale('log')

                # Format x-axis
                queue_sizes = sorted(interval_data['queue_size'].unique())
                ax.set_xticks(queue_sizes)
                ax.set_xticklabels([f'{int(qs):,}' if qs % 1 == 0 else f'{qs:,.1f}' for qs in queue_sizes])

                # Prevent scientific notation on axes
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}' if y >= 1 else f'{y:.2f}'))

            plt.tight_layout()

            if progress_mode:
                plt.savefig(percentile_plots_dir / f'{test_type}_{percentile}_progress.png',
                           dpi=600, bbox_inches='tight')
            else:
                plt.savefig(percentile_plots_dir / f'{test_type}_{percentile}.png',
                           dpi=600, bbox_inches='tight')
            plt.close()

        # Create combined plot with all percentiles
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']

        for i, (percentile, label, color, marker) in enumerate(zip(percentiles, percentile_labels, colors, markers)):
            for interval in intervals:
                interval_data = percentile_data[percentile_data['interval'] == interval]
                valid_data = interval_data[interval_data[f'time_{percentile}'] > 0]

                if not valid_data.empty:
                    ax.scatter(valid_data['queue_size'], valid_data[f'time_{percentile}'],
                             label=f'{label} - {interval}', marker=marker, color=color,
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Queue Size', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latency (ns)', fontweight='bold', fontsize=12)
        ax.set_title(f'{test_type.title()}: Drain Latency Percentiles\n(All Without Locks Category)',
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Prevent scientific notation on axes
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}' if x % 1 == 0 else f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}' if y >= 1 else f'{y:.2f}'))

        plt.tight_layout()

        if progress_mode:
            plt.savefig(percentile_plots_dir / f'{test_type}_percentiles_combined_progress.png',
                       dpi=600, bbox_inches='tight')
        else:
            plt.savefig(percentile_plots_dir / f'{test_type}_percentiles_combined.png',
                       dpi=600, bbox_inches='tight')
        plt.close()

        print(f"📊 Queue size percentile plots saved to {percentile_plots_dir}")

    def _create_native_percentile_combination_plots(self, drain_df, output_dir, progress_mode=False):
        """Create native-specific plots for different interval/stack/duration combinations"""
        print("📊 Creating native percentile combination plots...")

        # Get unique combinations
        combinations = drain_df[['interval', 'stack_depth', 'native_duration']].drop_duplicates()

        percentiles = ['p95', 'p99', 'p99_9']
        percentile_labels = ['P95', 'P99', 'P99.9']

        # Create plots for each combination
        for _, combo in combinations.iterrows():
            interval = combo['interval']
            stack_depth = combo['stack_depth']
            native_duration = combo['native_duration']

            combo_data = drain_df[
                (drain_df['interval'] == interval) &
                (drain_df['stack_depth'] == stack_depth) &
                (drain_df['native_duration'] == native_duration)
            ]

            if combo_data.empty:
                continue

            # Create subplot for each percentile
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for p_idx, (percentile, p_label) in enumerate(zip(percentiles, percentile_labels)):
                ax = axes[p_idx]

                # Plot each category on left axis
                categories = combo_data['category'].unique()
                for category in categories:
                    cat_data = combo_data[combo_data['category'] == category]
                    if not cat_data.empty:
                        ax.scatter(cat_data['queue_size'], cat_data[percentile],
                                 label=category, s=60, marker='o', alpha=0.8)

                # Add right axis for loss percentage
                if not combo_data.empty:
                    # Get unique queue sizes and their corresponding loss percentages
                    loss_data = combo_data.groupby('queue_size')['loss_percentage'].first().reset_index()
                    ax2 = ax.twinx()
                    ax2.plot(loss_data['queue_size'], loss_data['loss_percentage'], 'r--',
                            label='Loss %', linewidth=2, alpha=0.7, marker='o', markersize=4)
                    ax2.set_ylabel('Loss Percentage (%)', fontsize=9, color='red')
                    ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
                    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

                ax.set_title(f'{p_label}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Queue Size', fontsize=10)
                ax.set_ylabel(f'{p_label} Time (ns)', fontsize=10)
                ax.grid(True, alpha=0.3)

                if p_idx == 0 and len(categories) > 1:
                    ax.legend(fontsize=8)

                # Set log scale if wide range
                if len(combo_data) > 1:
                    queue_range = combo_data['queue_size'].max() / combo_data['queue_size'].min()
                    if queue_range > 10:
                        ax.set_xscale('log')

                # Prevent scientific notation
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))

            plt.suptitle(f'Native Queue Percentiles - {interval}, Stack {stack_depth}, {native_duration}s',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            plot_filename = f"native_queue_percentiles_{interval}_stack{stack_depth}_dur{native_duration}s.png"
            if progress_mode:
                plot_filename = f"native_queue_percentiles_{interval}_stack{stack_depth}_dur{native_duration}s_progress.png"

            plt.savefig(output_dir / plot_filename, dpi=600, bbox_inches='tight')
            plt.close()

    def plot_signal_handler_duration_grid(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create grid plots for signal handler duration percentiles by queue size and interval"""
        print("📊 Creating signal handler duration grid plots...")

        # Filter out rows where we don't have signal handler data
        df_with_signals = df.dropna(subset=['loss_percentage'])

        if df_with_signals.empty:
            print("⚠️ No data with signal handler information found")
            return

        # Create the output directory for signal handler plots
        signal_plots_dir = PLOTS_DIR / "signal_handler_duration_grids"
        signal_plots_dir.mkdir(exist_ok=True)

        # Get unique intervals
        intervals = sorted(df_with_signals['interval'].unique())
        print(f"📊 Found {len(intervals)} intervals: {intervals}")

        # Define signal handler percentiles to track
        signal_percentiles = ['p50', 'p90', 'p95', 'p99', 'p99_9']
        signal_percentile_labels = {
            'p50': 'P50', 'p90': 'P90', 'p95': 'P95', 'p99': 'P99', 'p99_9': 'P99.9'
        }

        # Colors for different percentiles
        percentile_colors = {
            'p50': '#1f77b4',   # Blue
            'p90': '#ff7f0e',   # Orange
            'p95': '#2ca02c',   # Green
            'p99': '#d62728',   # Red
            'p99_9': '#9467bd'  # Purple
        }

        # Common signal types we expect to find
        signal_types = ['prof', 'quit', 'usr1', 'usr2']

        # Create grid plot with one subplot per interval, showing ALL percentiles in each plot
        if len(intervals) > 1:
            # Calculate grid dimensions
            if len(intervals) <= 3:
                cols = len(intervals)
                rows = 1
            elif len(intervals) <= 6:
                cols = 3
                rows = 2
            else:
                cols = 3
                rows = (len(intervals) + cols - 1) // cols

            # Create both normal and log scale versions
            for scale_type in ['normal', 'log']:
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                if len(intervals) == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if hasattr(axes, '__len__') else [axes]
                else:
                    axes = axes.flatten()

                for i, interval in enumerate(intervals):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    interval_data = df_with_signals[df_with_signals['interval'] == interval]

                    if interval_data.empty:
                        continue

                    # Extract signal handler data for this interval and plot ALL percentiles
                    plotted_any = False

                    # For each percentile, collect data across all signal types and plot
                    for percentile in signal_percentiles:
                        percentile_data = []

                        for _, row in interval_data.iterrows():
                            queue_size = row['queue_size']

                            # Look for signal handler data in different possible formats
                            for signal_type in signal_types:
                                # Try flattened CSV format first
                                signal_time_col = f'signal_{signal_type}_time_{percentile}_ns'
                                if signal_time_col in row and pd.notna(row[signal_time_col]) and row[signal_time_col] > 0:
                                    time_ns = row[signal_time_col]
                                    time_us = time_ns / 1000.0  # Convert to microseconds for better readability
                                    percentile_data.append((queue_size, time_us))

                        # Plot this percentile's data if we have any
                        if percentile_data:
                            queue_sizes = [d[0] for d in percentile_data]
                            times_us = [d[1] for d in percentile_data]

                            ax.scatter(queue_sizes, times_us,
                                     label=signal_percentile_labels[percentile],
                                     alpha=0.7, s=60, marker='o',
                                     color=percentile_colors[percentile])
                            plotted_any = True

                    if plotted_any:
                        scale_title = f'{interval} Interval - All Percentiles'
                        if scale_type == 'log':
                            scale_title += ' (Log Scale)'
                        ax.set_title(scale_title, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Queue Size', fontsize=10)
                        ax.set_ylabel(f'Signal Handler Time (μs)', fontsize=10)
                        ax.grid(True, alpha=0.3)

                        # Add legend to each subplot for clarity
                        ax.legend(fontsize=8, loc='upper left')

                        # Prevent scientific notation
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

                        # Set scales based on type
                        if scale_type == 'log':
                            ax.set_xscale('log')
                            ax.set_yscale('log')
                        elif len(interval_data) > 1:
                            # For normal scale, only set log x-axis if wide range
                            queue_range = interval_data['queue_size'].max() / interval_data['queue_size'].min()
                            if queue_range > 10:
                                ax.set_xscale('log')
                    else:
                        ax.text(0.5, 0.5, 'No Signal Data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=12, alpha=0.5)
                        ax.set_title(f'{interval} Interval - No Data', fontsize=12)

                # Hide unused subplots
                for i in range(len(intervals), len(axes)):
                    axes[i].set_visible(False)

                title_suffix = ' (Log Scale)' if scale_type == 'log' else ''
                plt.suptitle(f'Signal Handler Duration by Queue Size - All Percentiles{title_suffix}',
                           fontsize=16, fontweight='bold')
                plt.tight_layout()

                filename = f'signal_handler_duration_all_percentiles_grid_{scale_type}'
                if progress_mode:
                    filename += '_progress'
                filename += '.png'

                plt.savefig(signal_plots_dir / filename, dpi=600, bbox_inches='tight')
                plt.close()

        # Also create individual plots for each signal type showing all percentiles
        for signal_type in signal_types:
            # Create both normal and log scale versions
            for scale_type in ['normal', 'log']:
                if len(intervals) > 1:
                    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                    if len(intervals) == 1:
                        axes = [axes]
                    elif rows == 1:
                        axes = axes if hasattr(axes, '__len__') else [axes]
                    else:
                        axes = axes.flatten()

                    signal_found = False

                    for i, interval in enumerate(intervals):
                        if i >= len(axes):
                            break

                        ax = axes[i]
                        interval_data = df_with_signals[df_with_signals['interval'] == interval]

                        if interval_data.empty:
                            continue

                        plotted_any = False

                        # For each percentile, collect data for this specific signal type
                        for percentile in signal_percentiles:
                            percentile_data = []

                            for _, row in interval_data.iterrows():
                                queue_size = row['queue_size']

                                # Look for signal handler data for this specific signal type
                                signal_time_col = f'signal_{signal_type}_time_{percentile}_ns'
                                if signal_time_col in row and pd.notna(row[signal_time_col]) and row[signal_time_col] > 0:
                                    time_ns = row[signal_time_col]
                                    time_us = time_ns / 1000.0  # Convert to microseconds
                                    percentile_data.append((queue_size, time_us))

                            # Plot this percentile's data if we have any
                            if percentile_data:
                                queue_sizes = [d[0] for d in percentile_data]
                                times_us = [d[1] for d in percentile_data]

                                ax.scatter(queue_sizes, times_us,
                                         label=signal_percentile_labels[percentile],
                                         alpha=0.7, s=60, marker='o',
                                         color=percentile_colors[percentile])
                                plotted_any = True
                                signal_found = True

                        if plotted_any:
                            scale_title = f'{interval} Interval'
                            if scale_type == 'log':
                                scale_title += ' (Log Scale)'
                            ax.set_title(scale_title, fontsize=12, fontweight='bold')
                            ax.set_xlabel('Queue Size', fontsize=10)
                            ax.set_ylabel(f'{signal_type.upper()} Signal Time (μs)', fontsize=10)
                            ax.grid(True, alpha=0.3)

                            # Add legend to first subplot
                            if i == 0:
                                ax.legend(fontsize=8, loc='upper left')

                            # Prevent scientific notation
                            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

                            # Set scales based on type
                            if scale_type == 'log':
                                ax.set_xscale('log')
                                ax.set_yscale('log')
                            elif len(interval_data) > 1:
                                # For normal scale, only set log x-axis if wide range
                                queue_range = interval_data['queue_size'].max() / interval_data['queue_size'].min()
                                if queue_range > 10:
                                    ax.set_xscale('log')
                        else:
                            ax.text(0.5, 0.5, f'No {signal_type.upper()} Data', transform=ax.transAxes,
                                   ha='center', va='center', fontsize=12, alpha=0.5)
                            ax.set_title(f'{interval} Interval - No Data', fontsize=12)

                    # Hide unused subplots
                    for i in range(len(intervals), len(axes)):
                        axes[i].set_visible(False)

                    if signal_found:
                        title_suffix = ' (Log Scale)' if scale_type == 'log' else ''
                        plt.suptitle(f'{signal_type.upper()} Signal Handler Duration - All Percentiles{title_suffix}',
                                   fontsize=16, fontweight='bold')
                        plt.tight_layout()

                        filename = f'signal_handler_{signal_type}_all_percentiles_grid_{scale_type}'
                        if progress_mode:
                            filename += '_progress'
                        filename += '.png'

                        plt.savefig(signal_plots_dir / filename, dpi=600, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.close()  # Close the figure if no data was found

        print(f"📊 Signal handler duration grid plots saved to {signal_plots_dir}")

    def plot_drainage_duration_grid(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create grid plots for drainage duration percentiles by queue size and interval"""
        print("📊 Creating drainage duration grid plots...")

        # Load drain statistics using direct parsing
        drain_df = self.load_drain_statistics()
        if drain_df is None or drain_df.empty:
            print("⚠️ No drain statistics available for drainage duration grid plots")
            return

        # Create the output directory for drainage plots
        drainage_plots_dir = PLOTS_DIR / "drainage_duration_grids"
        drainage_plots_dir.mkdir(exist_ok=True)

        # Get unique intervals from the drain data
        intervals = sorted(drain_df['interval'].unique())
        print(f"📊 Found {len(intervals)} intervals: {intervals}")

        # Define drainage percentiles to track
        drainage_percentiles = ['p50', 'p90', 'p95', 'p99', 'p99_9']
        drainage_percentile_labels = {
            'p50': 'P50', 'p90': 'P90', 'p95': 'P95', 'p99': 'P99', 'p99_9': 'P99.9'
        }

        # Colors for different percentiles
        percentile_colors = {
            'p50': '#1f77b4',   # Blue
            'p90': '#ff7f0e',   # Orange
            'p95': '#2ca02c',   # Green
            'p99': '#d62728',   # Red
            'p99_9': '#9467bd'  # Purple
        }

        # Common drain categories we expect to find
        drain_categories = ['all without locks', 'out of thread', 'with locks', 'total']

        # Create grid plot with one subplot per interval, showing ALL percentiles in each plot
        if len(intervals) > 1:
            # Calculate grid dimensions
            if len(intervals) <= 3:
                cols = len(intervals)
                rows = 1
            elif len(intervals) <= 6:
                cols = 3
                rows = 2
            else:
                cols = 3
                rows = (len(intervals) + cols - 1) // cols

            # Create both normal and log scale versions
            for scale_type in ['normal', 'log']:
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                if len(intervals) == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if hasattr(axes, '__len__') else [axes]
                else:
                    axes = axes.flatten()

                for i, interval in enumerate(intervals):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    interval_data = drain_df[drain_df['interval'] == interval]

                    if interval_data.empty:
                        continue

                    # Extract drainage data for this interval and plot ALL percentiles
                    plotted_any = False

                    # For each percentile, collect data across all drain categories and plot
                    for percentile in drainage_percentiles:
                        percentile_data = []

                        for _, row in interval_data.iterrows():
                            queue_size = row['queue_size']

                            # Look for drainage time data in different possible formats
                            # Try flattened CSV format first (time_p50_ns, time_p90_ns, etc.)
                            time_col = f'time_{percentile}_ns'
                            if time_col in row and pd.notna(row[time_col]) and row[time_col] > 0:
                                time_ns = row[time_col]
                                time_us = time_ns / 1000.0  # Convert to microseconds for better readability
                                percentile_data.append((queue_size, time_us))

                            # Also try drain_statistics dict format
                            elif 'drain_statistics' in row and isinstance(row['drain_statistics'], dict):
                                drain_stats = row['drain_statistics']
                                if 'time' in drain_stats and isinstance(drain_stats['time'], dict):
                                    time_stats = drain_stats['time']
                                    if percentile in time_stats and time_stats[percentile] > 0:
                                        time_ns = time_stats[percentile]
                                        time_us = time_ns / 1000.0  # Convert to microseconds
                                        percentile_data.append((queue_size, time_us))

                        # Plot this percentile's data if we have any
                        if percentile_data:
                            queue_sizes = [d[0] for d in percentile_data]
                            times_us = [d[1] for d in percentile_data]

                            ax.scatter(queue_sizes, times_us,
                                     label=drainage_percentile_labels[percentile],
                                     alpha=0.7, s=60, marker='o',
                                     color=percentile_colors[percentile])
                            plotted_any = True

                    if plotted_any:
                        scale_title = f'{interval} Interval - All Percentiles'
                        if scale_type == 'log':
                            scale_title += ' (Log Scale)'
                        ax.set_title(scale_title, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Queue Size', fontsize=10)
                        ax.set_ylabel(f'Drainage Time (μs)', fontsize=10)
                        ax.grid(True, alpha=0.3)

                        # Add legend to each subplot for clarity
                        ax.legend(fontsize=8, loc='upper left')

                        # Prevent scientific notation
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

                        # Set scales based on type
                        if scale_type == 'log':
                            ax.set_xscale('log')
                            ax.set_yscale('log')
                        elif len(interval_data) > 1:
                            # For normal scale, only set log x-axis if wide range
                            queue_range = interval_data['queue_size'].max() / interval_data['queue_size'].min()
                            if queue_range > 10:
                                ax.set_xscale('log')
                    else:
                        ax.text(0.5, 0.5, 'No Drainage Data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=12, alpha=0.5)
                        ax.set_title(f'{interval} Interval - No Data', fontsize=12)

                # Hide unused subplots
                for i in range(len(intervals), len(axes)):
                    axes[i].set_visible(False)

                title_suffix = ' (Log Scale)' if scale_type == 'log' else ''
                plt.suptitle(f'Drainage Duration by Queue Size - All Percentiles{title_suffix}',
                           fontsize=16, fontweight='bold')
                plt.tight_layout()

                filename = f'drainage_duration_all_percentiles_grid_{scale_type}'
                if progress_mode:
                    filename += '_progress'
                filename += '.png'

                plt.savefig(drainage_plots_dir / filename, dpi=600, bbox_inches='tight')
                plt.close()        # Also create individual plots for each drain category showing all percentiles
        for category in drain_categories:
            # Create both normal and log scale versions
            for scale_type in ['normal', 'log']:
                if len(intervals) > 1:
                    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                    if len(intervals) == 1:
                        axes = [axes]
                    elif rows == 1:
                        axes = axes if hasattr(axes, '__len__') else [axes]
                    else:
                        axes = axes.flatten()

                    category_found = False

                    for i, interval in enumerate(intervals):
                        if i >= len(axes):
                            break

                        ax = axes[i]
                        interval_data = drain_df[
                            (drain_df['interval'] == interval) &
                            (drain_df['drain_category'] == category)
                        ]

                        if interval_data.empty:
                            continue

                        plotted_any = False

                        # For each percentile, collect data for this specific category
                        for percentile in drainage_percentiles:
                            percentile_data = []

                            for _, row in interval_data.iterrows():
                                queue_size = row['queue_size']

                                # Look for drainage time data
                                time_col = f'time_{percentile}_ns'
                                if time_col in row and pd.notna(row[time_col]) and row[time_col] > 0:
                                    time_ns = row[time_col]
                                    time_us = time_ns / 1000.0  # Convert to microseconds
                                    percentile_data.append((queue_size, time_us))

                                # Also try drain_statistics dict format
                                elif 'drain_statistics' in row and isinstance(row['drain_statistics'], dict):
                                    drain_stats = row['drain_statistics']
                                    if 'time' in drain_stats and isinstance(drain_stats['time'], dict):
                                        time_stats = drain_stats['time']
                                        if percentile in time_stats and time_stats[percentile] > 0:
                                            time_ns = time_stats[percentile]
                                            time_us = time_ns / 1000.0  # Convert to microseconds
                                            percentile_data.append((queue_size, time_us))

                            # Plot this percentile's data if we have any
                            if percentile_data:
                                queue_sizes = [d[0] for d in percentile_data]
                                times_us = [d[1] for d in percentile_data]

                                ax.scatter(queue_sizes, times_us,
                                         label=drainage_percentile_labels[percentile],
                                         alpha=0.7, s=60, marker='o',
                                         color=percentile_colors[percentile])
                                plotted_any = True
                                category_found = True

                        if plotted_any:
                            scale_title = f'{interval} Interval'
                            if scale_type == 'log':
                                scale_title += ' (Log Scale)'
                            ax.set_title(scale_title, fontsize=12, fontweight='bold')
                            ax.set_xlabel('Queue Size', fontsize=10)
                            ax.set_ylabel(f'{category.title()} Drainage Time (μs)', fontsize=10)
                            ax.grid(True, alpha=0.3)

                            # Add legend to first subplot
                            if i == 0:
                                ax.legend(fontsize=8, loc='upper left')

                            # Prevent scientific notation
                            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

                            # Set scales based on type
                            if scale_type == 'log':
                                ax.set_xscale('log')
                                ax.set_yscale('log')
                            elif len(interval_data) > 1:
                                # For normal scale, only set log x-axis if wide range
                                queue_range = interval_data['queue_size'].max() / interval_data['queue_size'].min()
                                if queue_range > 10:
                                    ax.set_xscale('log')
                        else:
                            ax.text(0.5, 0.5, f'No {category.title()} Data', transform=ax.transAxes,
                                   ha='center', va='center', fontsize=12, alpha=0.5)
                            ax.set_title(f'{interval} Interval - No Data', fontsize=12)

                    # Hide unused subplots
                    for i in range(len(intervals), len(axes)):
                        axes[i].set_visible(False)

                    if category_found:
                        title_suffix = ' (Log Scale)' if scale_type == 'log' else ''
                        plt.suptitle(f'{category.title()} Drainage Duration - All Percentiles{title_suffix}',
                                   fontsize=16, fontweight='bold')
                        plt.tight_layout()

                        safe_category = category.replace(' ', '_').replace('/', '_')
                        filename = f'drainage_duration_{safe_category}_all_percentiles_grid_{scale_type}'
                        if progress_mode:
                            filename += '_progress'
                        filename += '.png'

                        plt.savefig(drainage_plots_dir / filename, dpi=600, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.close()  # Close the figure if no data was found        print(f"📊 Drainage duration grid plots saved to {drainage_plots_dir}")

    def plot_vm_ops_loss_grid(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create grid plots for VM operations loss percentage by queue size with one plot per interval"""
        print("📊 Creating VM operations loss percentage grid plots...")

        # Filter out rows where we don't have loss data
        df_with_loss = df.dropna(subset=['loss_percentage'])

        if df_with_loss.empty:
            print("⚠️ No data with VM operations loss information found")
            return

        # Use only the latest entry for each unique configuration to avoid duplicates
        print(f"    📊 Deduplicating VM ops loss grid data: {len(df_with_loss)} rows before deduplication")

        # Group by configuration columns and take the last entry (most recent) for each group
        config_columns = ['queue_size', 'interval']
        if 'stack_depth' in df_with_loss.columns:
            config_columns.append('stack_depth')
        if 'native_duration' in df_with_loss.columns:
            config_columns.append('native_duration')

        df_deduplicated = df_with_loss.groupby(config_columns).last().reset_index()
        print(f"    📊 After deduplication: {len(df_deduplicated)} rows")

        # Create the output directory for VM ops loss plots
        vm_ops_grid_plots_dir = PLOTS_DIR / "vm_ops_loss_grids"
        vm_ops_grid_plots_dir.mkdir(exist_ok=True)

        # Get unique intervals
        intervals = sorted(df_with_loss['interval'].unique())
        print(f"📊 Found {len(intervals)} intervals: {intervals}")

        # Define VM operation categories to track
        vm_op_categories = [
            'no_vm_ops',
            'in_jfr_safepoint',
            'vm_op_handshakeallthreads',
            'vm_op_g1collectforfullgc',
            'vm_op_g1incremental',
            'vm_op_cleanup',
            'vm_op_other'
        ]

        vm_op_labels = {
            'no_vm_ops': 'No VM Ops',
            'in_jfr_safepoint': 'JFR Safepoint',
            'vm_op_handshakeallthreads': 'Handshake All Threads',
            'vm_op_g1collectforfullgc': 'G1 Full GC',
            'vm_op_g1incremental': 'G1 Incremental',
            'vm_op_cleanup': 'Cleanup',
            'vm_op_other': 'Other VM Ops'
        }

        # Create both normal and log scale versions
        for scale_type in ['normal', 'log']:
            # Create grid plot with one subplot per interval
            if len(intervals) > 1:
                # Calculate grid dimensions
                if len(intervals) <= 3:
                    cols = len(intervals)
                    rows = 1
                elif len(intervals) <= 6:
                    cols = 3
                    rows = 2
                else:
                    cols = 3
                    rows = (len(intervals) + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                if len(intervals) == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if hasattr(axes, '__len__') else [axes]
                else:
                    axes = axes.flatten()

                for i, interval in enumerate(intervals):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    interval_data = df_with_loss[df_with_loss['interval'] == interval]

                    if interval_data.empty:
                        continue

                    # Calculate VM operation loss percentages for this interval
                    vm_ops_data = []

                    for _, row in interval_data.iterrows():
                        queue_size = row['queue_size']

                        # Get VM operations data from the results
                        vm_operations = {}
                        total_lost_samples = 0

                        # Try to get from raw dict format first
                        if 'vm_operations' in row and isinstance(row['vm_operations'], dict):
                            vm_operations = row['vm_operations']
                            total_lost_samples = row.get('total_lost_samples', 0)
                        else:
                            # Try to get from flattened CSV format
                            for category in vm_op_categories:
                                csv_col = f'vm_op_{category}'
                                if csv_col in row:
                                    vm_operations[category] = row[csv_col] if pd.notna(row[csv_col]) else 0
                                    total_lost_samples += vm_operations[category]

                            if total_lost_samples == 0:
                                total_lost_samples = row.get('total_lost_samples', 0)

                        if not vm_operations or total_lost_samples == 0:
                            continue

                        # Calculate loss percentages for each VM operation category
                        overall_loss_pct = row['loss_percentage']

                        for category in vm_op_categories:
                            category_count = vm_operations.get(category, 0)
                            if total_lost_samples > 0:
                                category_loss_pct = (category_count / total_lost_samples) * overall_loss_pct
                            else:
                                category_loss_pct = 0.0

                            if category_loss_pct > 0:  # Only plot if there's actual loss
                                vm_ops_data.append({
                                    'queue_size': queue_size,
                                    'vm_op_category': category,
                                    'loss_percentage': category_loss_pct
                                })

                    # Plot the data for this interval - show all VM ops as different series
                    plotted_any = False
                    colors = plt.cm.Set1(np.linspace(0, 1, len(vm_op_categories)))

                    for j, category in enumerate(vm_op_categories):
                        cat_data = [d for d in vm_ops_data if d['vm_op_category'] == category]
                        if cat_data:
                            queue_sizes = [d['queue_size'] for d in cat_data]
                            loss_percentages = [d['loss_percentage'] for d in cat_data]

                            label = vm_op_labels.get(category, category)
                            ax.scatter(queue_sizes, loss_percentages,
                                     label=label, alpha=0.7, s=60, marker='o',
                                     color=colors[j])
                            plotted_any = True

                    if plotted_any:
                        scale_title = f'{interval} Interval'
                        if scale_type == 'log':
                            scale_title += ' (Log Scale)'
                        ax.set_title(scale_title, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Queue Size', fontsize=10)
                        ax.set_ylabel('VM Ops Loss Percentage (%)', fontsize=10)
                        ax.grid(True, alpha=0.3)

                        # Add legend only to first subplot
                        if i == 0:
                            ax.legend(fontsize=8, loc='upper left')

                        # Prevent scientific notation
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

                        # Set scales based on type
                        if scale_type == 'log':
                            ax.set_xscale('log')
                            ax.set_yscale('log')
                        elif len(interval_data) > 1:
                            # For normal scale, only set log x-axis if wide range
                            queue_range = interval_data['queue_size'].max() / interval_data['queue_size'].min()
                            if queue_range > 10:
                                ax.set_xscale('log')
                    else:
                        ax.text(0.5, 0.5, 'No VM Ops Loss Data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=12, alpha=0.5)
                        ax.set_title(f'{interval} Interval - No Data', fontsize=12)

                # Hide unused subplots
                for i in range(len(intervals), len(axes)):
                    axes[i].set_visible(False)

                title_suffix = ' (Log Scale)' if scale_type == 'log' else ''
                plt.suptitle(f'VM Operations Loss Percentage by Queue Size and Interval{title_suffix}',
                           fontsize=16, fontweight='bold')
                plt.tight_layout()

                filename = f'vm_ops_loss_percentage_grid_{scale_type}'
                if progress_mode:
                    filename += '_progress'
                filename += '.png'

                plt.savefig(vm_ops_grid_plots_dir / filename, dpi=600, bbox_inches='tight')
                plt.close()

        # Create both normal and log scale versions
        for scale_type in ['normal', 'log']:
            # Create individual plots for each interval showing all VM ops as different series
            for interval in intervals:
                interval_data = df_with_loss[df_with_loss['interval'] == interval]

                if interval_data.empty:
                    continue

                # Calculate VM operation loss percentages for this interval
                vm_ops_data = []

                for _, row in interval_data.iterrows():
                    queue_size = row['queue_size']

                    # Get VM operations data from the results
                    vm_operations = {}
                    total_lost_samples = 0

                    # Try to get from raw dict format first
                    if 'vm_operations' in row and isinstance(row['vm_operations'], dict):
                        vm_operations = row['vm_operations']
                        total_lost_samples = row.get('total_lost_samples', 0)
                    else:
                        # Try to get from flattened CSV format
                        for category in vm_op_categories:
                            csv_col = f'vm_op_{category}'
                            if csv_col in row:
                                vm_operations[category] = row[csv_col] if pd.notna(row[csv_col]) else 0
                                total_lost_samples += vm_operations[category]

                        if total_lost_samples == 0:
                            total_lost_samples = row.get('total_lost_samples', 0)

                    if not vm_operations or total_lost_samples == 0:
                        continue

                    # Calculate loss percentages for each VM operation category
                    overall_loss_pct = row['loss_percentage']

                    for category in vm_op_categories:
                        category_count = vm_operations.get(category, 0)
                        if total_lost_samples > 0:
                            category_loss_pct = (category_count / total_lost_samples) * overall_loss_pct
                        else:
                            category_loss_pct = 0.0

                        if category_loss_pct > 0:  # Only plot if there's actual loss
                            vm_ops_data.append({
                                'queue_size': queue_size,
                                'vm_op_category': category,
                                'loss_percentage': category_loss_pct
                            })

                # Create individual plot for this interval
                if vm_ops_data:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot all VM ops as different series
                    plotted_any = False
                    colors = plt.cm.Set1(np.linspace(0, 1, len(vm_op_categories)))

                    for j, category in enumerate(vm_op_categories):
                        cat_data = [d for d in vm_ops_data if d['vm_op_category'] == category]
                        if cat_data:
                            queue_sizes = [d['queue_size'] for d in cat_data]
                            loss_percentages = [d['loss_percentage'] for d in cat_data]

                            label = vm_op_labels.get(category, category)
                            ax.scatter(queue_sizes, loss_percentages,
                                     label=label, alpha=0.7, s=60, marker='o',
                                     color=colors[j])
                            plotted_any = True

                    if plotted_any:
                        scale_title = f'VM Operations Loss - {interval} Interval'
                        if scale_type == 'log':
                            scale_title += ' (Log Scale)'
                        ax.set_title(scale_title, fontsize=14, fontweight='bold')
                        ax.set_xlabel('Queue Size', fontsize=12)
                        ax.set_ylabel('VM Ops Loss Percentage (%)', fontsize=12)
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=10, loc='upper left')

                        # Prevent scientific notation
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

                        # Set scales based on type
                        if scale_type == 'log':
                            ax.set_xscale('log')
                            ax.set_yscale('log')
                        elif len(interval_data) > 1:
                            # For normal scale, only set log x-axis if wide range
                            queue_range = interval_data['queue_size'].max() / interval_data['queue_size'].min()
                            if queue_range > 10:
                                ax.set_xscale('log')

                        plt.tight_layout()

                        filename = f'vm_ops_loss_{interval}_all_categories_{scale_type}'
                        if progress_mode:
                            filename += '_progress'
                        filename += '.png'

                        plt.savefig(vm_ops_grid_plots_dir / filename, dpi=600, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.close()  # Close the figure if no data was plotted

        print(f"📊 VM operations loss grid plots saved to {vm_ops_grid_plots_dir.absolute()}")

    def plot_queue_memory_consumption(self, df: pd.DataFrame, progress_mode: bool = False):
        """Create plots showing loss percentage vs queue memory consumption"""

        try:
            # Check if we have the original loss_percentage from test results
            if 'loss_percentage' in df.columns:
                print("   ✅ Using main results DataFrame with existing loss percentages")
                loss_df = df.copy()

                # Ensure queue memory consumption is calculated consistently
                if 'queue_memory_consumption' not in loss_df.columns:
                    loss_df['queue_memory_consumption'] = loss_df['queue_size'] * 48

                # Add missing columns for main results DataFrame compatibility
                if 'final_max_queue_size' not in loss_df.columns:
                    loss_df['final_max_queue_size'] = loss_df['queue_size']  # Use original queue size as fallback
                if 'final_queue_size_increase_count' not in loss_df.columns:
                    loss_df['final_queue_size_increase_count'] = 0  # No dynamic sizing in main results

                # Debug: Print actual loss values being used in plots
                print(f"   🔍 DEBUG: Loss values from main results DataFrame:")
                for _, row in loss_df.iterrows():
                    queue_size = row.get('queue_size', 'N/A')
                    interval = row.get('interval', 'N/A')
                    loss_pct = row.get('loss_percentage', 'N/A')
                    print(f"      Queue={queue_size}, Interval={interval}, Loss={loss_pct}%")

            else:
                print("   📊 Using drain statistics DataFrame - aggregating and calculating loss percentages")
                # Use aggregated data across all drain categories for better coverage
                # Group by test configuration to get total events and drains
                group_columns = [
                    'test_type', 'queue_size', 'interval', 'log_file',
                    'final_max_queue_size', 'queue_memory_consumption', 'final_queue_size_increase_count'
                ]

                # Add test-specific fields if they exist
                if 'stack_depth' in df.columns:
                    group_columns.extend(['stack_depth', 'test_duration', 'native_duration'])
                if 'iterations' in df.columns:
                    group_columns.append('iterations')

                loss_df = df.groupby(group_columns).agg({
                    'total_events': 'sum',
                    'drains': 'sum'
                }).reset_index()

                if loss_df.empty:
                    print("⚠️ No drain data found for queue memory analysis")
                    return

                # Ensure queue memory consumption is calculated consistently
                loss_df['queue_memory_consumption'] = loss_df['queue_size'] * 48

                # Calculate loss percentage from drain statistics
                loss_df['loss_events'] = loss_df['total_events'] - loss_df['drains']
                loss_df['loss_percentage'] = np.where(
                    loss_df['total_events'] > 0,
                    (loss_df['loss_events'] / loss_df['total_events']) * 100,
                    0
                )

                # Debug: Print calculated loss values being used in plots
                print(f"   🔍 DEBUG: Calculated loss values from drain statistics:")
                for _, row in loss_df.iterrows():
                    queue_size = row.get('queue_size', 'N/A')
                    interval = row.get('interval', 'N/A')
                    total_events = row.get('total_events', 'N/A')
                    drains = row.get('drains', 'N/A')
                    loss_pct = row.get('loss_percentage', 'N/A')
                    print(f"      Queue={queue_size}, Interval={interval}, Total={total_events}, Drains={drains}, Loss={loss_pct}%")

            # Create output directory
            queue_memory_plots_dir = PLOTS_DIR / "queue_memory_plots"
            queue_memory_plots_dir.mkdir(parents=True, exist_ok=True)

            # Get unique intervals
            intervals = sorted(loss_df['interval'].unique())
            print(f"📊 Creating queue memory consumption plots for intervals: {intervals}")

            # Create individual plots for each interval
            for interval in intervals:
                self._create_loss_vs_memory_plot(loss_df, interval, queue_memory_plots_dir, progress_mode)
                self._create_queue_increments_plot(loss_df, interval, queue_memory_plots_dir, progress_mode)

            # Create combined grid plots for both plot types
            self._create_loss_vs_memory_grid_plot(loss_df, intervals, queue_memory_plots_dir, progress_mode)
            self._create_queue_increments_grid_plot(loss_df, intervals, queue_memory_plots_dir, progress_mode)

            # Create raw data tables for all plots
            self._create_memory_plots_data_tables(loss_df, intervals, queue_memory_plots_dir, progress_mode)

            # Create summary statistics
            self._create_queue_memory_summary(loss_df, queue_memory_plots_dir)

            print(f"✅ Queue memory consumption plots saved to {queue_memory_plots_dir.absolute()}")

        except Exception as e:
            print(f"❌ Error creating queue memory consumption plots: {e}")
            print("🔄 Benchmark will continue without queue memory plots")

    def _create_loss_vs_memory_plot(self, df: pd.DataFrame, interval: str,
                                   output_dir: Path, progress_mode: bool):
        """Create dual-axis plot: Loss Percentage vs Queue Memory Consumption"""

        df_interval = df[df['interval'] == interval].copy()

        if df_interval.empty:
            print(f"⚠️ No data for interval {interval}")
            return

        # Debug: Print data being plotted for this interval
        print(f"   🔍 DEBUG: Plotting data for interval {interval}:")
        for _, row in df_interval.iterrows():
            queue_size = row.get('queue_size', 'N/A')
            loss_pct = row.get('loss_percentage', 'N/A')
            memory = row.get('queue_memory_consumption', 'N/A')
            print(f"      Queue={queue_size}, Loss={loss_pct}%, Memory={memory} bytes")

        # Sort by queue size for better line plotting
        df_interval = df_interval.sort_values('queue_size')

        # Create figure with dual y-axis
        plt.style.use('seaborn-v0_8')
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Plot loss percentage on left axis
        color1 = 'tab:red'
        ax1.set_xlabel('Queue Size', fontsize=12)
        ax1.set_ylabel('Loss Percentage (%)', color=color1, fontsize=12)
        line1 = ax1.plot(df_interval['queue_size'], df_interval['loss_percentage'],
                         'o', color=color1, markersize=8, label='Loss %')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.6)  # Make grid more visible

        # Plot queue memory consumption on right axis
        color2 = 'tab:blue'
        ax2.set_ylabel('Queue Memory Consumption (bytes)', color=color2, fontsize=12)
        line2 = ax2.plot(df_interval['queue_size'], df_interval['queue_memory_consumption'],
                         's', color=color2, markersize=8, label='Memory')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Format y-axis to avoid scientific notation
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}' if np.isfinite(x) else '0'))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}' if np.isfinite(x) else '0'))

        # Set title
        plt.title(f'Loss Percentage vs Queue Memory Consumption\nInterval: {interval}',
                  fontsize=14, fontweight='bold')

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Adjust layout
        plt.tight_layout()

        # Save plot
        filename = self._get_plot_filename(f"loss_vs_memory_{interval.replace('ms', 'ms')}", progress_mode)
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📈 Saved loss vs memory plot: {filepath.absolute()}")

    def _create_queue_increments_plot(self, df: pd.DataFrame, interval: str,
                                    output_dir: Path, progress_mode: bool):
        """Create single-focus plot: Queue Size Increase Count"""

        df_interval = df[df['interval'] == interval].copy()

        if df_interval.empty:
            print(f"⚠️ No data for interval {interval}")
            return

        # Sort by queue size for better line plotting
        df_interval = df_interval.sort_values('queue_size')

        # Handle NaN values in queue size increase count
        df_interval['final_queue_size_increase_count'] = df_interval['final_queue_size_increase_count'].fillna(0)

        # Create figure with single y-axis
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot queue size increase count
        color = 'tab:green'
        ax.set_xlabel('Queue Size', fontsize=12)
        ax.set_ylabel('Queue Size Increase Count', color=color, fontsize=12)
        ax.plot(df_interval['queue_size'], df_interval['final_queue_size_increase_count'],
                '^', color=color, markersize=10, label='Queue Increases')
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.6)  # Make grid more visible

        # Format y-axis to avoid scientific notation
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if np.isfinite(x) else '0'))

        # Set y-axis minimum to 0 for better visualization
        ax.set_ylim(bottom=0)

        # Set title
        plt.title(f'Queue Size Increase Count by Queue Size\nInterval: {interval}',
                  fontsize=14, fontweight='bold')

        # Add legend
        ax.legend(loc='upper left')

        # Adjust layout
        plt.tight_layout()

        # Save plot
        filename = self._get_plot_filename(f"queue_increments_{interval.replace('ms', 'ms')}", progress_mode)
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📈 Saved queue increments plot: {filepath.absolute()}")

    def _create_loss_vs_memory_grid_plot(self, df: pd.DataFrame, intervals: List[str],
                                       output_dir: Path, progress_mode: bool):
        """Create combined grid plot: Loss vs Memory for all intervals"""

        n_intervals = len(intervals)
        cols = min(3, n_intervals)  # Max 3 columns
        rows = (n_intervals + cols - 1) // cols

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Find global y-axis limits for consistent scaling
        all_loss = df['loss_percentage'].dropna()
        all_memory = df['queue_memory_consumption'].dropna()

        loss_min, loss_max = 0, max(all_loss.max() * 1.1, 1) if not all_loss.empty else 1
        memory_min, memory_max = 0, max(all_memory.max() * 1.1, 1000) if not all_memory.empty else 1000

        # Ensure min and max are valid finite numbers
        def ensure_valid_limits(min_val, max_val, default_max=1):
            if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
                return 0, default_max
            return min_val, max_val

        loss_min, loss_max = ensure_valid_limits(loss_min, loss_max, 1)
        memory_min, memory_max = ensure_valid_limits(memory_min, memory_max, 1000)

        for i, interval in enumerate(intervals):
            if i >= len(axes):
                break

            ax1 = axes[i]
            ax2 = ax1.twinx()

            df_interval = df[df['interval'] == interval].copy()
            df_interval = df_interval.sort_values('queue_size')

            if not df_interval.empty:
                # Plot loss percentage
                color1 = 'tab:red'
                ax1.plot(df_interval['queue_size'], df_interval['loss_percentage'],
                        'o', color=color1, markersize=6)
                ax1.set_ylabel('Loss %', color=color1, fontsize=10)
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=8)

                # Plot queue memory consumption
                color2 = 'tab:blue'
                ax2.plot(df_interval['queue_size'], df_interval['queue_memory_consumption'],
                        's', color=color2, markersize=6)
                ax2.set_ylabel('Memory (bytes)', color=color2, fontsize=10)
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)

                # Set consistent y-axis limits
                ax1.set_ylim(loss_min, loss_max)
                ax2.set_ylim(memory_min, memory_max)

                # Format y-axis to avoid scientific notation
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}' if np.isfinite(x) else '0'))
                ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}' if np.isfinite(x) else '0'))

            ax1.set_xlabel('Queue Size', fontsize=10)
            ax1.set_title(f'{interval}', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.6)  # Make grid more visible
            ax1.tick_params(axis='x', labelsize=8)

        # Hide unused subplots
        for i in range(len(intervals), len(axes)):
            axes[i].set_visible(False)

        # Main title
        fig.suptitle('Loss Percentage vs Queue Memory Consumption - All Intervals',
                    fontsize=16, fontweight='bold')

        # Adjust layout
        plt.tight_layout()

        # Save plot
        filename = self._get_plot_filename("loss_vs_memory_grid", progress_mode)
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📊 Saved loss vs memory grid plot: {filepath.absolute()}")

    def _create_queue_increments_grid_plot(self, df: pd.DataFrame, intervals: List[str],
                                         output_dir: Path, progress_mode: bool):
        """Create combined grid plot: Queue Increments for all intervals"""

        n_intervals = len(intervals)
        cols = min(3, n_intervals)  # Max 3 columns
        rows = (n_intervals + cols - 1) // cols

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Find global y-axis limits for consistent scaling
        all_increases = df['final_queue_size_increase_count'].dropna()

        # Handle case where all_increases might be empty or contain only zeros
        if all_increases.empty:
            all_increases = pd.Series([0, 1])  # Default fallback
        elif all_increases.max() == 0:
            all_increases = pd.Series([0, 1])  # Add minimal range for plotting

        increases_min, increases_max = 0, max(all_increases.max() * 1.1, 1)

        # Ensure min and max are valid finite numbers
        def ensure_valid_limits(min_val, max_val, default_max=1):
            if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
                return 0, default_max
            return min_val, max_val

        increases_min, increases_max = ensure_valid_limits(increases_min, increases_max, 1)

        for i, interval in enumerate(intervals):
            if i >= len(axes):
                break

            ax = axes[i]
            df_interval = df[df['interval'] == interval].copy()
            df_interval = df_interval.sort_values('queue_size')

            if not df_interval.empty:
                # Handle NaN values in queue size increase count
                df_interval['final_queue_size_increase_count'] = df_interval['final_queue_size_increase_count'].fillna(0)

                # Plot queue size increase count
                color = 'tab:green'
                ax.plot(df_interval['queue_size'], df_interval['final_queue_size_increase_count'],
                       '^', color=color, markersize=8)
                ax.set_ylabel('Queue Increases', color=color, fontsize=10)
                ax.tick_params(axis='y', labelcolor=color, labelsize=8)

                # Set consistent y-axis limits
                ax.set_ylim(increases_min, increases_max)

                # Format y-axis to avoid scientific notation
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}' if np.isfinite(x) else '0'))

            ax.set_xlabel('Queue Size', fontsize=10)
            ax.set_title(f'{interval}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.6)  # Make grid more visible
            ax.tick_params(axis='x', labelsize=8)

        # Hide unused subplots
        for i in range(len(intervals), len(axes)):
            axes[i].set_visible(False)

        # Main title
        fig.suptitle('Queue Size Increase Count - All Intervals',
                    fontsize=16, fontweight='bold')

        # Adjust layout
        plt.tight_layout()

        # Save plot
        filename = self._get_plot_filename("queue_increments_grid", progress_mode)
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  📊 Saved queue increments grid plot: {filepath.absolute()}")

    def _create_queue_memory_summary(self, df: pd.DataFrame, output_dir: Path):
        """Create summary statistics table"""

        summary_data = []

        for interval in sorted(df['interval'].unique()):
            df_interval = df[df['interval'] == interval]

            if not df_interval.empty:
                summary_data.append({
                    'Interval': interval,
                    'Queue Sizes': f"{df_interval['queue_size'].min()}-{df_interval['queue_size'].max()}",
                    'Avg Loss %': f"{df_interval['loss_percentage'].mean():.2f}",
                    'Max Loss %': f"{df_interval['loss_percentage'].max():.2f}",
                    'Max Memory (MB)': f"{df_interval['queue_memory_consumption'].max() / (1024*1024):.2f}",
                    'Max Increases': f"{df_interval['final_queue_size_increase_count'].max()}",
                    'Avg Increases': f"{df_interval['final_queue_size_increase_count'].mean():.1f}",
                    'Data Points': len(df_interval)
                })

        summary_df = pd.DataFrame(summary_data)

        # Save as CSV
        csv_path = output_dir / "queue_memory_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        # Save as formatted text
        txt_path = output_dir / "queue_memory_summary.txt"
        with open(txt_path, 'w') as f:
            f.write("Queue Memory Consumption Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\nNotes:\n")
            f.write("- Loss % = (total_events - drained_events) / total_events * 100\n")
            f.write("- Memory consumption = queue_size * 48 bytes per entry\n")
            f.write("- Queue increases = number of times queue size was increased dynamically\n")
            f.write("- Data from 'out of thread' drain category\n")

        print(f"  📄 Saved summary: queue_memory_summary.csv/txt")

    def _create_memory_plots_data_tables(self, df: pd.DataFrame, intervals: List[str],
                                       output_dir: Path, progress_mode: bool):
        """Create raw data tables for all memory plots"""

        try:
            print(f"  📄 Creating raw data tables for memory plots...")

            # Create data tables directory
            data_tables_dir = output_dir / "data_tables"
            data_tables_dir.mkdir(parents=True, exist_ok=True)

            # Get the filename prefix based on progress mode
            prefix = "progress_" if progress_mode else ""

            # 1. Create individual data tables for each interval (loss vs memory)
            for interval in intervals:
                df_interval = df[df['interval'] == interval].copy()
                if df_interval.empty:
                    continue

                # Sort by queue size for consistency
                df_interval = df_interval.sort_values('queue_size')

                # Select relevant columns for loss vs memory plot
                loss_memory_cols = [
                    'queue_size',
                    'loss_percentage',
                    'queue_memory_consumption',
                    'interval'
                ]

                # Add optional columns if they exist
                optional_cols = ['test_type', 'stack_depth', 'test_duration', 'native_duration', 'iterations']
                for col in optional_cols:
                    if col in df_interval.columns:
                        loss_memory_cols.append(col)

                loss_memory_data = df_interval[loss_memory_cols].copy()

                # Save loss vs memory data
                filename = f"{prefix}loss_vs_memory_{interval.replace('ms', 'ms')}_data.csv"
                csv_path = data_tables_dir / filename
                loss_memory_data.to_csv(csv_path, index=False)

                # Save as formatted text
                txt_filename = f"{prefix}loss_vs_memory_{interval.replace('ms', 'ms')}_data.txt"
                txt_path = data_tables_dir / txt_filename
                with open(txt_path, 'w') as f:
                    f.write(f"Loss Percentage vs Queue Memory Consumption Data\\n")
                    f.write(f"Interval: {interval}\\n")
                    f.write("=" * 60 + "\\n\\n")
                    f.write(loss_memory_data.to_string(index=False))
                    f.write("\\n\\nColumn Descriptions:\\n")
                    f.write("- queue_size: Initial queue size configuration\\n")
                    f.write("- loss_percentage: Percentage of samples lost due to queue overflow\\n")
                    f.write("- queue_memory_consumption: Memory used by queue (queue_size * 48 bytes)\\n")
                    f.write("- interval: Sampling interval\\n")

            # 2. Create individual data tables for queue increments
            for interval in intervals:
                df_interval = df[df['interval'] == interval].copy()
                if df_interval.empty:
                    continue

                # Sort by queue size
                df_interval = df_interval.sort_values('queue_size')

                # Handle NaN values in queue size increase count
                df_interval['final_queue_size_increase_count'] = df_interval['final_queue_size_increase_count'].fillna(0)

                # Select relevant columns for queue increments plot
                queue_inc_cols = [
                    'queue_size',
                    'final_queue_size_increase_count',
                    'final_max_queue_size',
                    'interval'
                ]

                # Add optional columns if they exist
                for col in optional_cols:
                    if col in df_interval.columns:
                        queue_inc_cols.append(col)

                queue_inc_data = df_interval[queue_inc_cols].copy()

                # Save queue increments data
                filename = f"{prefix}queue_increments_{interval.replace('ms', 'ms')}_data.csv"
                csv_path = data_tables_dir / filename
                queue_inc_data.to_csv(csv_path, index=False)

                # Save as formatted text
                txt_filename = f"{prefix}queue_increments_{interval.replace('ms', 'ms')}_data.txt"
                txt_path = data_tables_dir / txt_filename
                with open(txt_path, 'w') as f:
                    f.write(f"Queue Size Increase Count Data\\n")
                    f.write(f"Interval: {interval}\\n")
                    f.write("=" * 50 + "\\n\\n")
                    f.write(queue_inc_data.to_string(index=False))
                    f.write("\\n\\nColumn Descriptions:\\n")
                    f.write("- queue_size: Initial queue size configuration\\n")
                    f.write("- final_queue_size_increase_count: Number of times queue size was increased\\n")
                    f.write("- final_max_queue_size: Maximum queue size reached during test\\n")
                    f.write("- interval: Sampling interval\\n")

            # 3. Create combined data table for all intervals (loss vs memory)
            combined_loss_memory_cols = [
                'interval', 'queue_size', 'loss_percentage', 'queue_memory_consumption'
            ]
            for col in optional_cols:
                if col in df.columns:
                    combined_loss_memory_cols.append(col)

            combined_loss_memory = df[combined_loss_memory_cols].copy().sort_values(['interval', 'queue_size'])

            filename = f"{prefix}loss_vs_memory_all_intervals_data.csv"
            csv_path = data_tables_dir / filename
            combined_loss_memory.to_csv(csv_path, index=False)

            txt_filename = f"{prefix}loss_vs_memory_all_intervals_data.txt"
            txt_path = data_tables_dir / txt_filename
            with open(txt_path, 'w') as f:
                f.write("Loss Percentage vs Queue Memory Consumption - All Intervals\\n")
                f.write("=" * 70 + "\\n\\n")
                f.write(combined_loss_memory.to_string(index=False))
                f.write("\\n\\nData corresponds to the Loss vs Memory Grid Plot\\n")

            # 4. Create combined data table for all intervals (queue increments)
            df_with_increments = df.copy()
            df_with_increments['final_queue_size_increase_count'] = df_with_increments['final_queue_size_increase_count'].fillna(0)

            combined_queue_inc_cols = [
                'interval', 'queue_size', 'final_queue_size_increase_count', 'final_max_queue_size'
            ]
            for col in optional_cols:
                if col in df_with_increments.columns:
                    combined_queue_inc_cols.append(col)

            combined_queue_inc = df_with_increments[combined_queue_inc_cols].copy().sort_values(['interval', 'queue_size'])

            filename = f"{prefix}queue_increments_all_intervals_data.csv"
            csv_path = data_tables_dir / filename
            combined_queue_inc.to_csv(csv_path, index=False)

            txt_filename = f"{prefix}queue_increments_all_intervals_data.txt"
            txt_path = data_tables_dir / txt_filename
            with open(txt_path, 'w') as f:
                f.write("Queue Size Increase Count - All Intervals\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(combined_queue_inc.to_string(index=False))
                f.write("\\n\\nData corresponds to the Queue Increments Grid Plot\\n")

            print(f"  ✅ Raw data tables saved to {data_tables_dir.absolute()}")
            print(f"     - Individual interval data: loss_vs_memory_*_data.csv/txt")
            print(f"     - Individual interval data: queue_increments_*_data.csv/txt")
            print(f"     - Combined data: *_all_intervals_data.csv/txt")

        except Exception as e:
            print(f"  ⚠️ Error creating data tables: {e}")

def main():
    parser = argparse.ArgumentParser(description='JFR Queue Size Benchmark Suite')
    parser.add_argument('--minimal', action='store_true',
                       help='Run minimal benchmark (2 queue sizes, 2 intervals, 60s duration)')
    parser.add_argument('--run-native', action='store_true',
                       help='Run native benchmarks')
    parser.add_argument('--run-renaissance', action='store_true',
                       help='Run Renaissance benchmarks')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations from existing data')
    parser.add_argument('--visualize-progress', action='store_true',
                       help='Generate in-progress visualizations from live CSV data (updates continuously)')
    parser.add_argument('--queue-memory-plots', action='store_true',
                       help='Generate only queue memory consumption plots from drain statistics')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode: continuously monitor and update progress visualizations')
    parser.add_argument('--csv-file', type=str,
                       help='Specify custom CSV file for visualizations (used with --visualize-progress)')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks and generate visualizations')
    parser.add_argument('--estimate', action='store_true',
                       help='Show runtime estimates without running benchmarks')
    parser.add_argument('--only-native', action='store_true',
                       help='Run only native benchmarks (skip Renaissance)')
    parser.add_argument('--only-renaissance', action='store_true',
                       help='Run only Renaissance benchmarks (skip native)')
    parser.add_argument('--threads', type=int, default=None,
                       help=f'Number of threads to use (default: {os.cpu_count() or 4}, auto-detect CPU count)')
    parser.add_argument('--retries', type=int, default=2,
                       help='Number of retries for tests with JSON parsing failures (default: 2)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging for debugging and detailed output')
    parser.add_argument('--dynamic-queue-size', action='store_true',
                       help='Enable dynamic queue sizing (DYNAMIC_QUEUE_SIZE=true)')

    args = parser.parse_args()

    # If minimal flag is used without other flags, run only native (not renaissance)
    if args.minimal and not any([args.run_native, args.run_renaissance, args.visualize, args.visualize_progress, args.queue_memory_plots, args.watch, args.all, args.estimate, args.only_native, args.only_renaissance]):
        args.run_native = True

    if not any([args.run_native, args.run_renaissance, args.visualize, args.visualize_progress, args.queue_memory_plots, args.watch, args.all, args.estimate, args.minimal, args.only_native, args.only_renaissance]):
        parser.print_help()
        return

    # Create benchmark runner with minimal configuration if requested
    runner = BenchmarkRunner(minimal=args.minimal, threads=args.threads, max_retries=args.retries, verbose=args.verbose, dynamic_queue_size=args.dynamic_queue_size)

    # Show configuration info
    print(f"JFR Queue Size Benchmark Suite")
    print(f"{'='*50}")
    if args.minimal:
        print(f"Mode: Minimal (quick validation)")
        print(f"Queue sizes: {runner.queue_sizes}")
        print(f"Sampling intervals: {runner.sampling_intervals}")
        print(f"Native durations: {runner.native_durations}")
        print(f"Test duration: {runner.test_duration}s")
        print(f"Threads: {runner.threads}")
    else:
        print(f"Mode: Full (comprehensive analysis)")
        print(f"Queue sizes: {len(runner.queue_sizes)} values from {min(runner.queue_sizes)} to {max(runner.queue_sizes)}")
        print(f"Sampling intervals: {len(runner.sampling_intervals)} values from {runner.sampling_intervals[0]} to {runner.sampling_intervals[-1]}")
        print(f"Test duration: {runner.test_duration}s")
        print(f"Threads: {runner.threads}")

    if args.dynamic_queue_size:
        print(f"🔄 Dynamic Queue Size: ENABLED (DYNAMIC_QUEUE_SIZE=true)")
    else:
        print(f"📏 Dynamic Queue Size: DISABLED (static queue sizes)")
    print(f"{'='*50}")

    # Show estimates if requested
    if args.estimate:
        print("⏱️ Benchmark Runtime Estimates")

        # Special case: if only-renaissance, show only Renaissance estimates
        if args.only_renaissance:
            renaissance_seconds, renaissance_time = runner.estimate_runtime('renaissance')
            renaissance_tests = len(runner.queue_sizes) * len(runner.sampling_intervals) * len(runner.stack_depths)
            print(f"🏛️ Renaissance Benchmark:")
            print(f"   Tests: {renaissance_tests}")
            print(f"   Estimated time: {renaissance_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + renaissance_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

            total_seconds = renaissance_seconds
            total_tests = renaissance_tests
        else:
            # Show native estimates (unless only-renaissance)
            native_seconds, native_time = runner.estimate_runtime('native')
            native_tests = len(runner.queue_sizes) * len(runner.sampling_intervals) * len(runner.native_durations) * len(runner.stack_depths)
            print(f"🔧 Native Benchmark:")
            print(f"   Tests: {native_tests}")
            print(f"   Estimated time: {native_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + native_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

            # Show Renaissance estimates if:
            # 1. --all is specified (regardless of minimal mode)
            # 2. Not in minimal mode and not only-native
            # 3. --run-renaissance is specified
            if args.all or args.run_renaissance or (not args.minimal and not args.only_native):
                renaissance_seconds, renaissance_time = runner.estimate_runtime('renaissance')
                renaissance_tests = len(runner.queue_sizes) * len(runner.sampling_intervals) * len(runner.stack_depths)
                print(f"🏛️ Renaissance Benchmark:")
                print(f"   Tests: {renaissance_tests}")
                print(f"   Estimated time: {renaissance_time}")
                print(f"   Expected completion: {datetime.fromtimestamp(time.time() + renaissance_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

                total_seconds = native_seconds + renaissance_seconds
                total_tests = native_tests + renaissance_tests
            else:
                # Only native will run
                total_seconds = native_seconds
                total_tests = native_tests

        hours = int(total_seconds) // 3600
        minutes = (int(total_seconds) % 3600) // 60
        remaining_seconds = int(total_seconds) % 60
        if hours > 0:
            total_time = f"{hours}h {minutes}m {remaining_seconds}s"
        elif minutes > 0:
            total_time = f"{minutes}m {remaining_seconds}s"
        else:
            total_time = f"{remaining_seconds}s"

        # Determine the appropriate total label based on what will actually run
        if args.only_renaissance:
            print(f"🎯 Total (Renaissance Only):")
            print(f"   Tests: {total_tests}")
            print(f"   Estimated time: {total_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
        elif args.only_native:
            print(f"🎯 Total (Native Only):")
            print(f"   Tests: {total_tests}")
            print(f"   Estimated time: {total_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
        elif args.all or args.run_renaissance or (not args.minimal and not args.only_native):
            print(f"🎯 Combined Total:")
            print(f"   Tests: {total_tests}")
            print(f"   Estimated time: {total_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"🎯 Total (Native Only):")
            print(f"   Tests: {total_tests}")
            print(f"   Estimated time: {total_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n💡 Note: Estimates include test execution time plus setup/teardown overhead.")
        if not args.minimal and not args.only_native and not args.only_renaissance:
            print("💡 Renaissance estimate updated: 3:30 minutes per test + 30s overhead = 4 minutes total per test")
        return

    # Handle visualize-progress mode
    if args.visualize_progress or args.watch:
        if args.watch:
            runner.watch_progress_visualizations(args.csv_file)
        else:
            runner.create_progress_visualizations(args.csv_file)
        return

    # Handle queue memory plots only
    if args.queue_memory_plots:
        print("📊 Generating queue memory consumption plots...")
        benchmarker = BenchmarkRunner()

        # Try to load main results first (contains proper loss_percentage values)
        main_results_df = None
        try:
            # Look for the latest results file
            data_dir = Path("benchmark_results/data")
            if data_dir.exists():
                # Find the most recent native results file
                native_files = list(data_dir.glob("native_*.csv"))
                if native_files:
                    latest_file = max(native_files, key=lambda x: x.stat().st_mtime)
                    print(f"📂 Loading main results from: {latest_file}")
                    main_results_df = pd.read_csv(latest_file)
                    print(f"✅ Loaded {len(main_results_df)} main test results")

                    # Add queue memory consumption if not present
                    if 'queue_memory_consumption' not in main_results_df.columns:
                        main_results_df['queue_memory_consumption'] = main_results_df['queue_size'] * 48

                    # Use main results (contains original loss_percentage from run.sh)
                    benchmarker.plot_queue_memory_consumption(main_results_df)
                    print("✅ Queue memory consumption plots generated successfully using main results!")
                    return
        except Exception as e:
            print(f"⚠️ Could not load main results: {e}")

        # Fallback to drain statistics if main results not available
        if main_results_df is None:
            print("📊 Falling back to drain statistics...")
            drain_df = benchmarker.load_drain_statistics()
            if drain_df is not None:
                benchmarker.plot_queue_memory_consumption(drain_df)
                print("✅ Queue memory consumption plots generated using drain statistics!")
            else:
                print("❌ No drain statistics found. Make sure you have log files in benchmark_results/logs/")
        else:
            print("❌ No data found. Run some benchmarks first or check benchmark_results/ directory")
        return

    try:
        overall_start = time.time()

        if args.all or args.run_native or args.only_native:
            # Only run native if --only-renaissance is not set
            if not args.only_renaissance:
                runner.run_native_benchmark()

        if args.all or args.run_renaissance or args.only_renaissance:
            # Only run Renaissance if --only-native is not set
            if not args.only_native:
                runner.run_renaissance_benchmark()

        if args.all or args.visualize:
            runner.create_visualizations()

        if args.all or (args.run_native and args.run_renaissance):
            total_time = time.time() - overall_start
            print(f"\n🎉 All benchmarks complete!")
            print(f"   Total runtime: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")

    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        # Save partial results
        if runner.results['native']:
            runner.save_results('native')
        if runner.results['renaissance']:
            runner.save_results('renaissance')
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        # Save partial results
        if runner.results['native']:
            runner.save_results('native')
        if runner.results['renaissance']:
            runner.save_results('renaissance')
        sys.exit(1)

if __name__ == "__main__":
    main()
