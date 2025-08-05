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
QUEUE_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 750, 1000, 2000]
SAMPLING_INTERVALS = ["1ms", "2ms", "5ms", "10ms", "20ms"]
NATIVE_DURATIONS = [5, 250]  # seconds
STACK_DEPTHS = [100, 1200]  # Different stack depths to test
TEST_DURATION = 250  # seconds
THREADS = os.cpu_count() or 4  # Use number of CPUs, fallback to 4

# Minimal configuration for quick testing
MINIMAL_QUEUE_SIZES = [100, 500]
MINIMAL_SAMPLING_INTERVALS = ["1ms", "10ms"]
MINIMAL_NATIVE_DURATIONS = [10, 25]  # seconds
MINIMAL_STACK_DEPTHS = [100, 1200]  # Test both stack depths
MINIMAL_TEST_DURATION = 30  # seconds

# Renaissance specific
RENAISSANCE_ITERATIONS = 2
MINIMAL_RENAISSANCE_ITERATIONS = 1  # Faster minimal testing

# Output directories
RESULTS_DIR = Path("benchmark_results")
LOGS_DIR = RESULTS_DIR / "logs"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"

class BenchmarkRunner:
    def __init__(self, minimal=False, threads=None, max_retries=2, verbose=False):
        self.setup_directories()
        self.results = {
            'native': [],
            'renaissance': []
        }
        self.minimal = minimal
        self.max_retries = max_retries
        self.verbose = verbose

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
                return f"{base_name}_progress"
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
        for dir_path in [RESULTS_DIR, LOGS_DIR, DATA_DIR, PLOTS_DIR]:
            dir_path.mkdir(exist_ok=True)

    def vprint(self, *args, **kwargs):
        """Print only if verbose mode is enabled"""
        if self.verbose:
            print(*args, **kwargs)

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

            # Remove the complex nested fields from CSV output
            flattened.pop('out_of_thread_details', None)
            flattened.pop('all_without_locks_details', None)
            flattened.pop('drain_categories', None)

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
            "-f", str(log_path.absolute()) + ".raw",
        ]

        if native_duration:
            cmd.extend(["--native-duration", str(native_duration)])

        # Generate log filename

        self.vprint(f"    Command: {' '.join(cmd)}")
        self.vprint(f"    Log file: {log_path}")

        # Run test
        try:
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd="../run_in_native",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
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
                print(f"    Loss: {loss_percentage}%")
                if out_of_thread_percentage is not None:
                    print(f"    Out-of-thread: {out_of_thread_events:,} events ({out_of_thread_percentage:.2f}%)")
                if all_without_locks_events is not None:
                    self.vprint(f"    All-without-locks: {all_without_locks_events:,} events")
            else:
                # Fallback for old format
                loss_percentage = extracted_data
                out_of_thread_events = None
                out_of_thread_percentage = None
                out_of_thread_details = None
                all_without_locks_events = None
                all_without_locks_details = None
                print(f"    Loss: {loss_percentage}%")

            result_data = {
                'queue_size': queue_size,
                'interval': interval,
                'stack_depth': stack_depth,
                'native_duration': native_duration,
                'test_duration': self.test_duration,
                'loss_percentage': loss_percentage,
                'out_of_thread_events': out_of_thread_events,
                'out_of_thread_percentage': out_of_thread_percentage,
                'out_of_thread_details': out_of_thread_details,
                'all_without_locks_events': all_without_locks_events,
                'all_without_locks_details': all_without_locks_details,
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

    def run_renaissance_test(self, queue_size: int, interval: str, stack_depth: int) -> Dict:
        """Run a single Renaissance test with retry logic for JSON parsing failures"""
        return self.run_test_with_retry(self._run_renaissance_test_internal, queue_size, interval, stack_depth)

    def _run_renaissance_test_internal(self, queue_size: int, interval: str, stack_depth: int) -> Dict:
        """Run a single Renaissance test and extract loss percentage"""
        print(f"  Running Renaissance: queue={queue_size}, interval={interval}, stack_depth={stack_depth}")

        log_filename = f"renaissance_q{queue_size}_{interval}_s{stack_depth}_n{self.renaissance_iterations}.log"
        log_path = LOGS_DIR / log_filename

        # Build command
        cmd = [
            "./run.sh",
            "--mode", "renaissance",
            "-n", str(self.renaissance_iterations),
            "-s", str(stack_depth),
            "-t", str(self.threads),
            "-i", interval,
            "-q", str(queue_size),
            "--no-analysis",  # Enable analysis tables but skip plots/visualizations
            "-f", str(log_path.absolute()) + ".raw",
        ]

        # Generate log filename


        # Run test
        try:
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd="../run_in_native",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
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
                print(f"    Loss: {loss_percentage}%")
                if out_of_thread_percentage is not None:
                    print(f"    Out-of-thread: {out_of_thread_events:,} events ({out_of_thread_percentage:.2f}%)")
                if all_without_locks_events is not None:
                    self.vprint(f"    All-without-locks: {all_without_locks_events:,} events")
            else:
                loss_percentage = extracted_data
                out_of_thread_events = None
                out_of_thread_percentage = None
                out_of_thread_details = None
                all_without_locks_events = None
                all_without_locks_details = None
                print(f"    Loss: {loss_percentage}%")

            result_data = {
                'queue_size': queue_size,
                'interval': interval,
                'stack_depth': stack_depth,
                'iterations': self.renaissance_iterations,
                'loss_percentage': loss_percentage,
                'out_of_thread_events': out_of_thread_events,
                'out_of_thread_percentage': out_of_thread_percentage,
                'out_of_thread_details': out_of_thread_details,
                'all_without_locks_events': all_without_locks_events,
                'all_without_locks_details': all_without_locks_details,
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
                'stack_depth': stack_depth,
                'iterations': self.renaissance_iterations,
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
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

        try:
            self.vprint(f"    🔍 Using analysis script to parse: {log_file_path}")

            # Read all DRAIN_STATS_JSON lines from the log file
            all_stats = []
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'DRAIN_STATS_JSON:' in line:
                        match = re.search(r'DRAIN_STATS_JSON:\s*(\{.*\})', line)
                        if match:
                            try:
                                data = json.loads(match.group(1))
                                all_stats.append(data)
                            except json.JSONDecodeError as e:
                                self.vprint(f"    ⚠️ Error parsing JSON: {e}")
                                continue

            if not all_stats:
                self.vprint(f"    ⚠️ No DRAIN_STATS_JSON found in {log_file_path}")
                return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

            self.vprint(f"    📊 Found {len(all_stats)} drain statistics entries")

            # Extract drain categories and calculate event counts
            result = {
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_events': None,
                'all_without_locks_details': None,
                'drain_categories': {}
            }

            total_events_without_locks = 0
            total_events_all = 0
            out_of_thread_events = 0
            all_without_locks_events = 0

            # Process each drain category
            for category_name in DRAIN_CATEGORIES:
                category_stats = extract_drain_stats_by_label(all_stats, category_name)
                if category_stats:
                    stats = category_stats[0]  # Get the last (and only) instance

                    # Extract event information
                    events_data = stats.get('events', {})
                    events_sum = events_data.get('sum', 0)

                    # Extract time information
                    time_data = stats.get('time', {})

                    # Build category data structure
                    # Calculate event-weighted queue size percentiles
                    event_histogram = stats.get('event_histogram', [])
                    queue_percentiles = calculate_event_weighted_percentiles(event_histogram)

                    category_data = {
                        'requests': stats.get('drains', 0),
                        'runtime_seconds': stats.get('runtime_ns', 0) / 1000000000.0 if 'runtime_ns' in stats else 0.0,  # Convert ns to seconds
                        'runtime_minutes': (stats.get('runtime_ns', 0) / 1000000000.0) / 60.0 if 'runtime_ns' in stats else 0.0,
                        'request_rate': stats.get('drains', 0) / (stats.get('runtime_ns', 1) / 1000000000.0) if stats.get('runtime_ns', 0) > 0 else 0.0,
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
                            'median': queue_percentiles.get('median', 0),
                            'p95': queue_percentiles.get('p95', 0),
                            'p99': queue_percentiles.get('p99', 0),
                            'p99_9': queue_percentiles.get('p999', 0),
                            'p99_99': queue_percentiles.get('p9999', 0),
                            'p99_999': queue_percentiles.get('p99999', 0)
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
            # First try to read the corresponding .raw file for more detailed drain statistics
            raw_file_path = Path(str(log_path) + ".raw")
            if raw_file_path.exists():
                self.vprint(f"    🔍 Found .raw file: {raw_file_path}")

                # Prioritize using the analysis script on the .raw file since it contains DRAIN_STATS_JSON
                if ANALYSIS_AVAILABLE:
                    self.vprint(f"    🔍 Using analysis script on .raw file: {raw_file_path}")
                    result = self.parse_drain_stats_with_analysis_script(raw_file_path)
                    if result.get('out_of_thread_events') is not None or result.get('loss_percentage') is not None:
                        return result
                    self.vprint(f"    ⚠️ Analysis script didn't find data in .raw file, falling back to raw parsing...")

                return self.parse_raw_drain_stats(raw_file_path)

            # If no .raw file, try using the analysis script on the main log file
            if ANALYSIS_AVAILABLE:
                self.vprint(f"    🔍 No .raw file found, trying analysis script on main log: {log_path}")
                result = self.parse_drain_stats_with_analysis_script(log_path)
                if result.get('out_of_thread_events') is not None or result.get('loss_percentage') is not None:
                    return result
                self.vprint(f"    ⚠️ Analysis script didn't find data in main log, falling back to basic parsing...")

            # Fallback to parsing the main log file
            # Try different encodings to handle special characters
            content = None
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(log_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    self.vprint(f"    📝 Successfully read log with {encoding} encoding: {log_path}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                # Fallback: read as binary and decode with error replacement
                with open(log_path, 'rb') as f:
                    raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace')
                self.vprint(f"    📝 Read log with fallback binary mode: {log_path}")

            self.vprint(f"    📏 Log size: {len(content):,} characters")

            result = {
                'loss_percentage': None,
                'out_of_thread_events': None,
                'out_of_thread_percentage': None,
                'out_of_thread_details': None,
                'all_without_locks_events': None,
                'all_without_locks_details': None
            }

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
                self.vprint(f"    ⚠️ Could not find any drain statistics sections with events")            # First try direct search for all statistics in entire content
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

                    # Return loss percentage and out-of-thread data
                    result['loss_percentage'] = loss_percentage
                    return result
                else:
                    self.vprint(f"    ⚠️ Total attempted samples is 0")
                    result['loss_percentage'] = 0.0
                    return result

            # If direct search didn't work, try the original section-based approach

            # Look for the loss rate in the final summary
            loss_pattern = r'Loss Rate:\s+([\d.]+)%'
            match = re.search(loss_pattern, content)

            if match:
                loss_rate = float(match.group(1))
                self.vprint(f"    Found loss rate pattern: {loss_rate}%")
                result['loss_percentage'] = loss_rate
                return result

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
                        result['loss_percentage'] = loss_percentage
                        return result
                    else:
                        print(f"    ⚠️ Total attempted samples is 0")
                        result['loss_percentage'] = 0.0
                        return result
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
            # Show last few lines of log for debugging
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
        print(f"   Results saved to {DATA_DIR}")

        # Generate final comprehensive real-time plots
        print(f"   📊 Generating final real-time plots...")
        self.plot_realtime_progress('native')

        self.save_results('native')

    def run_renaissance_benchmark(self):
        """Run comprehensive Renaissance benchmark"""
        print("🏛️ Starting Renaissance Benchmark Suite")

        # Calculate and display runtime estimate
        total_seconds, time_str = self.estimate_runtime('renaissance')
        total_tests = len(self.queue_sizes) * len(self.sampling_intervals) * len(self.stack_depths)

        print(f"📊 Benchmark Overview:")
        print(f"  Total tests: {total_tests}")
        print(f"  Estimated runtime: {time_str}")
        print(f"  Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n⚙️ Configuration:")
        print(f"  Queue sizes: {self.queue_sizes}")
        print(f"  Intervals: {self.sampling_intervals}")
        print(f"  Stack depths: {self.stack_depths}")
        print(f"  Iterations: {self.renaissance_iterations}")
        print(f"  Threads: {self.threads}")

        current_test = 0
        start_time = time.time()

        for queue_size in self.queue_sizes:
            for interval in self.sampling_intervals:
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

                    print(f"🏛️ Test {current_test}: Queue={queue_size}, Interval={interval}, Stack={stack_depth}")

                    # Run test
                    test_start = time.time()
                    result = self.run_renaissance_test(queue_size, interval, stack_depth)
                    test_time = time.time() - test_start

                    if result and result.get('success', False) and result.get('loss_percentage') is not None:
                        print(f"    ✅ Success: {result['loss_percentage']:.2f}% loss rate (took {test_time:.1f}s)")
                        result['test_number'] = current_test
                        result['test_duration_actual'] = test_time
                        self.update_csv_realtime('renaissance', result)

                        # Generate real-time plots after every successful test
                        self.plot_realtime_progress('renaissance')
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
        print(f"   Results saved to {DATA_DIR}")

        # Generate final comprehensive real-time plots
        print(f"   📊 Generating final real-time plots...")
        self.plot_realtime_progress('renaissance')

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

                            # Plot the points
                            ax.plot(subset['queue_size'], subset['loss_percentage'],
                                   marker='o', linestyle='None', markersize=4, label=label)

                elif test_type == 'renaissance':
                    # Plot points for each stack depth (no native duration in Renaissance)
                    for stack_depth in sorted(interval_data['stack_depth'].unique()):
                        subset = interval_data[interval_data['stack_depth'] == stack_depth]

                        if len(subset) < 1:
                            continue

                        # Sort by queue size for proper plotting
                        subset = subset.sort_values('queue_size')

                        # Create label for this combination
                        label = f"Stack {stack_depth}"

                        # Plot the points
                        ax.plot(subset['queue_size'], subset['loss_percentage'],
                               marker='o', linestyle='None', markersize=4, label=label)

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
            plt.subplots_adjust(top=0.92)  # Move title higher to avoid overlap
            plt.savefig(plot_path, dpi=600, bbox_inches='tight')
            plt.close(fig)

            print(f"    📊 Saved combined real-time plot: {plot_filename}")
            print(f"    📍 Absolute path: {plot_path.absolute()}")

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

                            # Plot the points
                            ax.plot(subset['queue_size'], subset['loss_percentage'],
                                   marker='o', linestyle='None', markersize=4, label=label)

                elif test_type == 'renaissance':
                    # Plot points for each stack depth (no native duration in Renaissance)
                    for stack_depth in sorted(interval_data['stack_depth'].unique()):
                        subset = interval_data[interval_data['stack_depth'] == stack_depth]

                        if len(subset) < 1:
                            continue

                        # Sort by queue size for proper plotting
                        subset = subset.sort_values('queue_size')

                        # Create label for this combination
                        label = f"Stack {stack_depth}"

                        # Plot the points
                        ax.plot(subset['queue_size'], subset['loss_percentage'],
                               marker='o', linestyle='None', markersize=4, label=label)

                # Add the psychologically important 1% reference line
                ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1, alpha=0.7, label='1% threshold')

                # Customize the subplot
                ax.set_xlabel('Queue Size (log scale)', fontsize=10)
                ax.set_ylabel('Loss % (log scale)', fontsize=10)
                ax.set_title(f'{interval} ({len(interval_data)} tests)', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Set unified y-range and log scale
                ax.set_ylim(y_min, y_max)
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
            plt.subplots_adjust(top=0.92)  # Move title higher to avoid overlap
            plt.savefig(log_plot_path, dpi=600, bbox_inches='tight')
            plt.close(fig)

            print(f"    📊 Saved log-scale combined plot: {log_plot_filename}")
            print(f"    📍 Absolute path: {log_plot_path.absolute()}")

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

                            # Plot the points
                            ax.plot(subset['queue_size'], subset['loss_percentage'],
                                   marker='o', linestyle='None', markersize=8, label=label)

                elif test_type == 'renaissance':
                    # Plot points for each stack depth (no native duration in Renaissance)
                    for stack_depth in sorted(interval_data['stack_depth'].unique()):
                        subset = interval_data[interval_data['stack_depth'] == stack_depth]

                        if len(subset) < 1:
                            continue

                        # Sort by queue size for proper plotting
                        subset = subset.sort_values('queue_size')

                        # Create label for this combination
                        label = f"Stack Depth {stack_depth}"

                        # Plot the points
                        ax.plot(subset['queue_size'], subset['loss_percentage'],
                               marker='o', linestyle='None', markersize=8, label=label)

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
                plt.savefig(plot_path, dpi=600, bbox_inches='tight')
                plt.close(fig)

                self.vprint(f"    📊 Saved individual plot: {plot_filename}")

            print(f"    📊 Saved {len(intervals)} individual interval plots to: individual/")
            print(f"    📍 Directory: {individual_plots_dir.absolute()}")

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

            # Define colors and markers for different intervals
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            markers = ['o', 's', '^', 'D', 'x']

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

                # Plot with distinct color and marker
                color_idx = i % len(colors)
                marker_idx = i % len(markers)

                ax.plot(interval_data['queue_size'], interval_data['loss_percentage'],
                       marker=markers[marker_idx], linestyle='None', markersize=10,
                       color=colors[color_idx], label=f'{interval}')

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
                    ax.set_ylim(y_min, y_max)

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
            plt.savefig(plot_path, dpi=600, bbox_inches='tight')
            plt.close(fig)

            print(f"    📊 Saved Renaissance combined intervals plot ({filename_suffix}): {plot_filename}")
            print(f"    📍 Absolute path: {plot_path.absolute()}")

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

        if renaissance_df is not None:
            self.plot_renaissance_results(renaissance_df)

        if native_df is not None and renaissance_df is not None:
            self.plot_comparison(native_df, renaissance_df)

        print(f"📊 Visualizations saved to {PLOTS_DIR}")

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
        else:
            self.plot_renaissance_results(df, progress_mode=True)

        print(f"📊 Progress visualizations saved to {PLOTS_DIR} with '_progress' suffix")

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
        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_heatmaps_by_duration_stack.png', progress_mode), dpi=600, bbox_inches='tight')
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
        plt.savefig(PLOTS_DIR / self._get_plot_filename('native_loss_vs_queue_size_all_combinations.png', progress_mode), dpi=600, bbox_inches='tight')
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
            plt.savefig(PLOTS_DIR / self._get_plot_filename(f'native_loss_vs_queue_size_stack{stack_depth}.png', progress_mode), dpi=600, bbox_inches='tight')
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

        # 1. Create separate heatmaps for each stack depth
        for stack_depth in df_success['stack_depth'].unique():
            df_stack = df_success[df_success['stack_depth'] == stack_depth]

            fig, ax = plt.subplots(figsize=(10, 8))

            pivot = df_stack.pivot(index='queue_size', columns='interval', values='loss_percentage')
            pivot = pivot.reindex(columns=interval_order)

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       ax=ax, cbar_kws={'label': 'Loss Rate (%)'})
            ax.set_title(f'Renaissance Test: Loss Rate Heatmap (Stack Depth: {stack_depth})', fontsize=14)
            ax.set_xlabel('Sampling Interval')
            ax.set_ylabel('Queue Size')

            # Prevent scientific notation on heatmap axes
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}' if y % 1 == 0 else f'{y:,.1f}'))

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / self._get_plot_filename(f'renaissance_heatmap_stack{stack_depth}.png', progress_mode), dpi=600, bbox_inches='tight')
            plt.close()

        # 2. Line plots: Loss Rate vs Queue Size for each Interval, showing all stack depths
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('Renaissance Test: Loss Rate vs Queue Size by Interval (All Stack Depths)', fontsize=16)

        for i, interval in enumerate(interval_order):
            df_int = df_success[df_success['interval'] == interval]

            for stack_depth in df_success['stack_depth'].unique():
                df_stack = df_int[df_int['stack_depth'] == stack_depth]
                if not df_stack.empty:
                    axes[i].plot(df_stack['queue_size'], df_stack['loss_percentage'],
                               marker='o', linestyle='None', markersize=8, label=f'Stack {stack_depth}')

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
        plt.savefig(PLOTS_DIR / self._get_plot_filename('renaissance_loss_vs_queue_size_all_stacks.png', progress_mode), dpi=600, bbox_inches='tight')
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
        plt.savefig(PLOTS_DIR / 'comparison_native_vs_renaissance.png', dpi=600, bbox_inches='tight')
        plt.close()

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

    args = parser.parse_args()

    # If minimal flag is used without other flags, run only native (not renaissance)
    if args.minimal and not any([args.run_native, args.run_renaissance, args.visualize, args.visualize_progress, args.all, args.estimate, args.only_native, args.only_renaissance]):
        args.run_native = True

    if not any([args.run_native, args.run_renaissance, args.visualize, args.visualize_progress, args.all, args.estimate, args.minimal, args.only_native, args.only_renaissance]):
        parser.print_help()
        return

    # Create benchmark runner with minimal configuration if requested
    runner = BenchmarkRunner(minimal=args.minimal, threads=args.threads, max_retries=args.retries, verbose=args.verbose)

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
    if args.visualize_progress:
        runner.create_progress_visualizations(args.csv_file)
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
