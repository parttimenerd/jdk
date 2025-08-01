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

Runtime Estimates:
    Native benchmark: ~26 hours (108 tests × 4.5 minutes avg per test)
    Renaissance benchmark: ~2.4 hours (36 tests × 4 minutes avg per test)
    Combined: ~28.4 hours total
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import analysis functions from analyze_drain_categories
try:
    from analyze_drain_categories import extract_drain_stats_by_label, DRAIN_CATEGORIES, calculate_event_weighted_percentiles
    ANALYSIS_AVAILABLE = True
    print("✅ Successfully imported drain analysis functions")
except ImportError as e:
    ANALYSIS_AVAILABLE = False
    print(f"⚠️ Could not import drain analysis functions: {e}")
    print("   Falling back to basic parsing methods")

# Configuration
QUEUE_SIZES = [50, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000]
SAMPLING_INTERVALS = ["1ms", "2ms", "5ms", "10ms", "20ms"]
NATIVE_DURATIONS = [5, 10, 20, 60, 250]  # seconds
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
    def __init__(self, minimal=False):
        self.setup_directories()
        self.results = {
            'native': [],
            'renaissance': []
        }
        self.minimal = minimal

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

    def estimate_runtime(self, test_type: str) -> Tuple[int, str]:
        """Estimate total runtime for benchmark suite"""
        if test_type == 'native':
            # Native test time = test_duration + setup/teardown (estimated 30s per test)
            single_test_time = self.test_duration + 30
            total_tests = len(self.queue_sizes) * len(self.sampling_intervals) * len(self.native_durations) * len(self.stack_depths)
            total_seconds = total_tests * single_test_time
        elif test_type == 'renaissance':
            # Renaissance tests take 3:30 minutes (210s) + setup/teardown (estimated 30s per test)
            single_test_time = 210 + 30  # 4 minutes total per test
            total_tests = len(self.queue_sizes) * len(self.sampling_intervals) * len(self.stack_depths)
            total_seconds = total_tests * single_test_time * self.renaissance_iterations
        else:
            return 0, "Unknown test type"

        # Convert to human readable format
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60

        if hours > 0:
            time_str = f"{hours}h {minutes}m {remaining_seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {remaining_seconds}s"
        else:
            time_str = f"{remaining_seconds}s"

        return total_seconds, time_str

    def setup_directories(self):
        """Create necessary directories for results"""
        for dir_path in [RESULTS_DIR, LOGS_DIR, DATA_DIR, PLOTS_DIR]:
            dir_path.mkdir(exist_ok=True)

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
            print(f"    💾 Updated {latest_csv.name} with {len(self.results[test_type])} results")

    def run_native_test(self, queue_size: int, interval: str, stack_depth: int, native_duration: Optional[int] = None) -> Dict:
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
            "-t", str(THREADS),
            "-i", interval,
            "-q", str(queue_size),
            "--no-analysis",  # Enable analysis tables but skip plots/visualizations
            "-f", str(log_path.absolute()) + ".raw",
        ]

        if native_duration:
            cmd.extend(["--native-duration", str(native_duration)])

        # Generate log filename

        print(f"    Command: {' '.join(cmd)}")
        print(f"    Log file: {log_path}")

        # Run test
        try:
            with open(log_path, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    cwd="../run_in_native",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            print(f"    Return code: {result.returncode}")

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
                print(f"    Extracted loss percentage: {loss_percentage}")
                if out_of_thread_percentage is not None:
                    print(f"    Extracted out-of-thread: {out_of_thread_events:,} events ({out_of_thread_percentage:.6f}%)")
                if all_without_locks_events is not None:
                    print(f"    Extracted all-without-locks: {all_without_locks_events:,} events")
            else:
                # Fallback for old format
                loss_percentage = extracted_data
                out_of_thread_events = None
                out_of_thread_percentage = None
                out_of_thread_details = None
                all_without_locks_events = None
                all_without_locks_details = None
                print(f"    Extracted loss percentage: {loss_percentage}")

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
                print(f"    Extracted loss percentage: {loss_percentage}")
                if out_of_thread_percentage is not None:
                    print(f"    Extracted out-of-thread: {out_of_thread_events:,} events ({out_of_thread_percentage:.6f}%)")
                if all_without_locks_events is not None:
                    print(f"    Extracted all-without-locks: {all_without_locks_events:,} events")
            else:
                loss_percentage = extracted_data
                out_of_thread_events = None
                out_of_thread_percentage = None
                out_of_thread_details = None
                all_without_locks_events = None
                all_without_locks_details = None
                print(f"    Extracted loss percentage: {loss_percentage}")

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
                print("    ⚠️ Could not find 'QUEUE SIZE ANALYSIS - Event-Weighted Percentiles' section")
                return {}

            analysis_text = section_match.group(1)
            print(f"    📊 Found queue size analysis section ({len(analysis_text)} chars)")

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
                print(f"    📊 Extracted percentiles for categories: {list(queue_stats.keys())}")
                for category, stats in queue_stats.items():
                    percentiles = [k for k in stats.keys() if k.startswith('p')]
                    events = stats.get('total_events', 'N/A')
                    print(f"      {category}: {len(percentiles)} percentiles, {events:,} total events")

            return queue_stats

        except Exception as e:
            print(f"    ⚠️ Error extracting queue size percentiles: {e}")
            return {}

    def parse_drain_stats_with_analysis_script(self, log_file_path: Path):
        """Parse drain statistics using the analysis script's parsing logic"""
        if not ANALYSIS_AVAILABLE:
            print("    ⚠️ Analysis script functions not available, falling back to basic parsing")
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

        try:
            print(f"    🔍 Using analysis script to parse: {log_file_path}")

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
                                print(f"    ⚠️ Error parsing JSON: {e}")
                                continue

            if not all_stats:
                print(f"    ⚠️ No DRAIN_STATS_JSON found in {log_file_path}")
                return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

            print(f"    📊 Found {len(all_stats)} drain statistics entries")

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
                            'p99_9': events_data.get('99.9th', events_data.get('p99.9', events_data.get('p99_9', 0)))
                        },
                        'time_ns': {
                            'sum': time_data.get('sum', 0),
                            'avg': time_data.get('average', time_data.get('avg', 0)),
                            'min': time_data.get('min', 0),
                            'max': time_data.get('max', 0),
                            'median': time_data.get('50th', time_data.get('median', 0)),
                            'p95': time_data.get('95th', time_data.get('p95', 0)),
                            'p99': time_data.get('99th', time_data.get('p99', 0)),
                            'p99_9': time_data.get('99.9th', time_data.get('p99.9', time_data.get('p99_9', 0)))
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
                        print(f"    📊 {category_name}: {events_sum:,} events (counted in without-locks total)")
                    else:
                        print(f"    📊 {category_name}: {events_sum:,} events (excluded from without-locks total)")

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
                    print(f"    📊 JFR Loss: {lost:,}/{total + lost:,} = {loss_percentage:.2f}%")

            return result

        except Exception as e:
            print(f"    ⚠️ Error parsing with analysis script: {e}")
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

    def parse_raw_drain_stats(self, raw_file_path: Path):
        """Parse comprehensive drain statistics from .raw file"""
        try:
            with open(raw_file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            print(f"    📊 Parsing drain statistics from .raw file ({len(content):,} chars)")

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
                print(f"    📋 Processing drain category: {category_name}")

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
                        print(f"      {category_name}: {events_sum:,} events (counted in without-locks total)")
                    else:
                        print(f"      {category_name}: {events_sum:,} events (excluded from without-locks total)")

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
                        print(f"    📊 JFR Loss: {lost:,}/{total + lost:,} = {loss_percentage:.2f}%")

            return result

        except Exception as e:
            print(f"    ⚠️ Error parsing .raw drain statistics: {e}")
            return {'loss_percentage': None, 'out_of_thread_events': None, 'out_of_thread_percentage': None}

    def extract_loss_percentage(self, log_path: Path):
        """Extract loss percentage, out-of-thread statistics, and queue size statistics from log file"""
        try:
            # First try to read the corresponding .raw file for more detailed drain statistics
            raw_file_path = Path(str(log_path) + ".raw")
            if raw_file_path.exists():
                print(f"    🔍 Found .raw file: {raw_file_path}")

                # Prioritize using the analysis script on the .raw file since it contains DRAIN_STATS_JSON
                if ANALYSIS_AVAILABLE:
                    print(f"    🔍 Using analysis script on .raw file: {raw_file_path}")
                    result = self.parse_drain_stats_with_analysis_script(raw_file_path)
                    if result.get('out_of_thread_events') is not None or result.get('loss_percentage') is not None:
                        return result
                    print(f"    ⚠️ Analysis script didn't find data in .raw file, falling back to raw parsing...")

                return self.parse_raw_drain_stats(raw_file_path)

            # If no .raw file, try using the analysis script on the main log file
            if ANALYSIS_AVAILABLE:
                print(f"    🔍 No .raw file found, trying analysis script on main log: {log_path}")
                result = self.parse_drain_stats_with_analysis_script(log_path)
                if result.get('out_of_thread_events') is not None or result.get('loss_percentage') is not None:
                    return result
                print(f"    ⚠️ Analysis script didn't find data in main log, falling back to basic parsing...")

            # Fallback to parsing the main log file
            # Try different encodings to handle special characters
            content = None
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(log_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    print(f"    📝 Successfully read log with {encoding} encoding: {log_path}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                # Fallback: read as binary and decode with error replacement
                with open(log_path, 'rb') as f:
                    raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace')
                print(f"    📝 Read log with fallback binary mode: {log_path}")

            print(f"    📏 Log size: {len(content):,} characters")

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
                print(f"    📊 Found out-of-thread section ({len(section_content)} chars)")

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

                    print(f"    📊 Parsed out-of-thread details:")
                    print(f"       Requests: {out_of_thread_details.get('requests', 'N/A')}")
                    print(f"       Runtime: {out_of_thread_details.get('runtime_seconds', 'N/A')}s ({out_of_thread_details.get('runtime_minutes', 'N/A')} min)")
                    print(f"       Request Rate: {out_of_thread_details.get('request_rate', 'N/A')} req/s")
                    print(f"       Events: {out_of_thread_events:,}")
                    if 'time_ns' in out_of_thread_details:
                        time_stats = out_of_thread_details['time_ns']
                        print(f"       Time (ns): avg={time_stats['avg']:,}, median={time_stats['median']:,}, p95={time_stats['p95']:,}")

                result['out_of_thread_details'] = out_of_thread_details
            else:
                print(f"    ⚠️ Could not find 'out of thread Drain Statistics' section")

            # Extract "all without locks" statistics from Drain Statistics sections
            # Look for the complete "=== all without locks Drain Statistics ===" section
            all_without_locks_section_pattern = r'=== all without locks Drain Statistics ===\s*(.*?)(?=\n===|\nAnalyzing|\nDRAIN_STATS_JSON|\n\[|\Z)'
            all_without_locks_match = re.search(all_without_locks_section_pattern, content, re.DOTALL)

            if all_without_locks_match:
                section_content = all_without_locks_match.group(1)
                print(f"    📊 Found all-without-locks section ({len(section_content)} chars)")

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

                    print(f"    📊 Parsed all-without-locks details:")
                    print(f"       Requests: {all_without_locks_details.get('requests', 'N/A'):,}")
                    print(f"       Runtime: {all_without_locks_details.get('runtime_seconds', 'N/A')}s ({all_without_locks_details.get('runtime_minutes', 'N/A')} min)")
                    print(f"       Request Rate: {all_without_locks_details.get('request_rate', 'N/A'):,} req/s")
                    print(f"       Events: {all_without_locks_events:,}")
                    if 'time_ns' in all_without_locks_details:
                        time_stats = all_without_locks_details['time_ns']
                        print(f"       Time (ns): sum={time_stats['sum']:,}, avg={time_stats['avg']:,}, median={time_stats['median']:,}")

                result['all_without_locks_details'] = all_without_locks_details
            else:
                print(f"    ⚠️ Could not find 'all without locks Drain Statistics' section")

            # Calculate out-of-thread percentage using all drain statistics sections
            all_events_pattern = r'Events:\s*sum=(\d+)'
            all_events_matches = re.findall(all_events_pattern, content)

            if len(all_events_matches) >= 1:
                total_events = sum(int(match) for match in all_events_matches)
                print(f"    📊 Found {len(all_events_matches)} drain statistics sections with total events: {total_events:,}")

                if result.get('out_of_thread_events') and total_events > 0:
                    out_of_thread_percentage = (result['out_of_thread_events'] / total_events) * 100
                    result['out_of_thread_percentage'] = out_of_thread_percentage
                    print(f"    📊 Calculated out-of-thread percentage: {out_of_thread_percentage:.6f}% ({result['out_of_thread_events']:,}/{total_events:,})")
                elif total_events == 0:
                    print(f"    ⚠️ Total events is 0, cannot calculate percentage")
            else:
                print(f"    ⚠️ Could not find any drain statistics sections with events")            # First try direct search for all statistics in entire content
            print(f"    🔍 Searching entire log for JFR statistics...")
            successful_match = re.search(r'Successful Samples:\s*([\d,]+)', content)
            total_match = re.search(r'Total Samples:\s*([\d,]+)', content)
            lost_match = re.search(r'Lost Samples:\s*([\d,]+)', content)

            print(f"    Direct search results:")
            print(f"       Successful: {'✅' if successful_match else '❌'} {successful_match.group(1) if successful_match else 'Not found'}")
            print(f"       Total: {'✅' if total_match else '❌'} {total_match.group(1) if total_match else 'Not found'}")
            print(f"       Lost: {'✅' if lost_match else '❌'} {lost_match.group(1) if lost_match else 'Not found'}")

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

                    # Return loss percentage and out-of-thread data
                    result['loss_percentage'] = loss_percentage
                    return result
                else:
                    print(f"    ⚠️ Total attempted samples is 0")
                    result['loss_percentage'] = 0.0
                    return result

            # If direct search didn't work, try the original section-based approach

            # Look for the loss rate in the final summary
            loss_pattern = r'Loss Rate:\s+([\d.]+)%'
            match = re.search(loss_pattern, content)

            if match:
                loss_rate = float(match.group(1))
                print(f"    Found loss rate pattern: {loss_rate}%")
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
                    else:
                        print(f"    ❌ Failed or no data (took {test_time:.1f}s)")
                        if result:
                            result['test_number'] = current_test
                            result['test_duration_actual'] = test_time
                            self.update_csv_realtime('renaissance', result)

                    # Small delay between tests
                    time.sleep(2)

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

    def plot_native_results(self, df: pd.DataFrame):
        """Create native test visualizations"""
        # Filter successful tests
        df_success = df[df['success'] == True].copy()

        if df_success.empty:
            print("No successful native tests to visualize")
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
            else:
                axes[i].set_title(f'Duration: {native_dur}s, Stack: {stack_depth} (No Data)')
                axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'native_heatmaps_by_duration_stack.png', dpi=300, bbox_inches='tight')
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
                               marker='o', label=f'{native_dur}s/S{stack_depth}', linewidth=2)

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'native_loss_vs_queue_size_all_combinations.png', dpi=300, bbox_inches='tight')
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
                                   marker='o', label=f'{native_dur}s native', linewidth=2)

                axes[i].set_title(f'Interval: {interval}')
                axes[i].set_xlabel('Queue Size')
                axes[i].set_ylabel('Loss Rate (%)')

                # Only add legend if there are actually lines plotted
                lines, labels = axes[i].get_legend_handles_labels()
                if lines:
                    axes[i].legend()

                axes[i].grid(True, alpha=0.3)
                axes[i].set_xscale('log')

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f'native_loss_vs_queue_size_stack{stack_depth}.png', dpi=300, bbox_inches='tight')
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

        plt.savefig(PLOTS_DIR / 'native_3d_surface.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_renaissance_results(self, df: pd.DataFrame):
        """Create Renaissance test visualizations"""
        # Filter successful tests
        df_success = df[df['success'] == True].copy()

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

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f'renaissance_heatmap_stack{stack_depth}.png', dpi=300, bbox_inches='tight')
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
                               marker='o', linewidth=2, markersize=8, label=f'Stack {stack_depth}')

            axes[i].set_title(f'Interval: {interval}')
            axes[i].set_xlabel('Queue Size')
            axes[i].set_ylabel('Loss Rate (%)')

            # Only add legend if there are actually lines plotted
            lines, labels = axes[i].get_legend_handles_labels()
            if lines:
                axes[i].legend()

            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'renaissance_loss_vs_queue_size_all_stacks.png', dpi=300, bbox_inches='tight')
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

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'comparison_native_vs_renaissance.png', dpi=300, bbox_inches='tight')
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
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks and generate visualizations')
    parser.add_argument('--estimate', action='store_true',
                       help='Show runtime estimates without running benchmarks')
    parser.add_argument('--only-native', action='store_true',
                       help='Run only native benchmarks (skip Renaissance)')
    parser.add_argument('--only-renaissance', action='store_true',
                       help='Run only Renaissance benchmarks (skip native)')

    args = parser.parse_args()

    # If minimal flag is used without other flags, run only native (not renaissance)
    if args.minimal and not any([args.run_native, args.run_renaissance, args.visualize, args.all, args.estimate, args.only_native, args.only_renaissance]):
        args.run_native = True

    if not any([args.run_native, args.run_renaissance, args.visualize, args.all, args.estimate, args.minimal, args.only_native, args.only_renaissance]):
        parser.print_help()
        return

    # Create benchmark runner with minimal configuration if requested
    runner = BenchmarkRunner(minimal=args.minimal)

    # Show configuration info
    print(f"JFR Queue Size Benchmark Suite")
    print(f"{'='*50}")
    if args.minimal:
        print(f"Mode: Minimal (quick validation)")
        print(f"Queue sizes: {runner.queue_sizes}")
        print(f"Sampling intervals: {runner.sampling_intervals}")
        print(f"Native durations: {runner.native_durations}")
        print(f"Test duration: {runner.test_duration}s")
    else:
        print(f"Mode: Full (comprehensive analysis)")
        print(f"Queue sizes: {len(runner.queue_sizes)} values from {min(runner.queue_sizes)} to {max(runner.queue_sizes)}")
        print(f"Sampling intervals: {len(runner.sampling_intervals)} values from {runner.sampling_intervals[0]} to {runner.sampling_intervals[-1]}")
        print(f"Test duration: {runner.test_duration}s")
    print(f"{'='*50}")

    # Show estimates if requested
    if args.estimate:
        print("⏱️ Benchmark Runtime Estimates")

        # Show native estimates
        native_seconds, native_time = runner.estimate_runtime('native')
        native_tests = len(runner.queue_sizes) * len(runner.sampling_intervals) * len(runner.native_durations) * len(runner.stack_depths)
        print(f"🔧 Native Benchmark:")
        print(f"   Tests: {native_tests}")
        print(f"   Estimated time: {native_time}")
        print(f"   Expected completion: {datetime.fromtimestamp(time.time() + native_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

        # Only show Renaissance estimates if not in minimal mode and not only-native
        if not args.minimal and not args.only_native:
            renaissance_seconds, renaissance_time = runner.estimate_runtime('renaissance')
            renaissance_tests = len(runner.queue_sizes) * len(runner.sampling_intervals) * len(runner.stack_depths)
            print(f"🏛️ Renaissance Benchmark:")
            print(f"   Tests: {renaissance_tests}")
            print(f"   Estimated time: {renaissance_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + renaissance_seconds).strftime('%Y-%m-%d %H:%M:%S')}")

            total_seconds = native_seconds + renaissance_seconds
            total_tests = native_tests + renaissance_tests
        else:
            # In minimal mode, only-native mode, or only-renaissance mode, only show native totals
            total_seconds = native_seconds
            total_tests = native_tests

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

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        if hours > 0:
            total_time = f"{hours}h {minutes}m {remaining_seconds}s"
        elif minutes > 0:
            total_time = f"{minutes}m {remaining_seconds}s"
        else:
            total_time = f"{remaining_seconds}s"

        if not args.minimal and not args.only_native and not args.only_renaissance:
            print(f"🎯 Combined Total:")
            print(f"   Tests: {total_tests}")
            print(f"   Estimated time: {total_time}")
            print(f"   Expected completion: {datetime.fromtimestamp(time.time() + total_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
        elif args.only_renaissance:
            print(f"🎯 Total (Renaissance Only):")
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
