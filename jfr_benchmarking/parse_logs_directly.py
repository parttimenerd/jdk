#!/usr/bin/env python3
"""
Direct log parser for DRAIN_STATS_JSON data
This bypasses the CSV/JSON storage issue by parsing directly from log files
"""

import json
import re
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

def parse_drain_stats_from_log(log_file_path: Path) -> Dict[str, Any]:
    """Parse drain statistics directly from log file"""

    drain_categories = {}
    max_queue_sizes = []  # Store all MAX_QUEUE_SIZE_SUM values
    queue_size_increase_counts = []  # Store all QUEUE_SIZE_INCREASE_COUNT values
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # Find all DRAIN_STATS_JSON entries using simple line-by-line parsing
        lines = content.split('\n')
        json_entries = []

        for line in lines:
            if 'DRAIN_STATS_JSON:' in line:
                # Extract JSON part after the colon
                json_start = line.find('DRAIN_STATS_JSON:') + len('DRAIN_STATS_JSON:')
                json_str = line[json_start:].strip()

                if json_str:
                    json_entries.append(json_str)

            # Also parse MAX_QUEUE_SIZE_SUM values
            elif 'MAX_QUEUE_SIZE_SUM:' in line:
                # Extract the numeric value after the colon
                match = re.search(r'MAX_QUEUE_SIZE_SUM:\s*(\d+)', line)
                if match:
                    max_queue_sizes.append(int(match.group(1)))

            # Also parse QUEUE_SIZE_INCREASE_COUNT values
            elif 'QUEUE_SIZE_INCREASE_COUNT:' in line:
                # Extract the numeric value after the colon
                match = re.search(r'QUEUE_SIZE_INCREASE_COUNT:\s*(\d+)', line)
                if match:
                    queue_size_increase_counts.append(int(match.group(1)))

        matches = json_entries

        for match in matches:
            try:
                drain_data = json.loads(match)
                category_name = drain_data.get('name', 'unknown')
                drain_categories[category_name] = drain_data

            except json.JSONDecodeError as e:
                continue

    except Exception as e:
        print(f"   ❌ Error reading log file: {e}")

    # Add the max queue sizes and increase counts to the result
    result = {
        'drain_categories': drain_categories,
        'max_queue_sizes': max_queue_sizes,
        'final_max_queue_size': max_queue_sizes[-1] if max_queue_sizes else 0,
        'queue_size_increase_counts': queue_size_increase_counts,
        'final_queue_size_increase_count': queue_size_increase_counts[-1] if queue_size_increase_counts else 0
    }

    return result

def extract_test_params_from_filename(filename: str) -> Dict[str, Any]:
    """Extract test parameters from log filename"""
    # Examples:
    # native_q1000_10ms_s100_250s_native5s.log
    # renaissance_q20_5ms_n2.log

    params = {}

    if filename.startswith('native_'):
        # Parse native format: native_q{queue_size}_{interval}_s{stack_depth}_{test_duration}_native{native_duration}.log
        match = re.match(r'native_q(\d+)_(\d+ms)_s(\d+)_(\d+s)_native(\d+s)\.log', filename)
        if match:
            params = {
                'test_type': 'native',
                'queue_size': int(match.group(1)),
                'interval': match.group(2),
                'stack_depth': int(match.group(3)),
                'test_duration': match.group(4),
                'native_duration': match.group(5)
            }
    elif filename.startswith('renaissance_'):
        # Parse renaissance format: renaissance_q{queue_size}_{interval}_n{iterations}.log
        # Also handle: renaissance_q{queue_size}_{interval}_s{stack_depth}_n{iterations}.log
        match = re.match(r'renaissance_q(\d+)_(\d+ms)_n(\d+)\.log', filename)
        if match:
            params = {
                'test_type': 'renaissance',
                'queue_size': int(match.group(1)),
                'interval': match.group(2),
                'iterations': int(match.group(3))
            }
        else:
            # Try pattern with stack depth: renaissance_q{queue_size}_{interval}_s{stack_depth}_n{iterations}.log
            match = re.match(r'renaissance_q(\d+)_(\d+ms)_s(\d+)_n(\d+)\.log', filename)
            if match:
                params = {
                    'test_type': 'renaissance',
                    'queue_size': int(match.group(1)),
                    'interval': match.group(2),
                    'stack_depth': int(match.group(3)),  # Include stack depth when present
                    'iterations': int(match.group(4))
                }

    return params

def parse_all_logs(logs_dir: Path = Path("benchmark_results/logs")) -> List[Dict[str, Any]]:
    """Parse drain statistics from all log files"""
    print(f"📂 Scanning for log files in: {logs_dir}")

    log_files = list(logs_dir.glob("*.log"))

    results = []

    for log_file in log_files:
        # Extract test parameters from filename
        params = extract_test_params_from_filename(log_file.name)
        if not params:
            print(f"   ⚠️ Could not parse parameters from: {log_file.name}")
            continue

        # Parse drain statistics and queue size data
        parse_result = parse_drain_stats_from_log(log_file)
        drain_categories = parse_result.get('drain_categories', {})

        if drain_categories:
            # Create result entry
            result = {
                **params,
                'log_file': log_file.name,
                'drain_categories': drain_categories,
                'max_queue_sizes': parse_result.get('max_queue_sizes', []),
                'final_max_queue_size': parse_result.get('final_max_queue_size', 0),
                'queue_memory_consumption': parse_result.get('final_max_queue_size', 0) * 48,  # 48 bytes per entry
                'queue_size_increase_counts': parse_result.get('queue_size_increase_counts', []),
                'final_queue_size_increase_count': parse_result.get('final_queue_size_increase_count', 0)
            }
            results.append(result)
        else:
            print(f"   ⚠️ No drain stats found in: {log_file.name}")

    return results

def create_percentile_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create DataFrame with percentile data for plotting"""
    rows = []

    for result in results:
        base_row = {
            'test_type': result['test_type'],
            'queue_size': result['queue_size'],
            'interval': result['interval'],
            'log_file': result['log_file'],
            'final_max_queue_size': result.get('final_max_queue_size', 0),
            'queue_memory_consumption': result.get('queue_memory_consumption', 0),
            'final_queue_size_increase_count': result.get('final_queue_size_increase_count', 0)
        }

        # Add test-specific fields
        if result['test_type'] == 'native':
            base_row.update({
                'stack_depth': result.get('stack_depth'),
                'test_duration': result.get('test_duration'),
                'native_duration': result.get('native_duration')
            })
        elif result['test_type'] == 'renaissance':
            base_row.update({
                'iterations': result.get('iterations')
            })

        # Extract percentile data from each drain category
        drain_categories = result['drain_categories']

        for category_name, category_data in drain_categories.items():
            row = base_row.copy()
            row['drain_category'] = category_name

            # Always extract basic drain count (this is what we need for "out of thread")
            row['drains'] = category_data.get('drains', 0)
            row['runtime_seconds'] = category_data.get('runtime_seconds', 0)

            # Extract time percentiles (mainly meaningful for "all without locks")
            time_stats = category_data.get('time', {})
            for percentile in ['p95', 'p99', 'p99_9']:
                if percentile in time_stats:
                    row[f'time_{percentile}'] = time_stats[percentile]
                else:
                    # Set default values for missing percentiles
                    row[f'time_{percentile}'] = 0

            # Extract other time stats
            row['avg_time'] = time_stats.get('avg', 0)
            row['median_time'] = time_stats.get('median', 0)

            # Extract event stats
            event_stats = category_data.get('events', {})
            row['avg_events'] = event_stats.get('avg', 0)
            row['total_events'] = event_stats.get('sum', 0)

            rows.append(row)

    return pd.DataFrame(rows)

def summarize_drain_data(df: pd.DataFrame) -> None:
    """Summarize the key drain statistics we care about"""
    if df.empty:
        print("❌ No data to summarize")
        return

    print("\n📊 DRAIN STATISTICS SUMMARY")
    print("=" * 50)

    # Group by test configuration and drain category
    summary_stats = []

    for test_type in df['test_type'].unique():
        test_data = df[df['test_type'] == test_type]

        print(f"\n🔬 {test_type.upper()} TESTS:")

        for queue_size in sorted(test_data['queue_size'].unique()):
            queue_data = test_data[test_data['queue_size'] == queue_size]

            print(f"  📦 Queue Size {queue_size}:")

            for category in sorted(queue_data['drain_category'].unique()):
                cat_data = queue_data[queue_data['drain_category'] == category]

                if category == 'all without locks':
                    # For this category, show percentiles
                    avg_p99 = cat_data['time_p99'].mean() if 'time_p99' in cat_data.columns else 0
                    total_drains = cat_data['drains'].sum()
                    print(f"    🔒 {category}: {total_drains} drains, avg P99: {avg_p99:.1f}ns")

                elif category == 'out of thread':
                    # For this category, just show drain count (this is what you requested)
                    total_drains = cat_data['drains'].sum()
                    print(f"    🧵 {category}: {total_drains} drains")

                else:
                    # For other categories, show basic stats
                    total_drains = cat_data['drains'].sum()
                    print(f"    ⚙️  {category}: {total_drains} drains")

    print("\n📈 KEY INSIGHTS:")

    # Show "out of thread" drain counts by queue size
    out_of_thread_data = df[df['drain_category'] == 'out of thread']
    if not out_of_thread_data.empty:
        print("  🧵 'Out of thread' drains by queue size:")
        for queue_size in sorted(out_of_thread_data['queue_size'].unique()):
            queue_drains = out_of_thread_data[out_of_thread_data['queue_size'] == queue_size]['drains'].sum()
            print(f"     Queue {queue_size}: {queue_drains} drains")

    # Show "all without locks" P99 trends
    all_locks_data = df[df['drain_category'] == 'all without locks']
    if not all_locks_data.empty and 'time_p99' in all_locks_data.columns:
        print("  🔒 'All without locks' P99 latency by queue size:")
        for queue_size in sorted(all_locks_data['queue_size'].unique()):
            queue_p99 = all_locks_data[all_locks_data['queue_size'] == queue_size]['time_p99'].mean()
            print(f"     Queue {queue_size}: {queue_p99:.1f}ns P99")

def test_direct_parsing():
    """Test the direct parsing approach"""
    print("Testing direct log parsing...")

    # Parse all logs
    results = parse_all_logs()

    if not results:
        print("❌ No results found")
        return

    # Create DataFrame
    df = create_percentile_dataframe(results)
    print(f"\n📊 Created DataFrame with {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Show sample data
    print("\n📋 Sample data:")
    print(df.head())

    # Show drain categories found
    if 'drain_category' in df.columns:
        categories = df['drain_category'].unique()
        print(f"\n🏷️ Drain categories found: {list(categories)}")

    # Show test types and parameters
    if not df.empty:
        print(f"\n🔬 Test types: {df['test_type'].unique()}")
        print(f"   Queue sizes: {sorted(df['queue_size'].unique())}")
        print(f"   Intervals: {sorted(df['interval'].unique())}")

    # Add detailed summary
    summarize_drain_data(df)

    # Save results
    output_file = Path("direct_log_parsing_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved results to: {output_file}")

    return df

if __name__ == "__main__":
    df = test_direct_parsing()
