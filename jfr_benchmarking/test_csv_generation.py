#!/usr/bin/env python3
"""
Test CSV generation with queue size percentiles
"""

import pandas as pd
from pathlib import Path

# Import our benchmark runner
from benchmark_queue_sizes import BenchmarkRunner

def test_csv_generation():
    """Test that queue size percentiles are included in CSV output"""
    print("Testing CSV generation with queue size percentiles...")

    runner = BenchmarkRunner()

    # Create a mock result with queue size percentiles
    test_result = {
        'queue_size': 100,
        'interval': '10ms',
        'stack_depth': 100,
        'loss_percentage': 95.5,  # High loss percentage to test axis capping
        'success': True,  # Add success flag
        'out_of_thread_details': {
            'requests': 5,
            'request_rate': 0.5,
            'time_ns': {'avg': 1000, 'max': 2000},
            'events': {'sum': 10, 'avg': 2.0, 'min': 1, 'max': 5, 'median': 2, 'p95': 4, 'p99': 5, 'p99_9': 5},
            'queue_size_percentiles': {
                'median': 15,
                'p95': 45,
                'p99': 78,
                'p99_9': 95,
                'p99_99': 99,
                'p99_999': 100
            }
        },
        'all_without_locks_details': {
            'requests': 100,
            'request_rate': 10.0,
            'time_ns': {'avg': 5000, 'max': 10000},
            'events': {'sum': 200, 'avg': 2.0, 'min': 1, 'max': 8, 'median': 2, 'p95': 6, 'p99': 7, 'p99_9': 8},
            'queue_size_percentiles': {
                'median': 25,
                'p95': 55,
                'p99': 88,
                'p99_9': 96,
                'p99_99': 99,
                'p99_999': 100
            }
        },
        'drain_categories': {
            'all without locks': {
                'requests': 100,
                'runtime_seconds': 30.0,
                'request_rate': 3.33,
                'events': {'sum': 200, 'avg': 2.0, 'min': 1, 'max': 8, 'median': 2, 'p95': 6, 'p99': 7, 'p99_9': 8},
                'time_ns': {'sum': 500000, 'avg': 2500, 'min': 1000, 'max': 5000, 'median': 2000, 'p95': 4000, 'p99': 5000, 'p99_9': 5000},
                'queue_size_percentiles': {
                    'median': 25,
                    'p95': 55,
                    'p99': 88,
                    'p99_9': 96,
                    'p99_99': 99,
                    'p99_999': 100
                }
            },
            'safepoint': {
                'requests': 50,
                'runtime_seconds': 15.0,
                'request_rate': 3.33,
                'events': {'sum': 100, 'avg': 2.0, 'min': 1, 'max': 4, 'median': 2, 'p95': 3, 'p99': 4, 'p99_9': 4},
                'time_ns': {'sum': 250000, 'avg': 2500, 'min': 1000, 'max': 4000, 'median': 2000, 'p95': 3500, 'p99': 4000, 'p99_9': 4000},
                'queue_size_percentiles': {
                    'median': 12,
                    'p95': 28,
                    'p99': 42,
                    'p99_9': 48,
                    'p99_99': 49,
                    'p99_999': 50
                }
            }
        }
    }

    # Generate CSV data
    test_results = [test_result]

    # Add a few more test cases with varying loss percentages
    for queue_size, loss_pct in [(50, 0.5), (200, 5.2), (500, 25.8), (1000, 78.9)]:
        additional_result = test_result.copy()
        additional_result['queue_size'] = queue_size
        additional_result['loss_percentage'] = loss_pct
        test_results.append(additional_result)

    df = runner.flatten_for_csv(test_results)

    print(f"\n📊 Generated CSV with {len(df.columns)} columns")

    # Check for queue size percentile columns
    queue_columns = [col for col in df.columns if 'queue' in col.lower()]
    print(f"\n🏗️ Queue size percentile columns ({len(queue_columns)}):")
    for col in sorted(queue_columns):
        value = df[col].iloc[0] if len(df) > 0 else 'N/A'
        print(f"  {col}: {value}")

    # Check specific columns we expect
    expected_columns = [
        'out_of_thread_queue_p99',
        'out_of_thread_queue_p99_9',
        'all_without_locks_queue_p99',
        'all_without_locks_queue_p99_9',
        'drain_all_without_locks_queue_p99',
        'drain_all_without_locks_queue_p99_9',
        'drain_safepoint_queue_p99',
        'drain_safepoint_queue_p99_9'
    ]

    print(f"\n✅ Expected column verification:")
    for col in expected_columns:
        if col in df.columns:
            value = df[col].iloc[0] if len(df) > 0 else 'N/A'
            print(f"  ✅ {col}: {value}")
        else:
            print(f"  ❌ Missing: {col}")

    # Save to test CSV
    test_csv_path = Path("test_queue_percentiles.csv")
    df.to_csv(test_csv_path, index=False)
    print(f"\n💾 Test CSV saved to: {test_csv_path}")
    print(f"   Columns: {len(df.columns)}, Rows: {len(df)}")

    return df

if __name__ == "__main__":
    test_csv_generation()
