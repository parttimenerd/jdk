#!/usr/bin/env python3
"""
Test script to verify queue size percentile extraction from .raw files
"""

import json
import sys
from pathlib import Path

# Import our parsing function
from benchmark_queue_sizes import BenchmarkRunner

def test_queue_percentiles(raw_file_path):
    """Test queue size percentile extraction"""
    print(f"Testing queue size percentile extraction from: {raw_file_path}")

    runner = BenchmarkRunner()
    result = runner.extract_loss_percentage(Path(raw_file_path))

    print(f"\n📊 Extraction Results:")
    print(f"Loss percentage: {result.get('loss_percentage')}")
    print(f"Out-of-thread events: {result.get('out_of_thread_events')}")
    print(f"All-without-locks events: {result.get('all_without_locks_events')}")

    # Check for queue size percentiles in out_of_thread_details
    out_details = result.get('out_of_thread_details')
    if out_details and out_details.get('queue_size_percentiles'):
        percs = out_details['queue_size_percentiles']
        print(f"\n🏗️ Out-of-thread queue size percentiles:")
        print(f"  Median: {percs.get('median')}")
        print(f"  P95: {percs.get('p95')}")
        print(f"  P99: {percs.get('p99')}")
        print(f"  P99.9: {percs.get('p99_9')}")
        print(f"  P99.99: {percs.get('p99_99')}")
        print(f"  P99.999: {percs.get('p99_999')}")
    else:
        print(f"\n❌ No queue size percentiles found in out_of_thread_details")

    # Check for queue size percentiles in all_without_locks_details
    all_details = result.get('all_without_locks_details')
    if all_details:
        print(f"\n🏗️ All-without-locks details found:")
        print(f"  Requests: {all_details.get('requests', 'N/A')}")
        print(f"  Events: {all_details.get('events', {}).get('sum', 'N/A')}")
        if all_details.get('queue_size_percentiles'):
            percs = all_details['queue_size_percentiles']
            print(f"  Queue size percentiles:")
            print(f"    Median: {percs.get('median')}")
            print(f"    P95: {percs.get('p95')}")
            print(f"    P99: {percs.get('p99')}")
            print(f"    P99.9: {percs.get('p99_9')}")
            print(f"    P99.99: {percs.get('p99_99')}")
            print(f"    P99.999: {percs.get('p99_999')}")
        else:
            print(f"  ❌ No queue size percentiles in all_without_locks_details")
    else:
        print(f"\n❌ No all_without_locks_details found")

    # Check for queue size percentiles in drain_categories
    drain_cats = result.get('drain_categories', {})
    print(f"\n🏗️ Drain categories with queue size percentiles:")
    if drain_cats:
        for cat_name, cat_data in drain_cats.items():
            requests = cat_data.get('requests', 0)
            events_sum = cat_data.get('events', {}).get('sum', 0) if cat_data.get('events') else 0
            print(f"  {cat_name}:")
            print(f"    Requests: {requests}, Events: {events_sum}")
            if cat_data.get('queue_size_percentiles'):
                percs = cat_data['queue_size_percentiles']
                print(f"    P99: {percs.get('p99')}, P99.9: {percs.get('p99_9')}")
            else:
                print(f"    ❌ No queue size percentiles")
    else:
        print(f"  ❌ No drain categories found")

    return result

if __name__ == "__main__":
    # Test with Renaissance file that shows 0 requests
    raw_file = "benchmark_results/logs/renaissance_q100_1ms_s100_n1.log.raw"
    test_queue_percentiles(raw_file)
