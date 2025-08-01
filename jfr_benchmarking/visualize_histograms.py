#!/usr/bin/env python3
"""
Visualize DrainStats histograms from JFR sampling output.
Usage: python3 visualize_histograms.py [input_file]
       cat log_file | python3 visualize_histograms.py
"""

import sys
import json
import re
import math
from typing import List, Dict, Any

def parse_drain_stats_json(line: str) -> Dict[str, Any]:
    """Parse a DRAIN_STATS_JSON line and return the parsed data."""
    if not line.startswith("DRAIN_STATS_JSON:"):
        return None

    json_str = line[len("DRAIN_STATS_JSON:"):]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def parse_jfr_statistics(lines: List[str]) -> Dict[str, Any]:
    """Parse JFR sample statistics from log lines."""
    jfr_stats = {}
    in_jfr_section = False
    jfr_lines = []

    for line in lines:
        line = line.strip()
        if line == "JFR_SAMPLE_STATISTICS:":
            in_jfr_section = True
            jfr_lines = []
            continue
        elif line == "JFR_SAMPLE_STATISTICS_END":
            in_jfr_section = False
            break
        elif in_jfr_section:
            jfr_lines.append(line)

    if not jfr_lines:
        return {}

    # Parse the JFR statistics
    for line in jfr_lines:
        if ":" in line:
            if "Successful Samples:" in line:
                jfr_stats["successful_samples"] = int(line.split(":")[1].strip().replace(",", ""))
            elif "Failed Samples:" in line:
                jfr_stats["failed_samples"] = int(line.split(":")[1].strip().replace(",", ""))
            elif "Biased Samples:" in line:
                jfr_stats["biased_samples"] = int(line.split(":")[1].strip().replace(",", ""))
            elif "Total Samples:" in line:
                jfr_stats["total_samples"] = int(line.split(":")[1].strip().replace(",", ""))
            elif "Lost Samples:" in line:
                jfr_stats["lost_samples"] = int(line.split(":")[1].strip().replace(",", ""))

    return jfr_stats

def visualize_time_histogram(data: Dict[str, Any], width: int = 80) -> str:
    """Visualize the time histogram."""
    histogram_data = data.get("time_histogram", [])
    if not histogram_data:
        return "No time histogram data"

    lines = []
    lines.append(f"Time Histogram ({len(histogram_data)} buckets):")
    lines.append("=" * width)

    # Find max count for scaling
    max_count = max((bucket.get("count", 0) for bucket in histogram_data), default=0)
    if max_count == 0:
        return "No time histogram data (all counts are zero)"

    # Display histogram buckets
    for bucket in histogram_data:
        count = bucket.get("count", 0)
        if count == 0:
            continue

        from_ns = bucket.get("from", 0)
        to_ns = bucket.get("to", 0)

        # Format time ranges nicely
        from_str = format_time_ns(from_ns)
        to_str = format_time_ns(to_ns)

        bar_len = int((count / max_count) * (width - 25))
        lines.append(f"{from_str}-{to_str}: {'█' * bar_len} {count}")

    return "\n".join(lines)

def visualize_event_histogram(data: Dict[str, Any], width: int = 80) -> str:
    """Visualize the event count histogram."""
    histogram_data = data.get("event_histogram", [])
    if not histogram_data:
        return "No event histogram data"

    lines = []
    lines.append(f"Event Count Histogram ({len(histogram_data)} buckets):")
    lines.append("=" * width)

    # Find max count for scaling
    max_count = max((bucket.get("count", 0) for bucket in histogram_data), default=0)
    if max_count == 0:
        return "No event histogram data (all counts are zero)"

    # Display histogram buckets
    for bucket in histogram_data:
        count = bucket.get("count", 0)
        if count == 0:
            continue

        from_events = bucket.get("from", 0)
        to_events = bucket.get("to", 0)

        # Format event range
        if from_events == to_events:
            range_str = f"{from_events} events"
        else:
            range_str = f"{from_events}-{to_events} events"

        bar_len = int((count / max_count) * (width - 20))
        lines.append(f"{range_str:>12}: {'█' * bar_len} {count}")

    return "\n".join(lines)

def format_time_ns(ns: int) -> str:
    """Format nanoseconds in a human-readable way."""
    if ns < 1000:
        return f"{ns}ns"
    elif ns < 1000000:
        return f"{ns // 1000}μs"
    elif ns < 1000000000:
        return f"{ns // 1000000}ms"
    else:
        return f"{ns // 1000000000}s"

def print_summary(data: Dict[str, Any]) -> None:
    """Print summary statistics."""
    name = data.get("name", "Unknown")
    drains = data.get("drains", 0)
    time_stats = data.get("time", {})
    event_stats = data.get("events", {})

    print(f"\n=== {name} Statistics ===")
    print(f"Total drain operations: {drains:,}")
    print(f"Time (ns): avg={time_stats.get('avg', 0):,}, min={time_stats.get('min', 0):,}, max={time_stats.get('max', 0):,}")
    print(f"Events: avg={event_stats.get('avg', 0):.2f}, min={event_stats.get('min', 0)}, max={event_stats.get('max', 0)}")

def print_jfr_statistics(jfr_stats: Dict[str, Any]) -> None:
    """Print JFR sample statistics in a formatted way."""
    if not jfr_stats:
        return

    print("\n" + "="*70)
    print("📊 JFR CPU TIME SAMPLE STATISTICS")
    print("="*70)

    # Display basic statistics
    successful = jfr_stats.get("successful_samples", 0)
    failed = jfr_stats.get("failed_samples", 0)
    biased = jfr_stats.get("biased_samples", 0)
    total = jfr_stats.get("total_samples", 0)
    lost = jfr_stats.get("lost_samples", 0)

    print(f"Successful Samples: {successful:,}")
    print(f"Failed Samples:     {failed:,}")
    print(f"Biased Samples:     {biased:,}")
    print(f"Total Samples:      {total:,}")
    print(f"Lost Samples:       {lost:,}")

    # Calculate percentages if we have total samples
    if total > 0:
        print(f"\n📈 SAMPLE ANALYSIS:")
        print(f"Success Rate:       {(successful/total)*100:.2f}%")
        print(f"Failure Rate:       {(failed/total)*100:.2f}%")
        print(f"Bias Rate:          {(biased/total)*100:.2f}%")
        if lost > 0:
            print(f"Lost Rate:          {(lost/(total+lost))*100:.2f}%")
            print(f"\n⚠️  WARNING: {lost:,} samples were lost!")
            print("   This may indicate JFR buffer overflow or high system load.")
        else:
            print(f"Lost Rate:          0.00%")
            print(f"\n✅ No samples were lost.")

    print("="*70)

def extract_queue_size_from_environment() -> str:
    """Try to extract queue size information from environment variable or defaults."""
    import os
    queue_size = os.environ.get('QUEUE_SIZE')
    if queue_size:
        return f"{queue_size} (environment override)"
    return "500 (default, scaled by sampling frequency)"

def print_final_summary(jfr_stats: Dict[str, Any], stats_data: list) -> None:
    """Print a final summary with queue size and event statistics."""
    print("\n" + "="*60)
    print("📋 JFR PROFILING SUMMARY")
    print("="*60)

    # Queue Size Information
    queue_size_info = extract_queue_size_from_environment()
    print(f"\nQueue Size: {queue_size_info}")

    # JFR Sample Statistics (most important)
    if jfr_stats:
        successful = jfr_stats.get("successful_samples", 0)
        failed = jfr_stats.get("failed_samples", 0)
        biased = jfr_stats.get("biased_samples", 0)
        total = jfr_stats.get("total_samples", 0)
        lost = jfr_stats.get("lost_samples", 0)

        print(f"\nSample Results:")
        print(f"  Successful: {successful:,}")
        print(f"  Failed:     {failed:,}")
        print(f"  Biased:     {biased:,}")
        print(f"  Lost:       {lost:,}")

        if total > 0:
            total_with_lost = total + lost
            loss_rate = (lost / total_with_lost) * 100 if total_with_lost > 0 else 0

            print(f"\nLoss Rate: {loss_rate:.2f}%", end="")

            # Critical loss warning
            if loss_rate > 5.0:
                print(f" ⚠️  CRITICAL - Consider increasing queue size (-q parameter)")
            elif loss_rate > 1.0:
                print(f" ⚠️  WARNING - Monitor system load")
            elif loss_rate > 0:
                print(f" ✓ Acceptable")
            else:
                print(f" ✓ Perfect")

    # Drain Statistics Summary
    if stats_data:
        total_drains = sum(data.get("drains", 0) for data in stats_data)
        total_events = sum(data.get("events", {}).get("sum", 0) for data in stats_data)

        print(f"\nDrain Statistics:")
        print(f"  Categories: {len(stats_data)}")
        print(f"  Operations: {total_drains:,}")
        print(f"  Events:     {total_events:,}")

    print("="*60)

def main():
    # Read input from file or stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    # Parse JFR statistics first
    jfr_stats = parse_jfr_statistics(lines)

    # Parse all DRAIN_STATS_JSON lines
    stats_data = []
    for line in lines:
        line = line.strip()
        data = parse_drain_stats_json(line)
        if data:
            stats_data.append(data)

    # Display JFR statistics if found
    if jfr_stats:
        print_jfr_statistics(jfr_stats)
        print()  # Add spacing

    if not stats_data:
        if not jfr_stats:
            print("No DRAIN_STATS_JSON lines or JFR statistics found in input")
        else:
            print("No DRAIN_STATS_JSON lines found, but JFR statistics were displayed above")
        # Still show final summary if we have JFR stats
        if jfr_stats:
            print_final_summary(jfr_stats, None)
        return

    # Visualize each stats entry
    for data in stats_data:
        print_summary(data)
        print()
        print(visualize_time_histogram(data))
        print()
        print(visualize_event_histogram(data))
        print("\n" + "="*80 + "\n")

    # Print final prominent summary
    print_final_summary(jfr_stats, stats_data)

if __name__ == "__main__":
    main()
