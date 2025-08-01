#!/usr/bin/env python3
"""
Enhanced visualization tool for structured JFR DrainStats JSON output.
This version handles the new structured format with {from, to, count} objects
instead of raw histogram arrays.

Usage:
    python3 visualize_structured_histograms.py [input_file]

    If no input file is provided, reads from stdin.

Example input format:
DRAIN_STATS_JSON:{"name":"all without locks","drains":1000,"time":{"sum":5000000,"avg":5000,"min":1000,"max":50000},"events":{"sum":2500,"avg":2.5,"min":0,"max":10},"time_histogram":[{"from":1000,"to":2000,"count":500},{"from":2000,"to":5000,"count":300}],"event_histogram":[{"from":0,"to":0,"count":200},{"from":1,"to":1,"count":300}]}
"""

import sys
import json
import re

def parse_drain_stats_json(line):
    """Extract and parse DRAIN_STATS_JSON from a log line."""
    match = re.search(r'DRAIN_STATS_JSON:(\{.*\})', line)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        return None

def format_time_ns(nanoseconds):
    """Format nanoseconds into human-readable time units."""
    if nanoseconds is None:
        return "∞"

    if nanoseconds < 1000:
        return f"{nanoseconds}ns"
    elif nanoseconds < 1000000:
        return f"{nanoseconds/1000:.1f}μs"
    elif nanoseconds < 1000000000:
        return f"{nanoseconds/1000000:.1f}ms"
    else:
        return f"{nanoseconds/1000000000:.1f}s"

def visualize_structured_histogram(histogram_data, title, max_bar_length=50):
    """Create ASCII visualization of structured histogram data."""
    if not histogram_data:
        print(f"\n{title}: No data")
        return

    print(f"\n{title}:")
    print("=" * len(title))

    # Find max count for scaling
    max_count = max(entry['count'] for entry in histogram_data)
    if max_count == 0:
        print("No data points")
        return

    # Sort by 'from' value for proper ordering
    sorted_data = sorted(histogram_data, key=lambda x: x['from'] if x['from'] is not None else float('inf'))

    for entry in sorted_data:
        count = entry['count']
        from_val = entry['from']
        to_val = entry['to']

        # Format the range label
        if 'range' in entry:
            range_label = f"{entry['range']}:"
        elif from_val == to_val:
            range_label = f"{from_val}:"
        else:
            from_str = format_time_ns(from_val) if title.lower().startswith('time') else str(from_val)
            to_str = format_time_ns(to_val) if title.lower().startswith('time') else str(to_val)
            range_label = f"{from_str}-{to_str}:"

        # Create progress bar
        bar_length = int((count * max_bar_length) / max_count)
        bar = "█" * bar_length

        # Calculate percentage
        total_count = sum(entry['count'] for entry in histogram_data)
        percentage = (count * 100.0) / total_count

        print(f"{range_label:>15} {bar:<{max_bar_length}} {count:>6} ({percentage:>5.1f}%)")

def display_stats_summary(stats):
    """Display summary statistics."""
    print(f"\n📊 Summary for '{stats['name']}':")
    print(f"   Total drains: {stats['drains']:,}")

    time_stats = stats['time']
    print(f"   Time - Sum: {format_time_ns(time_stats['sum'])}, "
          f"Avg: {format_time_ns(time_stats['avg'])}, "
          f"Min: {format_time_ns(time_stats['min'])}, "
          f"Max: {format_time_ns(time_stats['max'])}")

    event_stats = stats['events']
    print(f"   Events - Sum: {event_stats['sum']:,}, "
          f"Avg: {event_stats['avg']:.2f}, "
          f"Min: {event_stats['min']}, "
          f"Max: {event_stats['max']}")

def analyze_performance(stats):
    """Provide performance analysis and recommendations."""
    time_stats = stats['time']
    avg_time_ns = time_stats['avg']
    max_time_ns = time_stats['max']

    print(f"\n🔍 Performance Analysis:")

    # Time analysis
    if avg_time_ns < 1000:  # < 1μs
        print("   ✅ Excellent: Average drain time under 1μs")
    elif avg_time_ns < 10000:  # < 10μs
        print("   ✅ Good: Average drain time under 10μs")
    elif avg_time_ns < 100000:  # < 100μs
        print("   ⚠️  Moderate: Average drain time over 10μs")
    else:
        print("   ❌ Poor: Average drain time over 100μs")

    if max_time_ns > 1000000:  # > 1ms
        print("   ⚠️  Warning: Maximum drain time exceeds 1ms")
        print("      Consider investigating queue blocking or lock contention")

    # Event analysis
    event_stats = stats['events']
    avg_events = event_stats['avg']

    if avg_events > 50:
        print("   ⚠️  High average event count per drain (>50)")
        print("      Consider increasing queue drain frequency")
    elif avg_events < 1:
        print("   ⚠️  Very low average events per drain (<1)")
        print("      Queue may be draining too frequently")

def main():
    """Main function to process input and display visualizations."""
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as f:
                lines = f.readlines()
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        lines = sys.stdin.readlines()

    found_stats = False

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if 'DRAIN_STATS_JSON:' not in line:
            continue

        stats = parse_drain_stats_json(line)
        if not stats:
            print(f"Failed to parse JSON on line {line_num}", file=sys.stderr)
            continue

        found_stats = True

        # Display summary
        display_stats_summary(stats)

        # Visualize histograms
        visualize_structured_histogram(
            stats.get('time_histogram', []),
            f"Time Distribution - {stats['name']}"
        )

        visualize_structured_histogram(
            stats.get('event_histogram', []),
            f"Event Count Distribution - {stats['name']}"
        )

        # Performance analysis
        analyze_performance(stats)

        print("\n" + "="*80)

    if not found_stats:
        print("No DRAIN_STATS_JSON entries found in input", file=sys.stderr)
        print("Expected format: DRAIN_STATS_JSON:{...}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
