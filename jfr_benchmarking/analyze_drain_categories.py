#!/usr/bin/env python3
"""
Sequential Drain Categories Analysis

This script analyzes drain statistics for each category individually:
- all without locks
- safepoint
- safepoint with locks
- out of thread

Each category gets its own detailed analysis without comparisons.
"""

import sys
import json
import re
from typing import List, Dict, Any, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ASCII art characters for histograms
ASCII_BARS = ["", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
ASCII_FULL_BAR = "█"
ASCII_PARTIAL_BARS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]

class ASCIIHistogram:
    """Custom ASCII histogram plotting library for terminal output."""

    @staticmethod
    def create_horizontal_bar_chart(data: List[tuple], title: str, max_width: int = 60) -> str:
        """Create a horizontal bar chart from data tuples (label, value)."""
        if not data:
            return f"{title}: No data available\n"

        lines = []
        lines.append(f"\n📊 {title}")
        lines.append("=" * min(len(title) + 4, 80))

        # Find maximum value for scaling
        max_val = max(value for _, value in data if value > 0)
        if max_val == 0:
            lines.append("No data to display")
            return "\n".join(lines)

        # Calculate label width for alignment
        max_label_width = max(len(str(label)) for label, _ in data)
        label_width = min(max_label_width, 20)

        # Create bars
        for label, value in data:
            if value <= 0:
                continue

            # Truncate label if too long
            display_label = str(label)[:label_width]

            # Scale the bar length
            bar_length = int((value / max_val) * max_width)

            # Create the bar using block characters
            full_blocks = bar_length // 8
            partial_block = bar_length % 8

            bar = ASCII_FULL_BAR * full_blocks
            if partial_block > 0:
                bar += ASCII_BARS[partial_block]

            # Format the line
            percentage = (value / max_val) * 100
            lines.append(f"{display_label:>{label_width}}: {bar:<{max_width}} {value:>8,} ({percentage:5.1f}%)")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def create_vertical_bar_chart(data: List[tuple], title: str, height: int = 20, width: int = 80) -> str:
        """Create a vertical bar chart from data tuples (label, value)."""
        if not data:
            return f"{title}: No data available\n"

        lines = []
        lines.append(f"\n📊 {title}")
        lines.append("=" * min(len(title) + 4, 80))

        # Filter out zero values
        filtered_data = [(label, value) for label, value in data if value > 0]
        if not filtered_data:
            lines.append("No data to display")
            return "\n".join(lines)

        # Find maximum value for scaling
        max_val = max(value for _, value in filtered_data)

        # Calculate bar width
        num_bars = len(filtered_data)
        if num_bars == 0:
            lines.append("No data to display")
            return "\n".join(lines)

        bar_width = max(1, width // num_bars - 1)

        # Create the chart from top to bottom
        for row in range(height, 0, -1):
            line = ""
            for i, (label, value) in enumerate(filtered_data):
                # Calculate if this row should have a bar segment
                bar_height = (value / max_val) * height
                if bar_height >= row:
                    char = ASCII_FULL_BAR
                elif bar_height >= row - 1:
                    # Partial block for fractional heights
                    fraction = bar_height - (row - 1)
                    char_index = min(len(ASCII_BARS) - 1, max(1, int(fraction * len(ASCII_BARS))))
                    char = ASCII_BARS[char_index]
                else:
                    char = " "

                line += char * bar_width + " "
            lines.append(line)

        # Add axis line
        lines.append("─" * width)

        # Add labels
        label_line = ""
        for i, (label, value) in enumerate(filtered_data):
            label_str = str(label)[:bar_width]
            label_line += f"{label_str:<{bar_width}} "
        lines.append(label_line)

        # Add value line
        value_line = ""
        for i, (label, value) in enumerate(filtered_data):
            value_str = f"{value:,}"[:bar_width]
            value_line += f"{value_str:<{bar_width}} "
        lines.append(value_line)

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def create_spark_line(data: List[float], title: str = "", width: int = 60) -> str:
        """Create a spark line from numerical data."""
        if not data:
            return f"{title}: No data available\n"

        lines = []
        if title:
            lines.append(f"📈 {title}")

        # Normalize data to spark line characters
        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            spark_line = ASCII_BARS[4] * len(data)  # Middle character for flat line
        else:
            spark_line = ""
            for value in data:
                normalized = (value - min_val) / (max_val - min_val)
                char_index = min(len(ASCII_BARS) - 1, int(normalized * (len(ASCII_BARS) - 1)))
                spark_line += ASCII_BARS[char_index]

        lines.append(spark_line)
        lines.append(f"Range: {min_val:.2f} to {max_val:.2f}")
        lines.append("")
        return "\n".join(lines)

# Define the drain categories we want to analyze
DRAIN_CATEGORIES = [
    "all without locks",
    "safepoint",
    "safepoint with locks",
    "out of thread"
]

# Combined category name (when individual categories are combined)
COMBINED_CATEGORY = "CPU Time Thread Sampling"

def extract_drain_stats_by_label(data: List[Dict[str, Any]], target_label: str) -> List[Dict[str, Any]]:
    """Extract drain statistics for a specific label only - returns only the last instance."""
    filtered_stats = []
    for stat in data:
        name = stat.get('name', '').strip()
        if name == target_label:
            filtered_stats.append(stat)

    # Return only the last instance if any found
    return [filtered_stats[-1]] if filtered_stats else []

def format_time_ns(nanoseconds: Optional[int]) -> str:
    """Format nanoseconds into human-readable time units."""
    if nanoseconds is None or nanoseconds == 0:
        return "0ns"

    if nanoseconds < 0:
        return f"{nanoseconds}ns"
    elif nanoseconds < 1000:
        return f"{nanoseconds}ns"
    elif nanoseconds < 1000000:
        return f"{nanoseconds/1000:.1f}μs"
    elif nanoseconds < 1000000000:
        return f"{nanoseconds/1000000:.2f}ms"
    else:
        return f"{nanoseconds/1000000000:.3f}s"

def format_time_bucket_ms(bucket_value: int) -> str:
    """Convert time bucket value to milliseconds for display."""
    if bucket_value == 0:
        return "0ms"

    # Time buckets appear to be in nanoseconds based on the values we're seeing
    # Convert to milliseconds for better readability
    ms_value = bucket_value / 1000000

    if ms_value < 0.001:
        return f"{bucket_value}ns"
    elif ms_value < 1:
        return f"{ms_value:.3f}ms"
    elif ms_value < 1000:
        return f"{ms_value:.1f}ms"
    else:
        return f"{ms_value/1000:.2f}s"

def create_ascii_histogram(data: List[tuple], title: str, max_width: int = 50, max_buckets: int = 20) -> str:
    """Create an ASCII histogram from data tuples (label, value), limited to max_buckets."""
    if not data:
        return f"{title}: No data available\n"

    lines = []
    lines.append(f"\n📊 {title}")
    lines.append("=" * min(len(title) + 4, 60))

    # Limit to max_buckets, keeping the highest value buckets
    if len(data) > max_buckets:
        # Sort by value (descending) and take top buckets
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        data = data_sorted[:max_buckets]
        # Re-sort by original order (for time histograms) or keep value-sorted (for event histograms)
        # Try to preserve time ordering for time histograms
        try:
            data.sort(key=lambda x: float(x[0].replace('ms', '').replace('μs', '').replace('ns', '').replace('s', '')))
        except:
            # If sorting by time fails, keep value-sorted order
            pass
        lines.append(f"Showing top {max_buckets} buckets (of {len(data_sorted)} total)")

    # Find maximum value for scaling
    max_val = max(value for _, value in data)
    if max_val == 0:
        lines.append("No data to display")
        return "\n".join(lines)

    # Calculate optimal bar width
    for label, value in data:
        # Scale the bar length
        bar_length = int((value / max_val) * max_width)

        # Create the bar using block characters
        full_blocks = bar_length // 8
        partial_block = bar_length % 8

        bar = ASCII_FULL_BAR * full_blocks
        if partial_block > 0:
            bar += ASCII_BARS[partial_block]

        # Format the line
        percentage = (value / max_val) * 100
        lines.append(f"{label:>15}: {bar:<{max_width}} {value:>8} ({percentage:5.1f}%)")

    lines.append("")
    return "\n".join(lines)

def create_ascii_time_histogram(time_histogram: List[Dict[str, Any]], title: str, max_buckets: int = 20) -> str:
    """Create ASCII histogram for time distribution data, limited to max_buckets."""
    if not time_histogram:
        return f"{title}: No timing data available\n"

    # Prepare data for histogram
    data = []
    for bucket in time_histogram:
        if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
            time_bucket = bucket.get('from', 0)
            count = bucket.get('count', 0)
            label = format_time_bucket_ms(time_bucket)
            data.append((label, count))

    if not data:
        return f"{title}: No timing data available\n"

    # Sort by time value for better visualization
    data.sort(key=lambda x: float(x[0].replace('ms', '').replace('μs', '').replace('ns', '').replace('s', '')))

    return create_ascii_histogram(data, title, max_buckets=max_buckets)

def create_ascii_event_histogram(event_histogram: List[Dict[str, Any]], title: str, max_buckets: int = 20) -> str:
    """Create ASCII histogram for event distribution data, limited to max_buckets."""
    if not event_histogram:
        return f"{title}: No event data available\n"

    # Prepare data for histogram
    data = []
    for bucket in event_histogram:
        if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
            queue_size = bucket.get('from', 0)
            count = bucket.get('count', 0)
            label = f"Queue {queue_size}"
            data.append((label, count))

    if not data:
        return f"{title}: No event data available\n"

    # Sort by queue size
    data.sort(key=lambda x: int(x[0].split()[1]))

    return create_ascii_histogram(data, title, max_buckets=max_buckets)

def create_ascii_event_weighted_histogram(event_histogram: List[Dict[str, Any]], title: str, max_buckets: int = 20) -> str:
    """Create ASCII histogram for event-weighted distribution data (events × queue_size)."""
    if not event_histogram:
        return f"{title}: No event data available\n"

    # Prepare data for weighted histogram
    data = []
    for bucket in event_histogram:
        if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
            queue_size = bucket.get('from', 0)
            count = bucket.get('count', 0)
            weighted_value = count * queue_size if queue_size > 0 else count
            label = f"Queue {queue_size}"
            data.append((label, weighted_value, count))  # (label, weighted_value, raw_count)

    if not data:
        return f"{title}: No event data available\n"

    # Sort by queue size
    data.sort(key=lambda x: int(x[0].split()[1]))

    # Create weighted histogram
    lines = []
    lines.append(f"\n📊 {title}")
    lines.append("=" * min(len(title) + 4, 60))

    # Limit to max_buckets, keeping the highest weighted value buckets
    if len(data) > max_buckets:
        # Sort by weighted value (descending) and take top buckets
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        data = data_sorted[:max_buckets]
        # Re-sort by queue size for display
        data.sort(key=lambda x: int(x[0].split()[1]))
        lines.append(f"Showing top {max_buckets} buckets (of {len(data_sorted)} total)")

    # Find maximum weighted value for scaling
    max_val = max(weighted_value for _, weighted_value, _ in data)
    if max_val == 0:
        lines.append("No data to display")
        return "\n".join(lines)

    # Create bars
    for label, weighted_value, raw_count in data:
        # Scale the bar length
        bar_length = int((weighted_value / max_val) * 50)

        # Create the bar using block characters
        full_blocks = bar_length // 8
        partial_block = bar_length % 8

        bar = ASCII_FULL_BAR * full_blocks
        if partial_block > 0:
            bar += ASCII_BARS[partial_block]

        # Format the line with both weighted and raw values
        percentage = (weighted_value / max_val) * 100
        lines.append(f"{label:>15}: {bar:<50} {weighted_value:>8} ({percentage:5.1f}%) [{raw_count} ops]")

    lines.append("")
    lines.append("Note: Bar height shows event-weighted impact (operations × queue_size)")
    lines.append("")
    return "\n".join(lines)

def create_ascii_time_weighted_histogram(time_histogram: List[Dict[str, Any]], title: str, max_buckets: int = 20) -> str:
    """Create ASCII histogram for time-weighted distribution data (operations × duration)."""
    if not time_histogram:
        return f"{title}: No timing data available\n"

    # Prepare data for weighted histogram
    data = []
    for bucket in time_histogram:
        if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
            time_bucket = bucket.get('from', 0)
            count = bucket.get('count', 0)
            weighted_value = count * time_bucket  # Weight by time duration
            label = format_time_bucket_ms(time_bucket)
            data.append((label, weighted_value, count))  # (label, weighted_value, raw_count)

    if not data:
        return f"{title}: No timing data available\n"

    # Sort by time value for better visualization
    data.sort(key=lambda x: float(x[0].replace('ms', '').replace('μs', '').replace('ns', '').replace('s', '')))

    # Create weighted histogram
    lines = []
    lines.append(f"\n📊 {title}")
    lines.append("=" * min(len(title) + 4, 60))

    # Limit to max_buckets, keeping the highest weighted value buckets
    if len(data) > max_buckets:
        # Sort by weighted value (descending) and take top buckets
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        data = data_sorted[:max_buckets]
        # Re-sort by time for display
        try:
            data.sort(key=lambda x: float(x[0].replace('ms', '').replace('μs', '').replace('ns', '').replace('s', '')))
        except:
            pass  # Keep value-sorted order if time sorting fails
        lines.append(f"Showing top {max_buckets} buckets (of {len(data_sorted)} total)")

    # Find maximum weighted value for scaling
    max_val = max(weighted_value for _, weighted_value, _ in data)
    if max_val == 0:
        lines.append("No data to display")
        return "\n".join(lines)

    # Create bars
    for label, weighted_value, raw_count in data:
        # Find corresponding raw count
        raw_count = next((count for l, _, count in data if l == label), 0)

        # Scale the bar length
        bar_length = int((weighted_value / max_val) * 50)

        # Create the bar using block characters
        full_blocks = bar_length // 8
        partial_block = bar_length % 8

        bar = ASCII_FULL_BAR * full_blocks
        if partial_block > 0:
            bar += ASCII_BARS[partial_block]

        # Format the line with both weighted and raw values
        percentage = (weighted_value / max_val) * 100
        lines.append(f"{label:>15}: {bar:<50} {weighted_value:>8} ({percentage:5.1f}%) [{raw_count} ops]")

    lines.append("")
    lines.append("Note: Bar height shows time-weighted impact (operations × duration)")
    lines.append("")
    return "\n".join(lines)

def create_weighted_category_histograms(stats_list: List[Dict[str, Any]], category_name: str, save_plots: bool = True):
    """Create weighted histograms for a single category showing event and time weighted distributions."""
    if not MATPLOTLIB_AVAILABLE or not stats_list:
        return

    # Get the single stats entry (last instance)
    stats = stats_list[0]

    # Create figure with subplots for different weighted views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{category_name.upper()} - Weighted Distribution Analysis', fontsize=16, fontweight='bold')

    time_histogram = stats.get('time_histogram', [])
    event_histogram = stats.get('event_histogram', [])

    # 1. Regular Time Distribution
    if time_histogram:
        time_buckets = []
        time_counts = []
        for bucket in time_histogram:
            if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
                time_buckets.append(bucket.get('from', 0) / 1000000)  # Convert to ms
                time_counts.append(bucket.get('count', 0))

        if time_buckets:
            ax1.bar(time_buckets, time_counts, alpha=0.7, color='skyblue', edgecolor='navy')
            ax1.set_title('Time Distribution (Frequency)')
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Operation Count')
            ax1.grid(True, alpha=0.3)

    # 2. Time-Weighted Distribution (shows where time was spent)
    if time_histogram:
        time_buckets = []
        time_weighted = []
        for bucket in time_histogram:
            if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
                time_ns = bucket.get('from', 0)
                count = bucket.get('count', 0)
                time_buckets.append(time_ns / 1000000)  # Convert to ms
                time_weighted.append(count * time_ns)  # Weight by time

        if time_buckets:
            ax2.bar(time_buckets, time_weighted, alpha=0.7, color='orange', edgecolor='darkorange')
            ax2.set_title('Time-Weighted Distribution (Time Impact)')
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Weighted Impact (ops × duration)')
            ax2.grid(True, alpha=0.3)

    # 3. Regular Event Distribution
    if event_histogram:
        event_buckets = []
        event_counts = []
        for bucket in event_histogram:
            if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
                event_buckets.append(bucket.get('from', 0))
                event_counts.append(bucket.get('count', 0))

        if event_buckets:
            ax3.bar(event_buckets, event_counts, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
            ax3.set_title('Event Queue Size Distribution (Frequency)')
            ax3.set_xlabel('Queue Size')
            ax3.set_ylabel('Occurrence Count')
            ax3.grid(True, alpha=0.3)

    # 4. Event-Weighted Distribution (shows where events accumulated)
    if event_histogram:
        event_buckets = []
        event_weighted = []
        for bucket in event_histogram:
            if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
                queue_size = bucket.get('from', 0)
                count = bucket.get('count', 0)
                event_buckets.append(queue_size)
                event_weighted.append(count * queue_size if queue_size > 0 else count)  # Weight by queue size

        if event_buckets:
            ax4.bar(event_buckets, event_weighted, alpha=0.7, color='lightcoral', edgecolor='darkred')
            ax4.set_title('Event-Weighted Distribution (Event Impact)')
            ax4.set_xlabel('Queue Size')
            ax4.set_ylabel('Weighted Impact (events × queue_size)')
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        filename = f"{category_name.lower().replace(' ', '_')}_weighted_distributions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Weighted distribution plots saved as {filename}")
        plt.close()
    else:
        plt.show()

def calculate_queue_size_distribution(event_histogram: List[int]) -> Dict[str, Any]:
    """Calculate event-weighted queue size percentiles."""
    if not event_histogram:
        return {
            'total_events': 0,
            'median_queue_size': 0,
            'p95_queue_size': 0,
            'p99_queue_size': 0,
            'p999_queue_size': 0
        }

    total_events = sum(count * queue_size for queue_size, count in enumerate(event_histogram))
    if total_events == 0:
        return {
            'total_events': 0,
            'median_queue_size': 0,
            'p95_queue_size': 0,
            'p99_queue_size': 0,
            'p999_queue_size': 0
        }

    # Calculate cumulative distribution
    cumulative_events = 0
    percentiles = {'50th': 0, '95th': 0, '99th': 0, '99.9th': 0}
    targets = {'50th': 0.5, '95th': 0.95, '99th': 0.99, '99.9th': 0.999}

    for queue_size, count in enumerate(event_histogram):
        if count > 0:
            cumulative_events += count * queue_size
            cumulative_ratio = cumulative_events / total_events

            for percentile, target in targets.items():
                if percentiles[percentile] == 0 and cumulative_ratio >= target:
                    percentiles[percentile] = queue_size

    return {
        'total_events': total_events,
        'median_queue_size': percentiles['50th'],
        'p95_queue_size': percentiles['95th'],
        'p99_queue_size': percentiles['99th'],
        'p999_queue_size': percentiles['99.9th']
    }

def calculate_event_weighted_percentiles(event_histogram: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate event-weighted queue size percentiles from histogram data."""
    if not event_histogram:
        return {'median': 0, 'p95': 0, 'p99': 0, 'p999': 0, 'p9999': 0, 'p99999': 0}

    # Build cumulative distribution
    total_events = 0
    queue_counts = []

    for bucket in event_histogram:
        if isinstance(bucket, dict):
            queue_size = bucket.get('from', 0)
            count = bucket.get('count', 0)
            if count > 0:
                queue_counts.append((queue_size, count))
                total_events += count

    if total_events == 0:
        return {'median': 0, 'p95': 0, 'p99': 0, 'p999': 0, 'p9999': 0, 'p99999': 0}

    # Sort by queue size
    queue_counts.sort(key=lambda x: x[0])

    # Calculate percentiles
    cumulative = 0
    percentiles = {'median': 0, 'p95': 0, 'p99': 0, 'p999': 0, 'p9999': 0, 'p99999': 0}
    targets = {'median': 0.5, 'p95': 0.95, 'p99': 0.99, 'p999': 0.999, 'p9999': 0.9999, 'p99999': 0.99999}

    for queue_size, count in queue_counts:
        cumulative += count
        ratio = cumulative / total_events

        for name, target in targets.items():
            if percentiles[name] == 0 and ratio >= target:
                percentiles[name] = queue_size

    return percentiles

def calculate_time_weighted_percentiles(time_histogram: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate time-weighted percentiles from histogram data."""
    if not time_histogram:
        return {'median': 0, 'p95': 0, 'p99': 0, 'p999': 0, 'p9999': 0, 'p99999': 0}

    # Build cumulative distribution
    total_time_events = 0
    time_counts = []

    for bucket in time_histogram:
        if isinstance(bucket, dict):
            time_bucket = bucket.get('from', 0)
            count = bucket.get('count', 0)
            if count > 0:
                time_counts.append((time_bucket, count))
                total_time_events += count

    if total_time_events == 0:
        return {'median': 0, 'p95': 0, 'p99': 0, 'p999': 0, 'p9999': 0, 'p99999': 0}

    # Sort by time bucket
    time_counts.sort(key=lambda x: x[0])

    # Calculate percentiles
    cumulative = 0
    percentiles = {'median': 0, 'p95': 0, 'p99': 0, 'p999': 0, 'p9999': 0, 'p99999': 0}
    targets = {'median': 0.5, 'p95': 0.95, 'p99': 0.99, 'p999': 0.999, 'p9999': 0.9999, 'p99999': 0.99999}

    for time_bucket, count in time_counts:
        cumulative += count
        ratio = cumulative / total_time_events

        for name, target in targets.items():
            if percentiles[name] == 0 and ratio >= target:
                percentiles[name] = time_bucket

    return percentiles

def create_queue_size_analysis_table(all_categories_stats: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a table showing queue size analysis - what queue sizes did X% of events belong to."""
    table = []
    table.append("🗂️  QUEUE SIZE ANALYSIS - Event-Weighted Percentiles")
    table.append("=" * 70)
    table.append("")
    table.append("This shows: 'X% of all events belonged to a queue with at most size N'")
    table.append("")

    # Header
    header = f"{'Category':<25} {'50% ≤ size':<12} {'95% ≤ size':<12} {'99% ≤ size':<12} {'99.9% ≤ size':<14} {'99.99% ≤ size':<15} {'99.999% ≤ size':<15}"
    table.append(header)
    table.append("-" * len(header))

    for category_name, stats_list in all_categories_stats.items():
        if not stats_list:
            continue

        # Get the last (and only) instance for this category
        stats = stats_list[0]
        event_histogram = stats.get('event_histogram', [])

        percentiles = calculate_event_weighted_percentiles(event_histogram)

        category_display = category_name[:24]
        median_str = str(percentiles['median'])
        p95_str = str(percentiles['p95'])
        p99_str = str(percentiles['p99'])
        p999_str = str(percentiles['p999'])
        p9999_str = str(percentiles['p9999'])
        p99999_str = str(percentiles['p99999'])

        row = f"{category_display:<25} {median_str:<12} {p95_str:<12} {p99_str:<12} {p999_str:<14} {p9999_str:<15} {p99999_str:<15}"
        table.append(row)

    table.append("")
    return "\n".join(table)

def create_timing_analysis_table(all_categories_stats: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a table showing timing analysis - what timing buckets did X% of operations belong to."""
    table = []
    table.append("⏱️  TIMING ANALYSIS - Time-Weighted Percentiles")
    table.append("=" * 85)
    table.append("")
    table.append("This shows: 'X% of all timed operations took at most N milliseconds'")
    table.append("")

    # Header
    header = f"{'Category':<25} {'50% ≤ time':<15} {'95% ≤ time':<15} {'99% ≤ time':<15} {'99.9% ≤ time':<16} {'99.99% ≤ time':<17} {'99.999% ≤ time':<17}"
    table.append(header)
    table.append("-" * len(header))

    for category_name, stats_list in all_categories_stats.items():
        if not stats_list:
            continue

        # Get the last (and only) instance for this category
        stats = stats_list[0]
        time_histogram = stats.get('time_histogram', [])

        percentiles = calculate_time_weighted_percentiles(time_histogram)

        category_display = category_name[:24]
        median_str = format_time_bucket_ms(percentiles['median'])
        p95_str = format_time_bucket_ms(percentiles['p95'])
        p99_str = format_time_bucket_ms(percentiles['p99'])
        p999_str = format_time_bucket_ms(percentiles['p999'])
        p9999_str = format_time_bucket_ms(percentiles['p9999'])
        p99999_str = format_time_bucket_ms(percentiles['p99999'])

        row = f"{category_display:<25} {median_str:<15} {p95_str:<15} {p99_str:<15} {p999_str:<16} {p9999_str:<17} {p99999_str:<17}"
        table.append(row)

    table.append("")
    return "\n".join(table)

def create_runtime_summary_table(all_categories_stats: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a summary table showing runtime and throughput information for all categories."""
    table = []
    table.append("⏰ RUNTIME & THROUGHPUT SUMMARY")
    table.append("=" * 75)
    table.append("")

    # Find the category with the highest events/sec for comparison
    max_events_per_sec = 0
    max_events_per_sec_category = ""

    for category_name, stats_list in all_categories_stats.items():
        if stats_list:
            stats = stats_list[0]
            runtime_seconds = stats.get('runtime_seconds', 0)
            total_events = stats.get('events', {}).get('sum', 0)
            if runtime_seconds > 0:
                events_per_sec = total_events / runtime_seconds
                if events_per_sec > max_events_per_sec:
                    max_events_per_sec = events_per_sec
                    max_events_per_sec_category = category_name

    # Header
    header = f"{'Category':<25} {'Duration (s)':<12} {'Drains/sec':<12} {'Events/sec':<12} {'vs. Top':<15} {'Efficiency':<12}"
    table.append(header)
    table.append("-" * len(header))

    for category_name, stats_list in all_categories_stats.items():
        if not stats_list:
            continue

        # Get the last (and only) instance for this category
        stats = stats_list[0]
        runtime_seconds = stats.get('runtime_seconds', 0)
        total_drains = stats.get('drains', 0)
        total_events = stats.get('events', {}).get('sum', 0)

        category_display = category_name[:24]

        if runtime_seconds > 0:
            duration_str = f"{runtime_seconds:.3f}"
            drains_per_sec = total_drains / runtime_seconds
            events_per_sec = total_events / runtime_seconds
            drains_per_sec_str = f"{drains_per_sec:,.0f}"
            events_per_sec_str = f"{events_per_sec:,.0f}"

            # Calculate comparison with highest throughput
            if max_events_per_sec > 0:
                vs_top_ratio = (events_per_sec / max_events_per_sec) * 100
                if vs_top_ratio == 100:
                    vs_top_str = "100.000000%"
                else:
                    vs_top_str = f"{vs_top_ratio:.6f}%"
            else:
                vs_top_str = "N/A"

            # Calculate efficiency (events per drain)
            efficiency = total_events / total_drains if total_drains > 0 else 0
            efficiency_str = f"{efficiency:.2f}"
        else:
            duration_str = "N/A"
            drains_per_sec_str = "N/A"
            events_per_sec_str = "N/A"
            vs_top_str = "N/A"
            efficiency_str = "N/A"

        row = f"{category_display:<25} {duration_str:<12} {drains_per_sec_str:<12} {events_per_sec_str:<12} {vs_top_str:<15} {efficiency_str:<12}"
        table.append(row)

    table.append("")
    table.append("Note: Efficiency = Events per Drain (higher is better)")
    if max_events_per_sec_category:
        table.append(f"      '{max_events_per_sec_category}' has the highest throughput ({max_events_per_sec:,.0f} events/sec)")
    table.append("")
    return "\n".join(table)

def create_event_sum_statistics_table(all_categories_stats: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a summary table showing event sum statistics for all categories."""
    table = []
    table.append("📈 EVENT SUM STATISTICS SUMMARY")
    table.append("=" * 60)
    table.append("")

    # Find the category with the most events for comparison
    max_events_category = ""
    max_events_count = 0
    for category_name, stats_list in all_categories_stats.items():
        if stats_list:
            stats = stats_list[0]
            event_stats = stats.get('events', {})
            total_events = event_stats.get('sum', 0)
            if total_events > max_events_count:
                max_events_count = total_events
                max_events_category = category_name

    # Header
    header = f"{'Category':<25} {'Total Events':<15} {'vs. Top':<15} {'Min':<10} {'Avg':<10} {'Max':<10} {'Median':<10} {'P95':<10} {'P99':<10} {'P99.9':<10}"
    table.append(header)
    table.append("-" * len(header))

    for category_name, stats_list in all_categories_stats.items():
        if not stats_list:
            continue

        # Get the last (and only) instance for this category
        stats = stats_list[0]
        event_stats = stats.get('events', {})

        category_display = category_name[:24]  # Truncate if too long
        total_events = event_stats.get('sum', 0)
        total_events_str = f"{total_events:,}"

        # Calculate comparison with highest
        if max_events_count > 0:
            vs_top_ratio = (total_events / max_events_count) * 100
            if vs_top_ratio == 100:
                vs_top_str = "100.000000%"
            else:
                vs_top_str = f"{vs_top_ratio:.6f}%"
        else:
            vs_top_str = "N/A"

        min_events = str(event_stats.get('min', 0))
        avg_events = f"{event_stats.get('average', event_stats.get('avg', 0)):.1f}"
        max_events = str(event_stats.get('max', 0))
        median_events = str(event_stats.get('50th', event_stats.get('median', 0)))
        p95_events = str(event_stats.get('95th', event_stats.get('p95', 0)))
        p99_events = str(event_stats.get('99th', event_stats.get('p99', 0)))
        p999_events = str(event_stats.get('99.9th', event_stats.get('p99.9', 0)))

        row = f"{category_display:<25} {total_events_str:<15} {vs_top_str:<15} {min_events:<10} {avg_events:<10} {max_events:<10} {median_events:<10} {p95_events:<10} {p99_events:<10} {p999_events:<10}"
        table.append(row)

    table.append("")
    if max_events_category:
        table.append(f"Note: '{max_events_category}' has the highest event count ({max_events_count:,} events)")
    table.append("")
    return "\n".join(table)

def create_queue_distribution_table(stats_list: List[Dict[str, Any]], category_name: str) -> str:
    """Create queue size distribution table for a category."""
    if not stats_list:
        return ""

    table = []
    table.append(f"🏗️  Queue Size Distribution - {category_name}")
    table.append("-" * (len(category_name) + 30))
    table.append("")

    for i, stats in enumerate(stats_list, 1):
        event_histogram = stats.get('event_histogram', [])
        if not event_histogram:
            table.append(f"Instance #{i}: No event histogram data")
            continue

        # Handle both list and dict formats
        total_events = stats.get('events', {}).get('sum', 0)

        if isinstance(event_histogram, list) and event_histogram:
            # New format: [{"from": 1, "to": 1, "count": 1}] - just display the distribution info
            pass  # We'll use the total_events from the stats directly

        if total_events == 0:
            table.append(f"Instance #{i}: No events processed")
            continue

        table.append(f"Instance #{i}:")
        table.append(f"  Total Events: {total_events:,}")
        table.append("")

    return "\n".join(table)

def create_single_category_table(stats_list: List[Dict[str, Any]], category_name: str) -> str:
    """Create a detailed table for a single drain category."""
    if not stats_list:
        return f"\n{category_name} - No data available\n"

    table = []
    table.append(f"\n📊 {category_name.upper()} - Last Instance Analysis")
    table.append("=" * (len(category_name) + 30))
    table.append("")

    # Summary statistics (should only be 1 instance - the last one)
    total_instances = len(stats_list)
    total_drains = sum(stat.get('drains', 0) for stat in stats_list)
    total_events = sum(stat.get('events', {}).get('sum', 0) for stat in stats_list)

    table.append(f"Category: {category_name}")
    table.append(f"Analysis: Last instance only")
    table.append(f"Total Drains: {total_drains:,}")
    table.append(f"Total Events: {total_events:,}")

    # Add runtime information if available
    for stats in stats_list:
        runtime_seconds = stats.get('runtime_seconds', 0)
        runtime_ns = stats.get('runtime_ns', 0)
        if runtime_seconds > 0:
            table.append(f"Recording Duration: {runtime_seconds:.3f} seconds ({runtime_ns:,} ns)")
            # Calculate rates
            if runtime_seconds > 0:
                drains_per_sec = total_drains / runtime_seconds
                events_per_sec = total_events / runtime_seconds
                table.append(f"Throughput: {drains_per_sec:,.0f} drains/sec, {events_per_sec:,.0f} events/sec")
        break

    table.append("")

    # Table header
    header = f"{'Instance':<10} {'Drains':<12} {'Time Stats (min/avg/max/med/p95/p99/p999)':<55} {'Event Stats (min/avg/max/med/p95/p99/p999)':<55}"
    table.append(header)
    table.append("-" * len(header))

    # Individual instance analysis (should be only 1)
    for i, stats in enumerate(stats_list, 1):
        instance_name = f"Last"
        drains = f"{stats.get('drains', 0):,}"

        # Time statistics
        time_stats = stats.get('time', {})
        time_min = format_time_ns(time_stats.get('min', 0))
        time_avg = format_time_ns(time_stats.get('average', time_stats.get('avg', 0)))
        time_max = format_time_ns(time_stats.get('max', 0))
        time_median = format_time_ns(time_stats.get('50th', time_stats.get('median', 0)))
        time_p95 = format_time_ns(time_stats.get('95th', time_stats.get('p95', 0)))
        time_p99 = format_time_ns(time_stats.get('99th', time_stats.get('p99', 0)))
        time_p999 = format_time_ns(time_stats.get('99.9th', time_stats.get('p99.9', 0)))

        time_summary = f"{time_min}/{time_avg}/{time_max}/{time_median}/{time_p95}/{time_p99}/{time_p999}"

        # Event statistics
        event_stats = stats.get('events', {})
        event_min = event_stats.get('min', 0)
        event_avg = f"{event_stats.get('average', event_stats.get('avg', 0)):.1f}"
        event_max = event_stats.get('max', 0)
        event_median = event_stats.get('50th', event_stats.get('median', 0))
        event_p95 = event_stats.get('95th', event_stats.get('p95', 0))
        event_p99 = event_stats.get('99th', event_stats.get('p99', 0))
        event_p999 = event_stats.get('99.9th', event_stats.get('p99.9', 0))

        event_summary = f"{event_min}/{event_avg}/{event_max}/{event_median}/{event_p95}/{event_p99}/{event_p999}"

        row = f"{instance_name:<10} {drains:<12} {time_summary:<55} {event_summary:<55}"
        table.append(row)

    table.append("")

    # Performance insights
    table.append("🔍 Performance Insights:")
    for i, stats in enumerate(stats_list, 1):
        time_stats = stats.get('time', {})
        avg_time = time_stats.get('average', time_stats.get('avg', 0))

        if avg_time < 1000:
            performance = "🟢 Excellent"
        elif avg_time < 10000:
            performance = "🟡 Good"
        elif avg_time < 100000:
            performance = "🟠 Fair"
        else:
            performance = "🔴 Poor"

        table.append(f"   Last Instance: {performance} (avg: {format_time_ns(avg_time)})")

    table.append("")
    return "\n".join(table)

def create_category_histogram(stats_list: List[Dict[str, Any]], category_name: str, save_plots: bool = False):
    """Create histograms for a single category."""
    if not MATPLOTLIB_AVAILABLE or not stats_list:
        return

    # Since we're only analyzing last instance, create a single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'{category_name.upper()} - Last Instance Distribution Analysis', fontsize=16, fontweight='bold')

    # Get the single stats entry (last instance)
    stats = stats_list[0]

    # Create combined histogram showing both time and events
    time_histogram = stats.get('time_histogram', [])
    event_histogram = stats.get('event_histogram', [])

    # Handle histogram data structure: array of objects with from/to/count
    time_buckets = []
    time_counts = []
    if time_histogram:
        for bucket in time_histogram:
            if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
                time_buckets.append(bucket.get('from', 0))
                time_counts.append(bucket.get('count', 0))

        if time_buckets:
            ax.bar([i - 0.2 for i in range(len(time_buckets))], time_counts, width=0.4,
                  alpha=0.7, color='skyblue', label='Time Buckets', edgecolor='navy')

    event_buckets = []
    event_counts = []
    if event_histogram:
        for bucket in event_histogram:
            if isinstance(bucket, dict) and bucket.get('count', 0) > 0:
                event_buckets.append(bucket.get('from', 0))
                event_counts.append(bucket.get('count', 0))

        if event_buckets:
            ax.bar([i + 0.2 for i in range(len(event_buckets))], event_counts, width=0.4,
                  alpha=0.7, color='orange', label='Event Counts', edgecolor='darkorange')

    ax.set_title(f'Last Instance - Histogram Distribution')
    ax.set_xlabel('Bucket Index')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        filename = f"{category_name.lower().replace(' ', '_')}_last_instance_histogram.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Histogram saved as {filename}")
        plt.close()
    else:
        plt.show()

def analyze_single_category(input_data: List[Dict[str, Any]], category_name: str, no_plots: bool = False):
    """Analyze a single drain statistics category - last instance only."""
    print(f"\n{'='*80}")
    print(f"ANALYZING CATEGORY: {category_name.upper()} (LAST INSTANCE ONLY)")
    print(f"{'='*80}")

    # Filter for this specific category (returns only last instance)
    category_stats = extract_drain_stats_by_label(input_data, category_name)

    if not category_stats:
        print(f"❌ No statistics found for category: '{category_name}'")
        return

    print(f"✅ Found last instance for '{category_name}'")

    # Create detailed table
    table = create_single_category_table(category_stats, category_name)
    print(table)

    # Create queue distribution table
    queue_table = create_queue_distribution_table(category_stats, category_name)
    print(queue_table)

    # Create ASCII histograms for the last instance
    stats = category_stats[0]

    # Time distribution histogram
    time_histogram = stats.get('time_histogram', [])
    if time_histogram:
        time_hist_ascii = create_ascii_time_histogram(time_histogram, f"{category_name.title()} - Time Distribution")
        print(time_hist_ascii)

    # Event distribution histogram
    event_histogram = stats.get('event_histogram', [])
    if event_histogram:
        event_hist_ascii = create_ascii_event_histogram(event_histogram, f"{category_name.title()} - Event Queue Size Distribution")
        print(event_hist_ascii)

    # Weighted histograms - showing impact-based distributions
    if time_histogram:
        time_weighted_ascii = create_ascii_time_weighted_histogram(time_histogram, f"{category_name.title()} - Time-Weighted Distribution (Impact)")
        print(time_weighted_ascii)

    if event_histogram:
        event_weighted_ascii = create_ascii_event_weighted_histogram(event_histogram, f"{category_name.title()} - Event-Weighted Distribution (Impact)")
        print(event_weighted_ascii)

    # Create matplotlib histograms if available and not disabled
    if MATPLOTLIB_AVAILABLE and not no_plots:
        create_weighted_category_histograms(category_stats, category_name, save_plots=True)
    elif no_plots:
        print("📊 Plot generation skipped (--no-plots flag)")
    else:
        print("⚠️  Matplotlib not available - weighted distribution plots not generated")

def load_drain_stats_from_log(file_path: str) -> List[Dict[str, Any]]:
    """Load drain statistics from log file or stdin."""
    all_stats = []

    if file_path == '-':
        # Read from stdin
        input_text = sys.stdin.read()
        lines = input_text.strip().split('\n')
    else:
        # Read from file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

    for line in lines:
        # Look for individual drain stats
        if 'DRAIN_STATS_JSON:' in line:
            match = re.search(r'DRAIN_STATS_JSON:\s*(\{.*\})', line)
            if match:
                try:
                    data = json.loads(match.group(1))
                    all_stats.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing JSON: {e}")
                    continue

        # Also look for combined drain stats
        elif 'COMBINED_DRAIN_STATS_JSON:' in line:
            match = re.search(r'COMBINED_DRAIN_STATS_JSON:\s*(\{.*\})', line)
            if match:
                try:
                    data = json.loads(match.group(1))
                    all_stats.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing combined JSON: {e}")
                    continue

    return all_stats

def main():
    # Simple argument parsing
    log_file = None
    no_plots = False

    if len(sys.argv) < 2:
        print("Usage: python3 analyze_drain_categories.py <log_file> [--no-plots]")
        print("       cat log_file | python3 analyze_drain_categories.py - [--no-plots]")
        print("")
        print("This script analyzes drain statistics for each category individually:")
        for category in DRAIN_CATEGORIES:
            print(f"  - {category}")
        sys.exit(1)

    # Parse arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--no-plots":
            no_plots = True
        elif not log_file:
            log_file = arg

    if not log_file:
        print("Error: Log file required")
        sys.exit(1)

    # Load data
    print("🔍 Loading drain statistics...")
    all_stats = load_drain_stats_from_log(log_file)

    if not all_stats:
        print("❌ No drain statistics found in input")
        sys.exit(1)

    print(f"✅ Loaded {len(all_stats)} drain statistics entries")

    # Collect stats for all categories (last instance only)
    all_categories_stats = {}

    # Analyze each category sequentially
    for category in DRAIN_CATEGORIES:
        category_stats = extract_drain_stats_by_label(all_stats, category)
        all_categories_stats[category] = category_stats

        analyze_single_category(all_stats, category, no_plots)
        print("\n" + "="*80 + "\n")

    # Print summary table with event sum statistics
    print(create_event_sum_statistics_table(all_categories_stats))

    # Print runtime and throughput summary
    print(create_runtime_summary_table(all_categories_stats))

    # Print queue size analysis table
    print(create_queue_size_analysis_table(all_categories_stats))

    # Print timing analysis table
    print(create_timing_analysis_table(all_categories_stats))

    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main()
