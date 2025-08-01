#!/usr/bin/env python3
"""
Enhanced JFR Drain Statistics Analysis Tool

This script parses JFR drain statistics JSON output from log files and creates
comprehensive tables with percentile information, timing analysis, and performance
visualizations.

Usage:
    python3 analyze_jfr_drain_stats.py [log_file] [--format table|json|both] [--save-plots]

Features:
    - Parses both individual and combined drain statistics JSON
    - Creates detailed statistical tables with percentiles
    - Generates performance analysis reports
    - Optional histogram visualizations
    - Supports multiple output formats
"""

import sys
import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

def format_large_number(number: int) -> str:
    """Format large numbers with thousand separators."""
    return f"{number:,}"

def parse_drain_stats_json(line: str) -> Optional[Dict[str, Any]]:
    """Extract and parse individual DRAIN_STATS_JSON from a log line."""
    match = re.search(r'DRAIN_STATS_JSON:(\{.*\})', line)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        data['type'] = 'individual'
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing individual JSON: {e}", file=sys.stderr)
        return None

def parse_combined_drain_stats_json(line: str) -> Optional[Dict[str, Any]]:
    """Extract and parse combined COMBINED_DRAIN_STATS_JSON from a log line."""
    match = re.search(r'COMBINED_DRAIN_STATS_JSON:(\{.*\})', line)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        data['type'] = 'combined'
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing combined JSON: {e}", file=sys.stderr)
        return None

def parse_log_file(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a log file and extract all drain statistics."""
    individual_stats = []
    combined_stats = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Try to parse individual stats
                individual = parse_drain_stats_json(line)
                if individual:
                    individual_stats.append(individual)
                    continue

                # Try to parse combined stats
                combined = parse_combined_drain_stats_json(line)
                if combined:
                    combined_stats.append(combined)

    except IOError as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return individual_stats, combined_stats

def create_statistics_table(stats_list: List[Dict[str, Any]], title: str) -> str:
    """Create a formatted table of drain statistics."""
    if not stats_list:
        return f"\n{title}\n{'='*len(title)}\nNo data available.\n"

    table = []
    table.append(f"\n{title}")
    table.append("="*len(title))
    table.append("")

    # Header
    header = ["Name", "Requests", "Time Stats (min/avg/max/median/p95/p99/p99.9)", "Event Stats (min/avg/max/median/p95/p99/p99.9)"]
    table.append(f"{'Name':<20} {'Requests':<12} {'Time Stats':<50} {'Event Stats':<50}")
    table.append("-" * 132)

    for stats in stats_list:
        name = stats.get('name', 'Unknown')[:18]
        requests = format_large_number(stats.get('drains', 0))

        # Time statistics
        time_stats = stats.get('time', {})
        time_min = format_time_ns(time_stats.get('min', 0))
        time_avg = format_time_ns(time_stats.get('avg', 0))
        time_max = format_time_ns(time_stats.get('max', 0))
        time_median = format_time_ns(time_stats.get('median', 0))
        time_p95 = format_time_ns(time_stats.get('p95', 0))
        time_p99 = format_time_ns(time_stats.get('p99', 0))
        time_p999 = format_time_ns(time_stats.get('p99_9', time_stats.get('p99.9', 0)))

        time_summary = f"{time_min}/{time_avg}/{time_max}/{time_median}/{time_p95}/{time_p99}/{time_p999}"

        # Event statistics
        event_stats = stats.get('events', {})
        event_min = event_stats.get('min', 0)
        event_avg = f"{event_stats.get('avg', 0):.1f}"
        event_max = event_stats.get('max', 0)
        event_median = event_stats.get('median', 0)
        event_p95 = event_stats.get('p95', 0)
        event_p99 = event_stats.get('p99', 0)
        event_p999 = event_stats.get('p99_9', event_stats.get('p99.9', 0))

        event_summary = f"{event_min}/{event_avg}/{event_max}/{event_median}/{event_p95}/{event_p99}/{event_p999}"

        table.append(f"{name:<20} {requests:<12} {time_summary:<50} {event_summary:<50}")

    table.append("")
    return "\n".join(table)

def create_detailed_analysis(stats_list: List[Dict[str, Any]], title: str) -> str:
    """Create detailed performance analysis."""
    if not stats_list:
        return ""

    analysis = []
    analysis.append(f"\n{title} - Detailed Analysis")
    analysis.append("="*(len(title) + 20))
    analysis.append("")

    for stats in stats_list:
        name = stats.get('name', 'Unknown')
        analysis.append(f"📊 {name}")
        analysis.append("-" * (len(name) + 3))

        requests = stats.get('drains', 0)
        time_stats = stats.get('time', {})
        event_stats = stats.get('events', {})

        # Basic metrics
        analysis.append(f"  Total Operations: {format_large_number(requests)}")
        analysis.append(f"  Total Time: {format_time_ns(time_stats.get('sum', 0))}")
        analysis.append(f"  Total Events: {format_large_number(event_stats.get('sum', 0))}")
        analysis.append("")

        # Timing analysis
        analysis.append("  ⏱️  Timing Analysis:")
        analysis.append(f"    Average: {format_time_ns(time_stats.get('avg', 0))}")
        analysis.append(f"    Median:  {format_time_ns(time_stats.get('median', 0))}")
        analysis.append(f"    95th %:  {format_time_ns(time_stats.get('p95', 0))}")
        analysis.append(f"    99th %:  {format_time_ns(time_stats.get('p99', 0))}")
        analysis.append(f"    99.9%:   {format_time_ns(time_stats.get('p99_9', time_stats.get('p99.9', 0)))}")
        analysis.append(f"    Range:   {format_time_ns(time_stats.get('min', 0))} - {format_time_ns(time_stats.get('max', 0))}")
        analysis.append("")

        # Event analysis
        analysis.append("  📋 Event Analysis:")
        analysis.append(f"    Average: {event_stats.get('avg', 0):.2f} events/drain")
        analysis.append(f"    Median:  {event_stats.get('median', 0)} events")
        analysis.append(f"    95th %:  {event_stats.get('p95', 0)} events")
        analysis.append(f"    99th %:  {event_stats.get('p99', 0)} events")
        analysis.append(f"    99.9%:   {event_stats.get('p99_9', event_stats.get('p99.9', 0))} events")
        analysis.append(f"    Range:   {event_stats.get('min', 0)} - {event_stats.get('max', 0)} events")
        analysis.append("")

        # Performance insights
        analysis.append("  🔍 Performance Insights:")

        avg_time = time_stats.get('avg', 0)
        max_time = time_stats.get('max', 0)
        p99_time = time_stats.get('p99', 0)
        avg_events = event_stats.get('avg', 0)

        if avg_time < 1000:
            analysis.append("    ✅ Excellent average drain time (<1μs)")
        elif avg_time < 10000:
            analysis.append("    ✅ Very good average drain time (<10μs)")
        elif avg_time < 100000:
            analysis.append("    ⚠️  Good average drain time (<100μs)")
        elif avg_time < 1000000:
            analysis.append("    ⚠️  Fair average drain time (<1ms)")
        else:
            analysis.append("    ❌ Poor average drain time (>1ms)")

        if p99_time < 100000:
            analysis.append("    ✅ Good 99th percentile performance (<100μs)")
        elif p99_time < 1000000:
            analysis.append("    ⚠️  Acceptable 99th percentile performance (<1ms)")
        else:
            analysis.append("    ❌ Poor 99th percentile performance (>1ms)")

        if max_time > avg_time * 100:
            analysis.append("    ⚠️  Very high variance detected (max >> avg)")
        elif max_time > avg_time * 10:
            analysis.append("    ⚠️  High variance in drain times")
        else:
            analysis.append("    ✅ Consistent drain times")

        if avg_events > 50:
            analysis.append("    ⚠️  High events per drain (>50) - consider more frequent draining")
        elif avg_events > 10:
            analysis.append("    ✅ Good batching efficiency (10-50 events/drain)")
        elif avg_events > 1:
            analysis.append("    ✅ Reasonable batching (1-10 events/drain)")
        else:
            analysis.append("    ⚠️  Low batching efficiency (<1 event/drain)")

        # Calculate efficiency
        if requests > 0:
            throughput = event_stats.get('sum', 0) / (time_stats.get('sum', 1) / 1000000000)  # events per second
            if throughput > 1000000:
                analysis.append(f"    ✅ Excellent throughput: {throughput/1000000:.1f}M events/sec")
            elif throughput > 100000:
                analysis.append(f"    ✅ Good throughput: {throughput/1000:.1f}K events/sec")
            else:
                analysis.append(f"    ℹ️  Throughput: {throughput:.0f} events/sec")

        analysis.append("")

    return "\n".join(analysis)

def create_histogram_visualization(stats_list: List[Dict[str, Any]], save_plots: bool = False):
    """Create histogram visualizations if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping visualizations")
        return

    if not stats_list:
        return

    # Create time distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('JFR Drain Statistics Analysis', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, stats in enumerate(stats_list[:4]):  # Max 4 plots
        if idx >= 4:
            break

        row, col = divmod(idx, 2)
        ax = axes[row, col] if len(stats_list) > 1 else axes

        # Extract timing data
        time_stats = stats.get('time', {})
        name = stats.get('name', f'Stats {idx+1}')

        # Create a simple bar chart of timing percentiles
        percentiles = ['min', 'median', 'p95', 'p99', 'p99_9', 'max']
        values = []
        labels = []

        for p in percentiles:
            val = time_stats.get(p, time_stats.get(p.replace('_', '.'), 0))
            if val > 0:
                values.append(val)
                labels.append(p.replace('_', '.'))

        if values:
            bars = ax.bar(range(len(values)), values, color=colors[idx % len(colors)], alpha=0.7)
            ax.set_title(f"{name} - Timing Distribution", fontweight='bold')
            ax.set_ylabel('Time (ns)')
            ax.set_xlabel('Percentiles')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       format_time_ns(int(val)), ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{name} - No Data")

    # Hide unused subplots
    for idx in range(len(stats_list), 4):
        row, col = divmod(idx, 2)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jfr_drain_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {filename}")
        plt.close()
    else:
        plt.show()

def export_json_summary(individual_stats: List[Dict[str, Any]],
                       combined_stats: List[Dict[str, Any]]) -> str:
    """Export a JSON summary of all statistics."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "individual_stats_count": len(individual_stats),
            "combined_stats_count": len(combined_stats),
            "total_requests": sum(s.get('drains', 0) for s in individual_stats + combined_stats),
        },
        "individual_stats": individual_stats,
        "combined_stats": combined_stats
    }

    return json.dumps(summary, indent=2)

def main():
    """Main function to process input and generate analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze JFR Drain Statistics from log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 analyze_jfr_drain_stats.py logfile.txt
    python3 analyze_jfr_drain_stats.py logfile.txt --format table
    python3 analyze_jfr_drain_stats.py logfile.txt --format both --save-plots
    cat logfile.txt | python3 analyze_jfr_drain_stats.py --format json
        """
    )

    parser.add_argument('log_file', nargs='?', help='Log file to analyze (default: stdin)')
    parser.add_argument('--format', choices=['table', 'json', 'both'], default='table',
                       help='Output format (default: table)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots as PNG files')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation (for table-only output)')

    args = parser.parse_args()

    # Parse input
    if args.log_file:
        if not os.path.exists(args.log_file):
            print(f"Error: File '{args.log_file}' not found", file=sys.stderr)
            sys.exit(1)
        individual_stats, combined_stats = parse_log_file(args.log_file)
    else:
        # Read from stdin
        individual_stats = []
        combined_stats = []

        for line_num, line in enumerate(sys.stdin, 1):
            line = line.strip()

            individual = parse_drain_stats_json(line)
            if individual:
                individual_stats.append(individual)
                continue

            combined = parse_combined_drain_stats_json(line)
            if combined:
                combined_stats.append(combined)

    if not individual_stats and not combined_stats:
        print("No drain statistics found in input", file=sys.stderr)
        sys.exit(1)

    # Generate output based on format
    if args.format in ['table', 'both']:
        print(f"Found {len(individual_stats)} individual and {len(combined_stats)} combined drain statistics")

        if individual_stats:
            print(create_statistics_table(individual_stats, "Individual Drain Statistics"))
            print(create_detailed_analysis(individual_stats, "Individual Stats"))

        if combined_stats:
            print(create_statistics_table(combined_stats, "Combined Drain Statistics"))
            print(create_detailed_analysis(combined_stats, "Combined Stats"))

        # Create visualizations
        if individual_stats and MATPLOTLIB_AVAILABLE and not args.no_plots:
            create_histogram_visualization(individual_stats, args.save_plots)

    if args.format in ['json', 'both']:
        json_output = export_json_summary(individual_stats, combined_stats)
        if args.format == 'both':
            print("\n" + "="*50)
            print("JSON EXPORT")
            print("="*50)
        print(json_output)

if __name__ == "__main__":
    main()
