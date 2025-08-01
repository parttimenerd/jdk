# JFR Queue Size Benchmarking Suite

This folder contains tools for comprehensive benchmarking and analysis of JFR (Java Flight Recorder) queue sizes and their impact on sampling performance.

## Overview

The JFR sampling system uses internal queues to buffer sampling events. The queue size directly affects sampling performance, loss rates, and overhead. This benchmarking suite systematically tests different queue sizes, sampling intervals, native durations, and stack depths to understand their interactions and measure sampling loss rates.

## Files

### Core Benchmarking
- **`benchmark_queue_sizes.py`** - Main benchmarking script that runs systematic tests across different configurations
- **`analyze_combined_results.py`** - Analysis script that reads CSV results and generates combined visualization graphs

### Analysis Scripts
- **`analyze_benchmark_results.py`** - Individual result analysis
- **`analyze_drain_categories.py`** - Queue drain category analysis
- **`analyze_drain_stats_unified.py`** - Unified drain statistics analysis
- **`visualize_histograms.py`** - Histogram visualization tools
- **`visualize_structured_histograms.py`** - Structured histogram analysis

### Data
- **`benchmark_results/`** - Directory containing all benchmark data
  - `data/` - CSV and JSON result files
  - `logs/` - Raw JFR log files from individual tests
  - `plots/` - Generated visualization plots

## Quick Start

### 1. Run Runtime Estimates
```bash
python3 benchmark_queue_sizes.py --estimate
```

### 2. Run Minimal Test Suite (Quick Validation)
```bash
python3 benchmark_queue_sizes.py --minimal
```

### 3. Run Full Native Benchmark
```bash
python3 benchmark_queue_sizes.py --run-native
```

### 4. Run Renaissance Benchmark
```bash
python3 benchmark_queue_sizes.py --run-renaissance
```

### 5. Generate Visualizations
```bash
python3 benchmark_queue_sizes.py --visualize
```

### 6. Run Everything
```bash
python3 benchmark_queue_sizes.py --all
```

## Configuration

### Test Parameters

**Queue Sizes**: `[50, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000]`
- Range of internal queue buffer sizes to test

**Sampling Intervals**: `["1ms", "5ms", "10ms", "20ms"]`
- How frequently JFR samples are taken

**Native Durations**: `[5, 10, 20, 60, 250]` seconds
- Length of time to run native workload tests

**Stack Depths**: `[100, 1200]`
- Different call stack depths to test sampling behavior

**Test Duration**: `250` seconds (full) / `30` seconds (minimal)
- Base duration for each individual test

### Minimal Configuration (for quick testing)
- Queue sizes: `[100, 500]`
- Intervals: `["1ms", "10ms"]`
- Native durations: `[10, 60]` seconds
- Stack depths: `[100, 1200]`
- Test duration: `30` seconds

## Real-time Updates

The benchmarking system now updates CSV files in real-time as tests complete, allowing you to:
- Monitor progress during long-running benchmarks
- Analyze partial results if benchmarks are interrupted
- Track individual test completion and success rates

CSV files are updated after each test completion in:
- `benchmark_results/data/native_results_latest.csv`
- `benchmark_results/data/renaissance_results_latest.csv`

## Data Analysis

### CSV Output Columns

**Native Tests**:
- `queue_size` - JFR queue buffer size
- `interval` - Sampling interval
- `stack_depth` - Call stack depth used
- `native_duration` - Length of native test
- `test_duration` - Total test duration
- `loss_percentage` - Percentage of samples lost due to queue overflow
- `log_file` - Path to detailed log file
- `timestamp` - When test was run
- `success` - Whether test completed successfully
- `queue_stats` - Detailed queue size percentile data (when available)

**Renaissance Tests**:
- Similar to native, but with `iterations` instead of `native_duration`

### Analysis Scripts

```bash
# Generate combined loss rate plots (recommended)
python3 analyze_combined_results.py --loss-only

# Generate full analysis including queue size percentiles (if available)
python3 analyze_combined_results.py

# Analyze specific CSV file
python3 analyze_combined_results.py --csv path/to/results.csv

# Analyze drain statistics (optional - requires special log output)
python3 analyze_drain_stats_unified.py

# Generate histogram visualizations
python3 visualize_histograms.py
```

## Runtime Estimates

**Minimal Configuration (~12 minutes)**:
- Native: 8 tests × 1.5 min = 12 minutes
- Renaissance: 8 tests × 4 min = 32 minutes (if enabled)

**Full Configuration (~52 hours)**:
- Native: 400 tests × 4.67 min = ~31 hours
- Renaissance: 80 tests × 4 min = ~5.3 hours
- Combined: ~36 hours + overhead

## Features

### Advanced Capabilities
- **Environment Variable Queue Override**: Uses `QUEUE_SIZE` environment variable to override JFR internal queue sizes
- **Emoji-Aware Log Parsing**: Handles modern JFR output with emoji prefixes in section headers
- **Multi-dimensional Analysis**: Tests combinations of queue size, interval, duration, and stack depth
- **Comprehensive Statistics**: Extracts loss rates, queue size percentiles, and timing data
- **Professional Visualizations**: Generates publication-ready plots with proper legends and formatting
- **Interrupted Benchmark Recovery**: Automatically saves partial results if interrupted

### Output Visualizations
- **Heatmaps**: Loss rate by queue size and interval for each configuration
- **Line Plots**: Loss rate trends across queue sizes, grouped by sampling intervals
- **3D Surface Plots**: Multi-dimensional relationship visualization
- **Comparison Plots**: Native vs Renaissance benchmark comparisons
- **Stack Depth Analysis**: Separate analysis for different call stack depths

## Requirements

- Python 3.7+
- Required packages: `pandas`, `matplotlib`, `seaborn`, `numpy`
- Custom JDK build with JFR queue size override support
- Access to Renaissance benchmark suite (for Renaissance tests)

## Notes

- All tests run in the `../run_in_native/` directory relative to this folder
- Log files contain detailed JFR statistics and sampling loss rate analysis
- CSV files capture queue size, sampling interval, stack depth, and loss percentage data
- The benchmarking system is designed for systematic performance analysis research
- Queue size percentile extraction disabled by default for cleaner analysis

## File Organization

```
jfr_benchmarking/
├── README.md                          # This file
├── benchmark_queue_sizes.py           # Main benchmarking script
├── analyze_combined_results.py        # Combined result analysis
├── analyze_*.py                       # Various analysis scripts
├── visualize_*.py                     # Visualization tools
└── benchmark_results/                 # All output data
    ├── data/                          # CSV/JSON results
    ├── logs/                          # Individual test logs
    └── plots/                         # Generated visualizations
```
