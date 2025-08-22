# JDK with Enhanced JFR Queue Size Analysis and Benchmarking Framework

This is a modernized JDK fork with enhanced Java Flight Recorder (JFR) capabilities, featuring comprehensive queue size histogram analysis, dynamic queue sizing, advanced Renaissance benchmark integration, and sophisticated data visualization with CSV export capabilities.

## ✨ Enhanced Features

### 🔍 **Extended Queue Size Analysis**
- **Histogram Range**: Extended from 1000 to 5000 events for comprehensive analysis
- **Detailed Buckets**: Fine-grained analysis with buckets up to 5000 events
- **Overflow Tracking**: Events >5000 tracked in overflow bucket
- **Advanced Visualization**: Python framework with grid plots and external legend positioning

### 🔄 **Dynamic Queue Sizing**
- **Adaptive Capacity**: Automatic queue size adjustment based on load
- **Real-time Scaling**: Queue grows dynamically to prevent sample loss
- **Queue Size Increase Tracking**: Monitor capacity changes through enhanced metrics
- **Triple Y-Axis Visualization**: Loss percentage, memory consumption, and queue increases

### 🏃 **Comprehensive Benchmarking Framework**
- **Native Stress Testing**: Dynamic stack depth testing with configurable parameters
- **Thread Restart Simulation**: Configurable thread restart frequency for realistic testing
- **Renaissance Benchmark Integration**: Full Renaissance benchmark suite support
- **Dual-Mode Operation**: Single script supporting both native and Renaissance testing
- **JFR Integration**: Automatic CPU time sampling with configurable intervals
- **Enhanced Data Export**: CSV generation for all plots enabling Excel analysis

### 📊 **Advanced Analysis & Visualization Tools**
- **Grid Plot System**: "By Interval" and "By Queue Size" visualization layouts
- **Legend Positioning**: External legends to prevent data overlap
- **Scientific Notation Prevention**: Clean axis formatting for better readability
- **CSV Export**: Comprehensive data export for all plots with Excel compatibility
- **Multiple Chart Types**: Signal handler duration, memory consumption, VM operations, drainage statistics
- **Markdown Summaries**: Auto-generated plot summaries with absolute file paths

### 🎯 **Recent Enhancements (Latest)**
- **Legend Fixes**: External positioning using `bbox_to_anchor` prevents overlap with data points
- **Axis Formatting**: Scientific notation prevented in "by Interval" plots for improved readability
- **Comprehensive CSV Export**: All plotting functions now generate corresponding CSV files
- **Data Export Structure**: Time metrics in both nanoseconds and microseconds for Excel compatibility
- **Markdown Summaries**: Auto-generated plot summaries with absolute file paths emitted on command line
- **Codebase Cleanup**: Removed unnecessary backup directories and temporary analysis files
- **Configuration Fixes**: Fixed syntax errors and streamlined local configuration settings

For build instructions please see the
[online documentation](https://openjdk.org/groups/build/doc/building.html),
or either of these files:

- [doc/building.html](doc/building.html) (html version)
- [doc/building.md](doc/building.md) (markdown version)

## Quick Start

### Comprehensive Benchmarking Framework
```bash
cd jfr_benchmarking

# FIRST TIME SETUP: Create your local configuration
cp config_default.py config.py
# Edit config.py with your preferred test parameters

# Run minimal benchmarking suite with CSV export
python3 benchmark_queue_sizes.py --minimal

# Full benchmarking with CSV export for Excel analysis
python3 benchmark_queue_sizes.py

# Progress mode with enhanced visualization
python3 benchmark_queue_sizes.py --progress

# Analysis outputs absolute paths to markdown summaries for easy access
```

**Configuration:**
- **`config_default.py`**: Comprehensive template with all options documented
- **`config.py`**: Your local settings (gitignored, safe to customize)
- **Quick setup**: `cp config_default.py config.py` then edit as needed

**Key Outputs:**
- **Plot Summaries**: Auto-generated markdown files with absolute paths displayed on command line
- **CSV Export**: All plots generate corresponding Excel-compatible CSV files
- **Enhanced Visualizations**: External legends prevent data overlap, clean axis formatting
- **Multiple Test Types**: Native stress tests and Renaissance benchmark analysis

### Testing with Native Stress Tests
```bash
cd run_in_native

# Basic native stress test with JFR sampling
./run.sh -d 30 -i 1ms -f /tmp/drain_stats.txt

### Testing with Native Stress Tests
```bash
cd run_in_native

# Basic native stress test with JFR sampling
./run.sh -d 30 -i 1ms -f /tmp/drain_stats.txt

# Deep stack analysis
./run.sh -d 60 -s 3000 -t 4 -i 1ms -f /tmp/deep_stack_stats.txt

# Dynamic queue sizing with enhanced monitoring
./run.sh --dynamic-queue-size -i 1ms -d 30 -f /tmp/dynamic_stats.txt

# Thread restart simulation for realistic JFR testing
./run.sh --restart-frequency 1000 --native-duration 2 -d 30 -i 1ms

# Comprehensive testing with all new features
./run.sh --dynamic-queue-size --restart-frequency 500 --native-duration 1 -d 60 -i 1ms
```

### Testing with Renaissance Benchmarks
```bash
cd run_in_native

# Run Renaissance benchmarks with JFR profiling
./run.sh --mode renaissance -n 3 -i 1ms -f /tmp/renaissance_stats.txt

# Renaissance with dynamic queue sizing
./run.sh --mode renaissance --dynamic-queue-size -i 100us -f /tmp/detailed_stats.txt

# High-frequency sampling on Renaissance suite
./run.sh --mode renaissance -i 100us -f /tmp/detailed_stats.txt
```

### Analyzing Results
```bash
# Analyze drain statistics with enhanced visualization including triple y-axis plots
python3 visualize_histograms.py /tmp/drain_stats.txt

# Comprehensive benchmarking with CSV export and absolute path output
python3 benchmark_queue_sizes.py --minimal

# Full analysis with progress tracking and enhanced visualization
python3 benchmark_queue_sizes.py --progress

# CSV files automatically generated for Excel analysis:
# - Signal handler duration data (nanoseconds + microseconds)
# - Memory consumption metrics
# - Loss percentage analysis
# - Drainage duration statistics
# - VM operations data

# Markdown summaries with absolute paths for easy navigation
# Paths automatically displayed on command line for quick access
```

## Key Improvements

### Dynamic Queue Sizing Implementation
- **C++ Implementation**: Added `DYNAMIC_QUEUE_SIZE` environment variable support in `jfrCPUTimeThreadSampler.cpp`
- **Automatic Scaling**: Queue capacity increases dynamically when approaching limits
- **Queue Size Increase Tracking**: New `QUEUE_SIZE_INCREASE_COUNT` metric for monitoring capacity changes
- **Throttled Output**: Printf output limited to 1-second intervals to prevent log spam

### Queue Size Histogram Extensions
- **C++ Implementation**: Updated `MAX_EVENT_COUNT` from 1000 to 5000 in `jfrThreadSampling.cpp`
- **Python Visualization**: Extended histogram ranges in `visualize_histograms.py`
- **Enhanced Buckets**: New ranges: 1001-1500, 1501-2000, 2001-3000, 3001-4000, 4001-5000
- **Overflow Handling**: Proper >5000 event tracking and visualization

### Triple Y-Axis Visualization
- **Advanced Plotting**: Simultaneous visualization of loss percentage, queue memory consumption, and queue size increases
- **Benchmark Integration**: Enhanced `benchmark_queue_sizes.py` with `--dynamic-queue-size` flag
- **Enhanced Analysis**: Triple y-axis plots show correlations between different queue metrics

### Thread Restart Simulation
- **Configurable Frequency**: `--restart-frequency N` option to restart threads every N native calls
- **Realistic Testing**: Simulates thread lifecycle events for more accurate JFR testing scenarios
- **Context Recreation**: Thread restart includes simulated context recreation for comprehensive testing
- **Integration**: Works seamlessly with dynamic queue sizing and JFR sampling

### Renaissance Benchmark Integration
- **Automatic Download**: Renaissance JAR automatically downloaded from GitHub releases
- **Smart Detection**: Intelligent JAR location detection (parent directory first)
- **Mode Switching**: `--mode renaissance` parameter for benchmark execution
- **JFR Integration**: Full JFR CPU sampling support for Renaissance benchmarks
- **Iteration Control**: `-n iterations` parameter for benchmark repetition

### Unified Testing Script
- **Single Entry Point**: `run.sh` handles both native and Renaissance testing modes
- **Parameter Validation**: Mode-specific parameter validation and help text
- **Automatic Setup**: Conditional compilation and setup based on execution mode
- **Enhanced Help**: Comprehensive usage examples for both modes

## Project Structure

```
jdk/
├── jfr_benchmarking/        # Enhanced benchmarking framework
│   ├── benchmark_queue_sizes.py    # Main benchmarking script with CSV export
│   ├── config_default.py   # Default configuration template (committed)
│   ├── config.py           # Local configuration (gitignored, user-created)
│   ├── CONFIG_README.md    # Configuration guide and examples
│   ├── visualize_histograms.py     # Queue histogram visualization
│   └── analyze_drain_categories.py # Drain statistics analysis
├── run_in_native/           # Testing framework directory
│   ├── run.sh              # Unified test runner (native + Renaissance)
│   ├── CPUStressTest.*     # Native stress test implementation
│   ├── DynamicStackCPUStressTest.*  # Dynamic stack testing
│   └── README.md           # Detailed testing documentation
├── plots/                   # Generated plot outputs with CSV exports
│   ├── summaries/          # Auto-generated markdown summaries
│   ├── signal_handler_duration_by_queue_size/  # + CSV files
│   ├── memory_consumption_by_queue_size/       # + CSV files
│   ├── drainage_duration_by_queue_size/        # + CSV files
│   └── vm_ops_loss_by_queue_size/              # + CSV files
├── benchmark_results/       # Test execution results and logs
├── src/hotspot/share/jfr/periodic/sampling/jfrThreadSampling.cpp  # Extended queue analysis
└── README.md               # This file
```

## Analysis Capabilities

### Queue Size Distribution Analysis
- **Extended Range**: Histograms now analyze up to 5000 events per queue
- **Multiple Views**: Frequency-based and time-weighted distributions
- **ASCII Visualization**: Terminal-friendly charts with Unicode bars
- **PNG Export**: Professional charts when matplotlib is available

### Drain Statistics Analysis
- **Performance Metrics**: Events/sec, drains/sec, efficiency ratios
- **Categorical Analysis**: Per-category drain performance breakdown
- **Comparative Views**: "vs. Top" percentage comparisons
- **Distribution Analysis**: Queue size and duration distributions

### Renaissance Benchmark Profiling
- **JVM Benchmark Suite**: Comprehensive Scala, Akka, Spark, and other JVM workloads
- **Realistic Workloads**: Real-world application patterns for JFR analysis
- **Configurable Execution**: Control iterations and sampling parameters
- **JFR Integration**: Full CPU time sampling with configurable intervals

## Testing Scenarios

### Performance Regression Testing
```bash
# Test before changes
./run.sh -d 120 -s 2000 -t 8 -f /tmp/before_stats.txt

# Test after changes
./run.sh -d 120 -s 2000 -t 8 -f /tmp/after_stats.txt

# Compare with Renaissance benchmarks
./run.sh --mode renaissance -n 5 -f /tmp/renaissance_before.txt
./run.sh --mode renaissance -n 5 -f /tmp/renaissance_after.txt
```

### Deep Analysis Workflow
```bash
# Comprehensive native analysis
./run.sh -d 180 -s 3000 -t 4 -i 1ms -f /tmp/comprehensive_stats.txt

# Renaissance benchmark analysis
./run.sh --mode renaissance -n 3 -i 1ms -f /tmp/renaissance_stats.txt

# Analyze all results
python3 visualize_histograms.py /tmp/comprehensive_stats.txt
python3 analyze_drain_categories.py /tmp/comprehensive_stats.txt
python3 visualize_histograms.py /tmp/renaissance_stats.txt
```

See <https://openjdk.org/> for more information about the OpenJDK
Community and the JDK and see <https://bugs.openjdk.org> for JDK issue
tracking.
