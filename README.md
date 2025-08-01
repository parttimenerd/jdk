# JDK with Enhanced JFR Queue Size Analysis and Renaissance Benchmark Support

This is a modernized JDK fork with enhanced Java Flight Recorder (JFR) capabilities, including extended queue size histogram analysis (up to 5000 events) and comprehensive testing tools with Renaissance benchmark integration.

## Enhanced Features

### 🔍 **Extended Queue Size Analysis**
- **Histogram Range**: Extended from 1000 to 5000 events for comprehensive analysis
- **Detailed Buckets**: Fine-grained analysis with buckets up to 5000 events
- **Overflow Tracking**: Events >5000 tracked in overflow bucket
- **Visualization Tools**: Python scripts for analyzing queue size distributions

### 🏃 **Comprehensive Testing Framework**
- **Native Stress Testing**: Dynamic stack depth testing with configurable parameters
- **Renaissance Benchmark Integration**: Full Renaissance benchmark suite (v0.16.0) support
- **Dual-Mode Operation**: Single script supporting both native and Renaissance testing
- **JFR Integration**: Automatic CPU time sampling with configurable intervals
- **Drain Statistics**: Enhanced drain statistics collection and analysis

### 📊 **Advanced Analysis Tools**
- **Queue Histogram Visualization**: Extended range visualization up to 5000 events
- **Drain Statistics Analysis**: Comprehensive drain performance analysis
- **Comparative Analysis**: Before/after performance comparisons
- **Multiple Chart Types**: ASCII histograms, weighted distributions, PNG outputs

For build instructions please see the
[online documentation](https://openjdk.org/groups/build/doc/building.html),
or either of these files:

- [doc/building.html](doc/building.html) (html version)
- [doc/building.md](doc/building.md) (markdown version)

## Quick Start

### Testing with Native Stress Tests
```bash
cd run_in_native

# Basic native stress test with JFR sampling
./run.sh -d 30 -i 1ms -f /tmp/drain_stats.txt

# Deep stack analysis
./run.sh -d 60 -s 3000 -t 4 -i 1ms -f /tmp/deep_stack_stats.txt
```

### Testing with Renaissance Benchmarks
```bash
cd run_in_native

# Run Renaissance benchmarks with JFR profiling
./run.sh --mode renaissance -n 3 -i 1ms -f /tmp/renaissance_stats.txt

# High-frequency sampling on Renaissance suite
./run.sh --mode renaissance -i 100us -f /tmp/detailed_stats.txt
```

### Analyzing Results
```bash
# Analyze drain statistics with enhanced visualization
python3 visualize_histograms.py /tmp/drain_stats.txt

# Comprehensive drain categories analysis
python3 analyze_drain_categories.py /tmp/drain_stats.txt
```

## Key Improvements

### Queue Size Histogram Extensions
- **C++ Implementation**: Updated `MAX_EVENT_COUNT` from 1000 to 5000 in `jfrThreadSampling.cpp`
- **Python Visualization**: Extended histogram ranges in `visualize_histograms.py`
- **Enhanced Buckets**: New ranges: 1001-1500, 1501-2000, 2001-3000, 3001-4000, 4001-5000
- **Overflow Handling**: Proper >5000 event tracking and visualization

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
├── run_in_native/           # Testing framework directory
│   ├── run.sh              # Unified test runner (native + Renaissance)
│   ├── CPUStressTest.*     # Native stress test implementation
│   ├── DynamicStackCPUStressTest.*  # Dynamic stack testing
│   └── README.md           # Detailed testing documentation
├── visualize_histograms.py # Queue histogram visualization (extended to 5000)
├── analyze_drain_categories.py     # Drain statistics analysis
├── src/hotspot/share/jfr/periodic/sampling/jfrThreadSampling.cpp  # Extended queue analysis
└── README.md              # This file
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
