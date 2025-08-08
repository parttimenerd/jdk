# JFR Benchmarking & Renaissance Visualization Suite

A comprehensive suite for JFR (Java Flight Recorder) queue size benchmarking and Renaissance benchmark visualization with advanced web interface and analysis capabilities.

## 🚀 Quick Start

### Web Interface (Recommended) - Auto-Setup Enabled! ✨
```bash
python3 web_frontend.py
# 🔧 Auto-creates virtual environment
# 📦 Auto-installs Flask, pandas, matplotlib, seaborn, numpy
# 🌐 Opens web interface at http://localhost:8080
```

**No manual dependency installation needed!** The web frontend automatically:
- Creates a virtual environment (`.venv_web_frontend`)
- Installs all required packages
- Activates the environment and starts the server

Advanced options:
```bash
python3 web_frontend.py --port 9000 --debug          # Custom port and debug mode
python3 web_frontend.py --no-auto-setup              # Skip auto-setup (manual mode)
```

### Command Line Visualization
```bash
python3 renaissance_visualizer.py --yaxis-mode p --plot-type scatter
# Note: Use web_frontend.py for the modern web interface
```

### Benchmarking
```bash
python3 benchmark_queue_sizes.py --minimal     # Quick test
python3 benchmark_queue_sizes.py --run-native  # Full native benchmark
```

## ✨ Key Features

- **🌐 Advanced Web Interface**: Real-time visualization with dual Y-axes, 13+ data modes, professional styling
- **📊 Multiple Visualization Types**: Scatter, line, bar charts with statistical analysis
- **🔬 Comprehensive Benchmarking**: Systematic JFR queue size testing across configurations
- **📦 Standalone Packages**: Self-contained analysis scripts with embedded data
- **📈 High-DPI Exports**: Publication-quality outputs (150-1200 DPI)
- **🎨 Professional Styling**: 10+ color palettes, responsive design, interactive features

## 📁 Core Files

| File | Purpose |
|------|---------|
| `web_frontend.py` | **Modern web interface** (1,500+ lines) with advanced features |
| `renaissance_visualizer.py` | Legacy visualization engine (used as backend by web_frontend.py) |
| `benchmark_queue_sizes.py` | Benchmarking execution |
| `static/js/app.js` | Dynamic UI framework (2,000+ lines) |
| `static/css/framework.css` | Professional styling system |

## 📊 Visualization Modes

**Data Types**: Percentiles, drainage stats, loss analysis, signal handlers, thread metrics, GC analysis, method profiling

**Features**: Dual Y-axis, trend analysis, statistical overlays, optimal point detection, correlation analysis

## 🔧 Requirements

- Python 3.7+ with pandas, matplotlib, seaborn, numpy, flask
- Auto-setup creates virtual environment and installs dependencies
- Modern web browser for full web interface experience

## 📚 Documentation

- **📖 Complete Guide**: `README_COMPREHENSIVE.md` - Full documentation
- **🎯 Quick Reference**: This file for essential information
- **🔧 Troubleshooting**: See comprehensive guide for detailed help

## 🎯 Use Cases

- **Research**: Performance analysis and regression testing
- **Production**: Configuration optimization and monitoring
- **Education**: Performance engineering and JFR training
- **Collaboration**: Shareable analysis configurations and packages

---

**For complete documentation, examples, and advanced features, see `README_COMPREHENSIVE.md`**
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
