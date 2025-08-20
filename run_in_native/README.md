# JFR Testing Framework

Testing framework for JFR CPU time sampling with dynamic stack tests, dynamic queue sizing, thread restart simulation, and comprehensive analysis.

## Features

- **Dynamic stack testing** - Runtime class generation with configurable depths
- **Dynamic queue sizing** - Automatic queue capacity adjustment based on load (`--dynamic-queue-size`)
- **Thread restart simulation** - Configurable thread restart frequency for realistic testing (`--restart-frequency N`)
- **Queue size control** - Override JFR queue size to prevent sample loss
- **Triple y-axis visualization** - Simultaneous tracking of loss percentage, memory consumption, and queue increases
- **JFR sample analysis** - Automatic loss rate calculation and recommendations
- **Chunked execution** - Native→Java→native transitions for testing boundaries
- **Renaissance benchmarks** - Real-world JVM workloads

## Quick Start

```bash
# Basic test
./run.sh -d 30 -s 1000 -i 1ms

# With dynamic queue sizing
./run.sh --dynamic-queue-size -d 30 -i 1ms

# Thread restart simulation
./run.sh --restart-frequency 1000 --native-duration 2 -d 30 -i 1ms

# Comprehensive testing with all features
./run.sh --dynamic-queue-size --restart-frequency 500 --native-duration 1 -d 60 -i 1ms

# With queue size override (traditional approach)
./run.sh -d 30 -i 1ms -q 2000

# Chunked execution (5s native chunks)
./run.sh -d 60 --native-duration 5 -i 1ms

# Renaissance benchmarks with dynamic queue sizing
./run.sh --mode renaissance --dynamic-queue-size -n 5 -i 1ms
```

## Options

- `-d duration` - Test duration in seconds (default: 10)
- `-s stack_depth` - Call stack depth (default: 2000)
- `-t threads` - Thread count (default: CPU cores)
- `-i interval` - JFR sampling interval (e.g., 1ms, 100us)
- `-q size` - JFR queue size override (prevents sample loss)
- `--dynamic-queue-size` - Enable dynamic queue sizing (automatic capacity adjustment)
- `--restart-frequency N` - Restart threads every N native calls (0 = no restarts, default: 0)
- `--native-duration` - Native call duration before returning to Java
- `--mode renaissance` - Run Renaissance benchmarks instead
- `-n iterations` - Benchmark iterations (Renaissance mode)
- `-f file` - Save drain statistics to file
- `--no-analysis` - Skip automatic drain statistics analysis
- `--no-plots` - Run analysis but skip plot generation

## Analysis

The framework automatically analyzes JFR sample loss and provides recommendations. With dynamic queue sizing enabled, additional metrics are tracked:

```
📋 JFR PROFILING SUMMARY
============================================================

Queue Size: Dynamic (auto-scaling enabled)
MAX_QUEUE_SIZE_SUM: 1250
QUEUE_SIZE_INCREASE_COUNT: 3

Sample Results:
  Successful: 15,234
  Failed:     0
  Biased:     12
  Lost:       45

Loss Rate: 0.29% ✓ Acceptable

Thread Restart Summary:
  Thread 0: Completed 10 native chunks with 2 restarts
  Thread 1: Completed 10 native chunks with 2 restarts
```

**Dynamic Queue Sizing Benefits:**
- Automatic capacity adjustment prevents sample loss
- `QUEUE_SIZE_INCREASE_COUNT` tracks scaling events
- `MAX_QUEUE_SIZE_SUM` shows peak memory usage
- Eliminates need for manual queue size tuning

**Thread Restart Simulation:**
- Tests JFR behavior during thread lifecycle events
- Configurable restart frequency for realistic testing
- Simulates thread context recreation

Loss rate guidance:
- **< 1%**: ✓ Acceptable
- **1-5%**: ⚠️ Consider dynamic queue sizing (`--dynamic-queue-size`)
- **> 5%**: ⚠️ CRITICAL - Use dynamic queue sizing or increase static size significantly

## Requirements

- JDK (for compilation)
- GCC (for native libraries)
- Python 3 (optional, for enhanced analysis)
