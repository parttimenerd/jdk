#!/usr/bin/env python3
"""
Default Configuration for JFR Queue Size Benchmarking

This is a comprehensive sample configuration file that demonstrates all available options.
Copy this file to config.py and modify it for your local testing needs.

Usage:
    cp config_default.py config.py
    # Edit config.py with your preferred settings

The config.py file is gitignored and won't be committed to the repository.
"""

import os

# =============================================================================
# MAIN CONFIGURATION - Comprehensive Testing
# =============================================================================

# Queue sizes to test - covers full range for comprehensive analysis
# Start small for quick feedback, include larger sizes for stress testing
QUEUE_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 750, 1000]

# Thread restart frequency - restart threads after every N native calls
# Lower values = more realistic thread lifecycle simulation
# Set to 1 for maximum thread churn, higher values for more stable testing
RESTART_THREADS_EVERY = 1

# Sampling intervals for JFR CPU sampling
# Balance between granularity and performance impact
SAMPLING_INTERVALS = ["1ms", "2ms", "5ms", "10ms", "20ms"]

# Native test durations in seconds
# Longer durations provide more stable statistics but take more time
NATIVE_DURATIONS = [5, 250]  # seconds

# Stack depths to test - affects memory pressure and sampling complexity
# 100 = shallow stacks, 1200 = deep stacks (realistic application depth)
STACK_DEPTHS = [100, 1200]  # Different stack depths to test

# Total test duration per configuration in seconds
# This is the overall test time, not per native duration
TEST_DURATION = 250  # seconds

# Number of threads for stress testing
# Uses system CPU count by default for optimal resource utilization
THREADS = os.cpu_count() or 4  # Use number of CPUs, fallback to 4

# =============================================================================
# MINIMAL CONFIGURATION - Quick Testing
# =============================================================================

# Reduced queue sizes for faster testing during development
# Focus on key sizes that show different behaviors
MINIMAL_QUEUE_SIZES = [100, 500]

# Essential sampling intervals for quick verification
# Cover both high-frequency and moderate-frequency sampling
MINIMAL_SAMPLING_INTERVALS = ["1ms", "10ms"]

# Shorter native durations for rapid iteration
# Good for development and CI testing
MINIMAL_NATIVE_DURATIONS = [10, 25]  # seconds

# Test both stack depths but with minimal queue sizes
# Ensures both shallow and deep stack scenarios are covered
MINIMAL_STACK_DEPTHS = [100, 1200]  # Test both stack depths

# Shorter test duration for development
# Balances meaningful data with development speed
MINIMAL_TEST_DURATION = 30  # seconds

# =============================================================================
# RENAISSANCE BENCHMARK CONFIGURATION
# =============================================================================

# Number of iterations for Renaissance benchmarks
# Higher values provide more stable results but take longer
# Renaissance benchmarks have JVM warmup considerations
RENAISSANCE_ITERATIONS = 3

# Minimal iterations for quick testing
# Useful for CI or development verification
MINIMAL_RENAISSANCE_ITERATIONS = 1  # Faster minimal testing

# =============================================================================
# ADVANCED OPTIONS (Extend your config.py with these as needed)
# =============================================================================

"""
# Uncomment and modify these in your config.py for advanced usage:

# Enable dynamic queue sizing during tests
# When True, queues can grow automatically to prevent sample loss
ENABLE_DYNAMIC_QUEUE_SIZE = False

# Output verbosity level
# 0 = Minimal output, 1 = Normal, 2 = Verbose, 3 = Debug
VERBOSITY_LEVEL = 1

# Parallel test execution
# Number of concurrent test configurations (be careful with system resources)
MAX_PARALLEL_TESTS = 1

# Plot generation options
GENERATE_INDIVIDUAL_PLOTS = True
GENERATE_GRID_PLOTS = True
GENERATE_CSV_EXPORTS = True
GENERATE_MARKDOWN_SUMMARIES = True

# Ultra-minimal configuration for rapid debugging
DEBUG_QUEUE_SIZES = [20]
DEBUG_NATIVE_DURATIONS = [1]
DEBUG_STACK_DEPTHS = [1200]
DEBUG_TEST_DURATION = 10
DEBUG_RENAISSANCE_ITERATIONS = 1
"""

# =============================================================================
# EXAMPLE USAGE SCENARIOS
# =============================================================================

"""
Copy and modify the values above in your config.py for different testing scenarios:

1. QUICK DEVELOPMENT TESTING:
   QUEUE_SIZES = [20]
   NATIVE_DURATIONS = [1]
   TEST_DURATION = 10

2. MEMORY PRESSURE TESTING:
   QUEUE_SIZES = [500, 750, 1000]
   STACK_DEPTHS = [1200]
   NATIVE_DURATIONS = [250]

3. HIGH-FREQUENCY SAMPLING ANALYSIS:
   SAMPLING_INTERVALS = ["100us", "1ms"]
   QUEUE_SIZES = [100, 200, 500]

4. RENAISSANCE BENCHMARK FOCUS:
   RENAISSANCE_ITERATIONS = 5
   QUEUE_SIZES = [100, 200, 500]
   SAMPLING_INTERVALS = ["1ms", "10ms"]

5. CI/AUTOMATED TESTING:
   QUEUE_SIZES = MINIMAL_QUEUE_SIZES
   NATIVE_DURATIONS = MINIMAL_NATIVE_DURATIONS
   TEST_DURATION = MINIMAL_TEST_DURATION
   RENAISSANCE_ITERATIONS = MINIMAL_RENAISSANCE_ITERATIONS
"""
