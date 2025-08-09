#!/bin/bash

# Dynamic Stack CPU Stress Test Runner & Renaissance Benchmark Runner
# Usage: ./run2.sh [--mode native|renaissance] [-d duration] [-s stack_depth] [-t threads] [-j jvm_opts] [-f stats_file] [-i sampling_interval] [-n iterations] [-h]

DURATION=10
STACK_DEPTH=2000
THREADS=$(nproc)
JVM_OPTS=""
STATS_FILE=""
SAMPLING_INTERVAL=""
MODE="native"
RENAISSANCE_ITERATIONS=""
NATIVE_DURATION=""
QUEUE_SIZE=""
NO_ANALYSIS=false
NO_PLOTS=false

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat << EOF
Dynamic Stack CPU Stress Test & Renaissance Benchmark Runner

USAGE:
    run.sh [--mode native|renaissance] [-d duration] [-s stack_depth] [-t threads] [-j jvm_opts] [-f stats_file] [-i sampling_interval] [-n iterations] [--native-duration seconds] [-q queue_size] [-h]

OPTIONS:
    --mode MODE          Execution mode: 'native' (default) or 'renaissance'
    -d duration          Duration to run test in seconds (default: 10) [native mode only]
    -s stack_depth       Stack depth for dynamic class generation (default: 2000) [native mode only]
    -t threads           Number of threads to use (default: CPU cores) [native mode only]
    -j jvm_opts          Additional JVM options (default: none)
    -f stats_file        File to write drain statistics to (default: stdout)
    -i sampling_interval CPU sampling interval for JFR (e.g., 1ms, 100us, 10ms)
    -n iterations        Number of iterations for Renaissance benchmarks [renaissance mode only]
    --native-duration    Duration for each native call before returning to Java (default: entire duration) [native mode only]
    -q queue_size        JFR queue size override (default: 500, scaled by sampling frequency)
    --no-analysis        Skip automatic drain statistics analysis (no plots/visualizations)
    --no-plots           Run analysis but skip plot/visualization generation
    -h                   Show this help message

NATIVE MODE EXAMPLES:
    run.sh                                        # Run with defaults (10s, stack depth 2000, CPU cores threads)
    run.sh -d 30 -s 100 -t 4                     # Run for 30s with stack depth 100 and 4 threads
    run.sh -d 15 -s 500 -i 1ms                   # Run with 1ms CPU sampling
    run.sh -f /tmp/drain_stats.txt               # Write drain statistics to file instead of stdout
    run.sh -d 60 --native-duration 5             # Run for 60s total, native calls return every 5s
    run.sh -d 30 --native-duration 2 -i 1ms      # 30s total, 2s native chunks, with JFR sampling
    run.sh -d 20 -i 1ms -q 1000                  # 20s test with 1ms sampling and 1000 queue size
    run.sh -q 2000 -i 100us                      # Override queue size for high-frequency sampling

RENAISSANCE MODE EXAMPLES:
    run.sh --mode renaissance                 # Run all Renaissance benchmarks with default iterations
    run.sh --mode renaissance -n 5           # Run all benchmarks with 5 iterations
    run.sh --mode renaissance -i 100us       # Run with JFR CPU sampling at 100us
    run.sh --mode renaissance -n 3 -i 1ms    # Run 3 iterations with 1ms CPU sampling

DESCRIPTION:
    NATIVE MODE:
    This test creates dynamic classes at runtime to generate call stacks of the
    specified depth, making it challenging for CPU time sampling. The classes
    contain no work - they exist solely to create stack depth for the native
    method that does all the actual CPU work.

    With --native-duration, each native method call runs for the specified duration
    before returning to Java, creating a pattern of native->Java->native transitions
    that continues until the overall test duration is complete.

    RENAISSANCE MODE:
    Downloads and runs the Renaissance benchmark suite (v0.16.0) which includes
    a variety of JVM benchmarks including Scala, Akka, Apache Spark, and others.
    The benchmark suite provides realistic workloads for JFR profiling.

    The -f option sets the STATS_FILE environment variable, which causes the JVM
    to write drain statistics to the specified file instead of stdout.

    The -i option automatically enables JFR CPU time sampling with the specified
    interval and saves the profile to 'profile_[interval].jfr'.
EOF
}

# Renaissance-specific functions
setup_renaissance() {
    local renaissance_url="https://github.com/renaissance-benchmarks/renaissance/releases/download/v0.16.0/renaissance-gpl-0.16.0.jar"
    local renaissance_jar="$SCRIPT_DIR/../renaissance.jar"

    # Check if renaissance.jar exists in parent directory (top-level)
    if [[ -f "$renaissance_jar" ]]; then
        echo "✅ Found existing renaissance.jar at: $renaissance_jar"
        return 0
    fi

    # Check if renaissance.jar exists in current directory
    if [[ -f "renaissance.jar" ]]; then
        echo "✅ Found existing renaissance.jar in current directory"
        return 0
    fi

    echo "📥 Downloading Renaissance benchmark suite v0.16.0..."
    echo "From: $renaissance_url"

    # Download renaissance.jar to parent directory to share across runs
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$renaissance_jar" "$renaissance_url" || {
            echo "❌ Failed to download Renaissance with curl"
            return 1
        }
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$renaissance_jar" "$renaissance_url" || {
            echo "❌ Failed to download Renaissance with wget"
            return 1
        }
    else
        echo "❌ ERROR: Neither curl nor wget found. Cannot download Renaissance."
        echo "Please manually download renaissance-gpl-0.16.0.jar from:"
        echo "$renaissance_url"
        echo "And save it as: $renaissance_jar"
        return 1
    fi

    echo "✅ Successfully downloaded Renaissance to: $renaissance_jar"
    return 0
}

run_renaissance() {
    local renaissance_jar="$SCRIPT_DIR/../renaissance.jar"

    # Fallback to current directory if not found in parent
    if [[ ! -f "$renaissance_jar" ]]; then
        renaissance_jar="renaissance.jar"
    fi

    if [[ ! -f "$renaissance_jar" ]]; then
        echo "❌ ERROR: Renaissance JAR not found"
        return 1
    fi

    echo "🏃 Running Renaissance benchmark suite"
    echo "JAR: $renaissance_jar"

    # Build Renaissance command
    # omit fj-means (it causes stalls for whatever reason)akka-uct
    local renaissance_cmd="$JAVA_HOME/bin/java $JVM_OPTS -jar $renaissance_jar scrabble page-rank future-genetic movie-lens scala-doku chi-square rx-scrabble db-shootout neo4j-analytics finagle-http reactors dec-tree scala-stm-bench7 naive-bayes als par-mnemonics scala-kmeans philosophers log-regression gauss-mix mnemonics dotty finagle-chirper"

    # Add iterations if specified
    if [[ -n "$RENAISSANCE_ITERATIONS" ]]; then
        renaissance_cmd="$renaissance_cmd -r $RENAISSANCE_ITERATIONS"
        echo "Iterations: $RENAISSANCE_ITERATIONS"
    else
        echo "Iterations: default (benchmark-specific)"
    fi

    echo "Command: $renaissance_cmd"
    echo
    echo "Starting Renaissance benchmarks..."
    echo "=========================================="

    # Execute Renaissance
    eval "$renaissance_cmd"
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Renaissance benchmarks completed successfully"
    else
        echo "❌ Renaissance benchmarks failed with exit code: $exit_code"
        return $exit_code
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        -d) DURATION="$2"; shift 2 ;;
        -s) STACK_DEPTH="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        -j) JVM_OPTS="$2"; shift 2 ;;
        -f) STATS_FILE="$2"; shift 2 ;;
        -i) SAMPLING_INTERVAL="$2"; shift 2 ;;
        -n) RENAISSANCE_ITERATIONS="$2"; shift 2 ;;
        --native-duration) NATIVE_DURATION="$2"; shift 2 ;;
        -q) QUEUE_SIZE="$2"; shift 2 ;;
        --no-analysis) NO_ANALYSIS=true; shift ;;
        --no-plots) NO_PLOTS=true; shift ;;
        -h) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; echo "Use -h for help"; exit 1 ;;
    esac
done

# Validate mode
if [[ "$MODE" != "native" && "$MODE" != "renaissance" ]]; then
    echo "❌ ERROR: Invalid mode '$MODE'. Must be 'native' or 'renaissance'"
    echo "Use --mode native or --mode renaissance"
    exit 1
fi

# Mode-specific validation and setup
if [[ "$MODE" == "native" ]]; then
    # Safety limits to prevent excessive resource usage
    MAX_TOTAL_CLASSES=500000  # Limit to 500K classes max
    TOTAL_CLASSES=$((STACK_DEPTH * THREADS))

    if [[ $TOTAL_CLASSES -gt $MAX_TOTAL_CLASSES ]]; then
        echo "❌ ERROR: Total classes ($TOTAL_CLASSES) exceeds safety limit ($MAX_TOTAL_CLASSES)"
        echo "   Stack depth: $STACK_DEPTH, Threads: $THREADS"
        echo "   Consider reducing -s (stack depth) or -t (threads) parameters"
        echo "   Example: ./run2.sh -s $((MAX_TOTAL_CLASSES / THREADS)) -t $THREADS"
        exit 1
    fi

    echo "🔧 NATIVE MODE: Dynamic Stack CPU Stress Test"
    echo "Duration: ${DURATION}s, Stack depth: ${STACK_DEPTH}, Threads: ${THREADS}"
    if [[ -n "$NATIVE_DURATION" ]]; then
        echo "Native call duration: ${NATIVE_DURATION}s (returns to Java every ${NATIVE_DURATION}s)"
    else
        echo "Native call duration: entire test duration (${DURATION}s)"
    fi
    if [[ -n "$QUEUE_SIZE" ]]; then
        echo "JFR queue size: ${QUEUE_SIZE} (environment variable override)"
    else
        echo "JFR queue size: default (500, scaled by sampling frequency)"
    fi
    echo "Total classes to generate: $TOTAL_CLASSES"
elif [[ "$MODE" == "renaissance" ]]; then
    echo "🏛️ RENAISSANCE MODE: Benchmark Suite"
    if [[ -n "$RENAISSANCE_ITERATIONS" ]]; then
        echo "Iterations: $RENAISSANCE_ITERATIONS"
    else
        echo "Iterations: default (benchmark-specific)"
    fi

    # Setup Renaissance (download if needed)
    setup_renaissance || exit 1
fi

# Clean up previous builds (but preserve cached classes)
rm -f *.class *.h *.so *.dylib >/dev/null 2>&1
rm -rf /tmp/jfr_dynamic_stress_* >/dev/null 2>&1

# Create cache directory structure
CACHE_DIR="build/cache_s${STACK_DEPTH}_t${THREADS}"
mkdir -p build "$CACHE_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use custom-built JDK with STATS_FILE support (relative to script location)
CUSTOM_JDK_PATH="$SCRIPT_DIR/../build/linux-x86_64-server-release/jdk"
if [[ -d "$CUSTOM_JDK_PATH" ]]; then
    JAVA_HOME="$CUSTOM_JDK_PATH"
    echo "Using custom JDK release build: $JAVA_HOME"
else
    # Fallback to system JDK
    JAVA_HOME=${JAVA_HOME:-$(dirname $(dirname $(readlink -f $(which javac))))}
    echo "Warning: Custom JDK not found, using system JDK: $JAVA_HOME"
    echo "         STATS_FILE support may not be available"
fi

# Change to script directory for compilation and execution
cd "$SCRIPT_DIR" || { echo "Failed to change to script directory: $SCRIPT_DIR"; exit 1; }

# LRU Cache Management Functions
manage_cache_lru() {
    local max_total_classes=2000000  # 2 million class limit
    local default_stack_depth=2000
    local default_threads=$(nproc)
    local default_cache_dir="build/cache_s${default_stack_depth}_t${default_threads}"

    # Get all cache directories and their sizes
    local cache_dirs=($(find build -maxdepth 1 -type d -name "cache_s*_t*" 2>/dev/null | sort))
    local total_classes=0

    # Calculate total cached classes
    for dir in "${cache_dirs[@]}"; do
        if [[ -f "$dir/dynamic_classes.jar" ]]; then
            # Extract class count from directory name: cache_s{STACK_DEPTH}_t{THREADS}
            if [[ "$dir" =~ cache_s([0-9]+)_t([0-9]+) ]]; then
                local stack_depth="${BASH_REMATCH[1]}"
                local threads="${BASH_REMATCH[2]}"
                local classes_in_jar=$((stack_depth * threads))
                total_classes=$((total_classes + classes_in_jar))
            fi
        fi
    done

    echo "📊 Cache status: $total_classes total classes across ${#cache_dirs[@]} cache directories"

    # If under limit, no cleanup needed
    if [[ $total_classes -lt $max_total_classes ]]; then
        return 0
    fi

    echo "⚠️  Cache limit exceeded ($total_classes >= $max_total_classes), running LRU cleanup..."

    # Create array with cache info: "access_time:cache_dir:class_count"
    local cache_info=()
    for dir in "${cache_dirs[@]}"; do
        # Skip default configuration - never remove it
        if [[ "$dir" == "$default_cache_dir" ]]; then
            echo "🔒 Protecting default cache: $dir"
            continue
        fi

        if [[ -f "$dir/dynamic_classes.jar" ]]; then
            local access_time=$(stat -c %Y "$dir/dynamic_classes.jar" 2>/dev/null || echo "0")
            # Extract class count from directory name
            if [[ "$dir" =~ cache_s([0-9]+)_t([0-9]+) ]]; then
                local stack_depth="${BASH_REMATCH[1]}"
                local threads="${BASH_REMATCH[2]}"
                local classes_in_dir=$((stack_depth * threads))
                cache_info+=("$access_time:$dir:$classes_in_dir")
            fi
        fi
    done

    # Sort by access time (oldest first)
    IFS=$'\n' cache_info=($(sort -n <<< "${cache_info[*]}"))

    # Remove oldest caches until under limit
    for info in "${cache_info[@]}"; do
        IFS=':' read -r access_time cache_dir classes_in_dir <<< "$info"

        if [[ $total_classes -lt $max_total_classes ]]; then
            break
        fi

        echo "🗑️  Removing LRU cache: $cache_dir ($classes_in_dir classes, last used: $(date -d @$access_time 2>/dev/null || echo "unknown"))"
        rm -rf "$cache_dir"
        total_classes=$((total_classes - classes_in_dir))
    done

    echo "✅ LRU cleanup complete. Remaining classes: $total_classes"
}

update_cache_access_time() {
    local cache_dir="$1"
    local jar_file="$cache_dir/dynamic_classes.jar"

    if [[ -f "$jar_file" ]]; then
        touch "$jar_file"  # Update access time
    fi
}

# Run LRU cache management (native mode only)
if [[ "$MODE" == "native" ]]; then
    manage_cache_lru
fi

# Define class generation function (outside conditional blocks for proper scope)
generate_thread_classes() {
    local thread_id=$1
    local stack_depth=$2
    local cache_dir=$3

    for ((i=1; i<=stack_depth; i++)); do
        local class_name="DynamicStressClass${thread_id}_${i}"
        local java_file="$cache_dir/${class_name}.java"

        # Generate Java source
        cat > "$java_file" << EOF
public class $class_name {
    public static void call(DynamicStackCPUStressTest test, int threadId, int duration) {
        // Level $i - no work, just stack depth
EOF

        if [ $i -eq $stack_depth ]; then
            # Last class calls native method
            echo "        test.work(threadId, duration);" >> "$java_file"
        else
            # Call next class in chain
            local next_class="DynamicStressClass${thread_id}_$((i+1))"
            echo "        $next_class.call(test, threadId, duration);" >> "$java_file"
        fi

        cat >> "$java_file" << EOF
    }
}
EOF
    done
}

# NATIVE MODE: Generate dynamic classes in bash with JAR caching
if [[ "$MODE" == "native" ]]; then
    echo "Checking for cached JAR (stack depth: $STACK_DEPTH, threads: $THREADS)..."

    # Check if we have a cached JAR for this configuration
    CACHED_JAR="$CACHE_DIR/dynamic_classes.jar"
    if [[ -f "$CACHED_JAR" ]]; then
        echo "✅ Found cached JAR with compiled classes, skipping generation"
        # Update access time for LRU tracking
        update_cache_access_time "$CACHE_DIR"
        SKIP_COMPILATION=true
    else
        echo "⚠️  No cached JAR found, will generate and compile classes"
        mkdir -p "$CACHE_DIR"
        SKIP_COMPILATION=false
    fi

# Generate and compile classes if not cached
if [[ "$SKIP_COMPILATION" == "false" ]]; then
    echo "Generating and compiling $((STACK_DEPTH * THREADS)) classes..."

    # Create temporary directories for generation and compilation
    TEMP_SRC_DIR="/tmp/dynamic_src_$$"
    TEMP_BUILD_DIR="/tmp/dynamic_build_$$"
    mkdir -p "$TEMP_SRC_DIR" "$TEMP_BUILD_DIR"

    # Generate classes in parallel to temporary directory
    MAX_PARALLEL_JOBS=${THREADS}
    if [[ $MAX_PARALLEL_JOBS -gt 8 ]]; then
        MAX_PARALLEL_JOBS=8  # Limit to 8 parallel jobs for file generation
    fi

    echo "Generating classes using $MAX_PARALLEL_JOBS parallel jobs..."

    for ((t=0; t<THREADS; t++)); do
        # Run in background, but limit parallel jobs
        (($(jobs -r | wc -l) >= MAX_PARALLEL_JOBS)) && wait

        generate_thread_classes $t $STACK_DEPTH "$TEMP_SRC_DIR" &

        # Show progress for large numbers of threads
        if [[ $THREADS -gt 10 ]] && [[ $((t % 10)) -eq 0 ]]; then
            echo "Started generation jobs for threads 0-$t..."
        fi
    done

    # Wait for all background jobs to complete
    wait
    echo "✅ Class generation complete"

    # Compile all generated classes
    echo "Compiling $((STACK_DEPTH * THREADS)) generated classes..."
    find "$TEMP_SRC_DIR" -name "*.java" > /tmp/java_files_$$.txt
    if [[ -s "/tmp/java_files_$$.txt" ]]; then
        # Compile all classes to temporary directory
        $JAVA_HOME/bin/javac -d "$TEMP_BUILD_DIR" -cp .:build @/tmp/java_files_$$.txt || {
            echo "Failed to compile generated classes"
            rm -f /tmp/java_files_$$.txt
            rm -rf "$TEMP_SRC_DIR" "$TEMP_BUILD_DIR"
            exit 1
        }
        rm -f /tmp/java_files_$$.txt

        # Create JAR from compiled classes
        echo "Creating JAR with $((STACK_DEPTH * THREADS)) compiled classes..."

        # Use absolute path for JAR creation
        ABSOLUTE_CACHED_JAR="$SCRIPT_DIR/$CACHED_JAR"

        # Create JAR using -C option to avoid command line length issues
        $JAVA_HOME/bin/jar cf "$ABSOLUTE_CACHED_JAR" -C "$TEMP_BUILD_DIR" . || {
            echo "Failed to create JAR file"
            rm -rf "$TEMP_SRC_DIR" "$TEMP_BUILD_DIR"
            exit 1
        }

        # Cleanup temporary directories
        rm -rf "$TEMP_SRC_DIR" "$TEMP_BUILD_DIR"

        # Update cache access time
        update_cache_access_time "$CACHE_DIR"

        echo "✅ Created and cached JAR with $((STACK_DEPTH * THREADS)) classes"
    else
        echo "No Java source files found to compile"
        rm -f /tmp/java_files_$$.txt
        rm -rf "$TEMP_SRC_DIR" "$TEMP_BUILD_DIR"
    fi
fi

    # Compile main Java class
    echo "Compiling main class..."
    $JAVA_HOME/bin/javac -h . DynamicStackCPUStressTest.java || { echo "Failed to compile main Java class"; exit 1; }

    # Detect OS and build native library
    OS=$(uname -s)
    if [[ "$OS" == "Linux"* ]]; then
        FLAGS="-I$JAVA_HOME/include -I$JAVA_HOME/include/linux -shared -fPIC -O2 -g"
        LIB="libcpustress.so"
    elif [[ "$OS" == "Darwin"* ]]; then
        FLAGS="-I$JAVA_HOME/include -I$JAVA_HOME/include/darwin -shared -fPIC -O2 -g"
        LIB="libcpustress.dylib"
    else
        echo "Unsupported OS: $OS"; exit 1
    fi

    gcc $FLAGS -o $LIB DynamicStackCPUStressTest.c -lm >/dev/null 2>&1 || { echo "Failed to build native library"; exit 1; }
fi

# Set up JFR recording if sampling interval is specified
if [[ -n "$SAMPLING_INTERVAL" ]]; then
    # Generate JFR filename based on stats file if specified, otherwise use sampling interval
    if [[ -n "$STATS_FILE" ]]; then
        # Extract basename and create related JFR filename
        STATS_BASE=$(basename "$STATS_FILE" .txt)
        JFR_FILENAME="${STATS_BASE}_${SAMPLING_INTERVAL}.jfr"
    else
        JFR_FILENAME="profile_${SAMPLING_INTERVAL}.jfr"
    fi

    JFR_OPTIONS="-XX:+FlightRecorder -XX:StartFlightRecording=jdk.CPUTimeSample#enabled=true,jdk.CPUTimeSample#throttle=${SAMPLING_INTERVAL},filename=${JFR_FILENAME}"
    echo "Enabling JFR CPU time sampling with ${SAMPLING_INTERVAL} interval"
    echo "JFR profile will be saved to: ${JFR_FILENAME}"

    # Auto-generate drain statistics file if not specified (native mode only)
    if [[ -z "$STATS_FILE" && "$MODE" == "native" ]]; then
        STATS_FILE="${THREADS}_${DURATION}_${STACK_DEPTH}_${SAMPLING_INTERVAL}.txt"
        echo "Auto-generated drain statistics file: $STATS_FILE"
        # Update JFR filename to match the auto-generated stats file
        STATS_BASE=$(basename "$STATS_FILE" .txt)
        JFR_FILENAME="${STATS_BASE}_${SAMPLING_INTERVAL}.jfr"
        JFR_OPTIONS="-XX:+FlightRecorder -XX:StartFlightRecording=jdk.CPUTimeSample#enabled=true,jdk.CPUTimeSample#throttle=${SAMPLING_INTERVAL},filename=${JFR_FILENAME}"
        echo "Updated JFR profile filename to: ${JFR_FILENAME}"
    fi

    # Add JFR options to JVM_OPTS
    if [[ -n "$JVM_OPTS" ]]; then
        JVM_OPTS="$JVM_OPTS $JFR_OPTIONS"
    else
        JVM_OPTS="$JFR_OPTIONS"
    fi
fi

# Export STATS_FILE if specified
if [[ -n "$STATS_FILE" ]]; then
    export STATS_FILE
    echo "Drain statistics will be written to: $STATS_FILE"
fi

# Export QUEUE_SIZE if specified
if [[ -n "$QUEUE_SIZE" ]]; then
    export QUEUE_SIZE
    echo "JFR queue size override: $QUEUE_SIZE"
fi

echo "Starting test execution..."

# Execute based on mode
if [[ "$MODE" == "native" ]]; then
    # Build classpath - include JAR if it exists
    CLASSPATH=".:build"
    if [[ -f "$CACHED_JAR" ]]; then
        CLASSPATH="$CLASSPATH:$CACHED_JAR"
    fi

    # Use custom JDK's java runtime for native stress test
    if [[ -n "$NATIVE_DURATION" ]]; then
        $JAVA_HOME/bin/java --enable-native-access=ALL-UNNAMED $JVM_OPTS -Djava.library.path=. -cp "$CLASSPATH" DynamicStackCPUStressTest $STACK_DEPTH $DURATION $THREADS $NATIVE_DURATION
    else
        $JAVA_HOME/bin/java --enable-native-access=ALL-UNNAMED $JVM_OPTS -Djava.library.path=. -cp "$CLASSPATH" DynamicStackCPUStressTest $STACK_DEPTH $DURATION $THREADS
    fi
elif [[ "$MODE" == "renaissance" ]]; then
    # Run Renaissance benchmarks
    run_renaissance
fi

# Function to extract and print JFR lost events statistics
print_jfr_lost_events() {
    local jfr_file="$1"

    if [[ ! -f "$jfr_file" ]]; then
        echo "Warning: JFR file not found: $jfr_file"
        return 1
    fi

    echo
    echo "📊 JFR CPU Time Sample Statistics:"
    echo "=================================="

    # Use the custom JDK's jfr tool to get CPU time statistics
    if [[ -f "$JAVA_HOME/bin/jfr" ]]; then
        local jfr_stats=$($JAVA_HOME/bin/jfr view cpu-time-statistics "$jfr_file" 2>/dev/null)
        if [[ $? -eq 0 && -n "$jfr_stats" ]]; then
            echo "$jfr_stats"

            # Save JFR statistics to log file if STATS_FILE is specified
            if [[ -n "$STATS_FILE" ]]; then
                echo "" >> "$STATS_FILE"
                echo "JFR_SAMPLE_STATISTICS:" >> "$STATS_FILE"
                echo "$jfr_stats" >> "$STATS_FILE"
                echo "JFR_SAMPLE_STATISTICS_END" >> "$STATS_FILE"
            fi

            # Extract and highlight lost samples count
            local lost_samples=$(echo "$jfr_stats" | grep "Lost Samples:" | awk '{print $3}' | tr -d ',')
            if [[ -n "$lost_samples" && "$lost_samples" != "0" ]]; then
                echo
                echo "⚠️  ATTENTION: $lost_samples lost samples detected!"
                echo "   This may indicate JFR buffer overflow or high system load."
            else
                echo
                echo "✅ No lost samples detected."
            fi
        else
            echo "Unable to extract CPU time statistics from JFR file."
            echo "This may indicate the recording doesn't contain CPU time sample events."
        fi
    else
        echo "Warning: jfr tool not found at $JAVA_HOME/bin/jfr"
        echo "Cannot extract lost events statistics."
    fi
    echo
}

# Run automatic analysis if both sampling interval and stats file are specified (native mode only)
if [[ -n "$STATS_FILE" ]]; then
    echo
    echo "=========================================="
    echo "🔍 AUTOMATIC DRAIN STATISTICS ANALYSIS"
    echo "=========================================="
    echo

    # Check if the stats file was created
    if [[ -f "$STATS_FILE" ]]; then
        echo "Analyzing drain statistics from: $STATS_FILE"
        echo "JFR profile saved to: ${JFR_FILENAME}"

        # Print JFR lost events statistics
        print_jfr_lost_events "${JFR_FILENAME}"

        # Check if the main analysis script exists and analysis is enabled
        if [[ "$NO_ANALYSIS" != "true" ]]; then
            if [[ -f "$SCRIPT_DIR/../analyze_drain_categories.py" ]]; then
                if [[ "$NO_PLOTS" == "true" ]]; then
                    python3 "$SCRIPT_DIR/../analyze_drain_categories.py" "$STATS_FILE" --no-plots
                else
                    python3 "$SCRIPT_DIR/../analyze_drain_categories.py" "$STATS_FILE"
                fi
            elif [[ -f "$SCRIPT_DIR/analyze_jfr_drain_stats.py" ]]; then
                if [[ "$NO_PLOTS" == "true" ]]; then
                    python3 "$SCRIPT_DIR/analyze_jfr_drain_stats.py" "$STATS_FILE" --no-plots
                else
                    python3 "$SCRIPT_DIR/analyze_jfr_drain_stats.py" "$STATS_FILE"
                fi
            else
                echo "Warning: No analysis script found. Generated files:"
                echo "  - Drain statistics: $STATS_FILE"
                echo "  - JFR profile: ${JFR_FILENAME}"
            fi
        else
            echo "Analysis skipped due to --no-analysis flag. Generated files:"
            echo "  - Drain statistics: $STATS_FILE"
            echo "  - JFR profile: ${JFR_FILENAME}"
        fi
    else
        echo "Warning: Drain statistics file was not created: $STATS_FILE"
        echo "This may indicate that the custom JDK build doesn't have STATS_FILE support."
        echo "JFR profile was still saved to: ${JFR_FILENAME}"

        # Still try to print JFR statistics even if drain stats file wasn't created
        print_jfr_lost_events "${JFR_FILENAME}"
    fi
elif [[ "$MODE" == "renaissance" && -n "$SAMPLING_INTERVAL" ]]; then
    echo
    echo "=========================================="
    echo "🏛️ RENAISSANCE JFR ANALYSIS"
    echo "=========================================="
    echo "JFR profile saved to: ${JFR_FILENAME}"

    # Print JFR lost events statistics for Renaissance mode
    print_jfr_lost_events "${JFR_FILENAME}"

    # Check if drain statistics were generated in Renaissance mode
    if [[ -n "$STATS_FILE" && -f "$STATS_FILE" ]]; then
        echo "Drain statistics from Renaissance benchmarks: $STATS_FILE"

        # Check if the main analysis script exists and analysis is enabled
        if [[ "$NO_ANALYSIS" != "true" ]]; then
            if [[ -f "$SCRIPT_DIR/../analyze_drain_categories.py" ]]; then
                if [[ "$NO_PLOTS" == "true" ]]; then
                    python3 "$SCRIPT_DIR/../analyze_drain_categories.py" "$STATS_FILE" --no-plots
                else
                    python3 "$SCRIPT_DIR/../analyze_drain_categories.py" "$STATS_FILE"
                fi
            elif [[ -f "$SCRIPT_DIR/analyze_jfr_drain_stats.py" ]]; then
                if [[ "$NO_PLOTS" == "true" ]]; then
                    python3 "$SCRIPT_DIR/analyze_jfr_drain_stats.py" "$STATS_FILE" --no-plots
                else
                    python3 "$SCRIPT_DIR/analyze_jfr_drain_stats.py" "$STATS_FILE"
                fi
            fi
        else
            echo "Analysis skipped due to --no-analysis flag."
        fi
    fi
elif [[ -n "$SAMPLING_INTERVAL" ]]; then
    echo
    echo "JFR profile saved to: ${JFR_FILENAME}"

    # Print JFR lost events statistics even without drain stats analysis
    print_jfr_lost_events "${JFR_FILENAME}"
fi

# Cleanup generated classes and temporary files (native mode only)
if [[ "$MODE" == "native" ]]; then
    find . -maxdepth 1 -name "DynamicStressClass*" -delete 2>/dev/null || true
    find build -maxdepth 1 -name "DynamicSt*" -delete 2>/dev/null || true
    rm -rf /tmp/jfr_dynamic_stress_* /tmp/dynamic_src_* /tmp/dynamic_build_* 2>/dev/null || true
fi
