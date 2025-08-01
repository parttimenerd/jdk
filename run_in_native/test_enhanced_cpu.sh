#!/bin/bash

echo "=== Enhanced CPU Stress Test Demo ==="
echo

# Build the tests
echo "Building CPU stress tests..."
make clean && make all

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"
echo

# Function to run JFR analysis
analyze_jfr() {
    local jfr_file=$1
    local name=$2

    echo "Analyzing $name JFR recording..."

    # Check if we have any drain statistics in the generated JFR
    if [ -f "$jfr_file" ]; then
        echo "JFR file $jfr_file created successfully ($(stat -c%s "$jfr_file") bytes)"

        # You could add JFR analysis commands here, for example:
        # jfr print --events CPULoad,ExecutionSample "$jfr_file" | head -20
    else
        echo "Warning: JFR file $jfr_file not found"
    fi
    echo
}

# Test 1: Original CPU stress test
echo "=== Test 1: Original CPU Stress Test ==="
timeout 15s java -XX:+FlightRecorder \
    -XX:StartFlightRecording=duration=10s,filename=test_original.jfr \
    -Djava.library.path=. CPUStressTest 8

analyze_jfr "test_original.jfr" "Original"

# Test 2: Enhanced test with moderate stack depth
echo "=== Test 2: Enhanced Test (Moderate Stack Depth) ==="
timeout 20s java -XX:+FlightRecorder \
    -XX:StartFlightRecording=duration=15s,filename=test_enhanced_moderate.jfr \
    -Djava.library.path=. EnhancedCPUStressTest 12 4 20 60 true

analyze_jfr "test_enhanced_moderate.jfr" "Enhanced Moderate"

# Test 3: Enhanced test with deep stack depth
echo "=== Test 3: Enhanced Test (Deep Stack Depth) ==="
timeout 25s java -XX:+FlightRecorder \
    -XX:StartFlightRecording=duration=20s,filename=test_enhanced_deep.jfr \
    -Djava.library.path=. EnhancedCPUStressTest 15 4 80 200 true

analyze_jfr "test_enhanced_deep.jfr" "Enhanced Deep"

# Test 4: Enhanced test with extreme parameters
echo "=== Test 4: Enhanced Test (Extreme Parameters) ==="
timeout 35s java -XX:+FlightRecorder \
    -XX:StartFlightRecording=duration=30s,filename=test_enhanced_extreme.jfr \
    -Djava.library.path=. EnhancedCPUStressTest 25 8 150 500 true

analyze_jfr "test_enhanced_extreme.jfr" "Enhanced Extreme"

# Test 5: Java-only version (no native code)
echo "=== Test 5: Enhanced Test (Java Only) ==="
timeout 20s java -XX:+FlightRecorder \
    -XX:StartFlightRecording=duration=15s,filename=test_enhanced_java_only.jfr \
    -Djava.library.path=. EnhancedCPUStressTest 12 4 30 100 false

analyze_jfr "test_enhanced_java_only.jfr" "Enhanced Java Only"

echo "=== Summary ==="
echo "Generated JFR files:"
ls -la *.jfr 2>/dev/null || echo "No JFR files found"

echo
echo "Generated class files during test:"
find /tmp/jfr_stress_test/ -name "*.java" -o -name "*.class" 2>/dev/null || echo "No dynamic classes found"

echo
echo "Cleaning up temporary files..."
make clean

echo "Demo completed!"
echo
echo "The enhanced CPU stress test provides several advantages for testing JFR CPU sampling:"
echo "1. Adjustable stack depths (10-500+ levels)"
echo "2. Dynamic class generation at runtime"
echo "3. Mixed Java/native execution"
echo "4. Variable computation loads per thread"
echo "5. Automatic cleanup of generated classes"
echo
echo "This should provide a much more challenging workload for the JFR CPU time sampler!"
