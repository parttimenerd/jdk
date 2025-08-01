# Makefile for CPU Stress Test (Original and Enhanced versions)

# Java and JNI settings
JAVA_HOME ?= $(shell readlink -f /usr/bin/javac | sed "s:bin/javac::")
JNI_INCLUDES = -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux

# Compiler settings
CC = gcc
CFLAGS = -fPIC -shared -O2 -g $(JNI_INCLUDES)
LIBS = -lm

# Source files
JAVA_SOURCES = CPUStressTest.java EnhancedCPUStressTest.java
C_SOURCES = CPUStressTest.c EnhancedCPUStressTest.c
HEADERS = CPUStressTest.h EnhancedCPUStressTest.h

# Output
NATIVE_LIB = libcpustress.so
CLASS_FILES = $(JAVA_SOURCES:.java=.class)

.PHONY: all clean test test-enhanced

all: $(NATIVE_LIB) $(CLASS_FILES)

# Compile Java sources
%.class: %.java
	javac $<

# Generate JNI headers (if needed)
%.h: %.class
	javah -jni $(basename $<)

# Build native library
$(NATIVE_LIB): $(C_SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(C_SOURCES) $(LIBS)

# Test targets
test: all
	@echo "Running original CPU stress test for 5 seconds..."
	java -Djava.library.path=. CPUStressTest 5

test-enhanced: all
	@echo "Running enhanced CPU stress test..."
	@echo "Usage: java -Djava.library.path=. EnhancedCPUStressTest <duration> <threads> <min_stack> <max_stack> <use_native>"
	@echo "Running with default parameters (10s, auto threads, stack 10-50, native enabled)..."
	java -Djava.library.path=. EnhancedCPUStressTest 10

test-enhanced-deep: all
	@echo "Running enhanced CPU stress test with deep stacks..."
	java -Djava.library.path=. EnhancedCPUStressTest 15 4 50 200 true

test-enhanced-java-only: all
	@echo "Running enhanced CPU stress test (Java only, no native)..."
	java -Djava.library.path=. EnhancedCPUStressTest 10 4 20 80 false

# JFR profiling targets
profile-original: all
	@echo "Profiling original CPU stress test with JFR..."
	java -XX:+FlightRecorder \
		 -XX:StartFlightRecording=duration=15s,filename=cpu_stress_original.jfr \
		 -Djava.library.path=. CPUStressTest 10

profile-enhanced: all
	@echo "Profiling enhanced CPU stress test with JFR..."
	java -XX:+FlightRecorder \
		 -XX:StartFlightRecording=duration=20s,filename=cpu_stress_enhanced.jfr \
		 -Djava.library.path=. EnhancedCPUStressTest 15 4 30 100 true

profile-enhanced-extreme: all
	@echo "Profiling enhanced CPU stress test with extreme parameters..."
	java -XX:+FlightRecorder \
		 -XX:StartFlightRecording=duration=30s,filename=cpu_stress_extreme.jfr \
		 -Djava.library.path=. EnhancedCPUStressTest 25 8 100 500 true

# Clean up
clean:
	rm -f $(CLASS_FILES) $(NATIVE_LIB) *.jfr
	rm -rf /tmp/jfr_stress_test/
	find . -name "DynamicStressClass*" -delete

# Help
help:
	@echo "Available targets:"
	@echo "  all                    - Build everything"
	@echo "  test                   - Run original CPU stress test"
	@echo "  test-enhanced          - Run enhanced CPU stress test (default params)"
	@echo "  test-enhanced-deep     - Run enhanced test with deep call stacks"
	@echo "  test-enhanced-java-only - Run enhanced test without native code"
	@echo "  profile-original       - Profile original test with JFR"
	@echo "  profile-enhanced       - Profile enhanced test with JFR"
	@echo "  profile-enhanced-extreme - Profile with extreme parameters"
	@echo "  clean                  - Clean up generated files"
	@echo "  help                   - Show this help"
	@echo ""
	@echo "Enhanced test parameters:"
	@echo "  java -Djava.library.path=. EnhancedCPUStressTest <duration> <threads> <min_stack> <max_stack> <use_native>"
	@echo "  - duration: seconds to run (default: 10)"
	@echo "  - threads: number of threads (default: CPU cores)"
	@echo "  - min_stack: minimum stack depth (default: 10)"
	@echo "  - max_stack: maximum stack depth (default: 50)"
	@echo "  - use_native: true/false for native code (default: true)"
