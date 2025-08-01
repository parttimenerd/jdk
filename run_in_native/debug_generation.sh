#!/bin/bash

# Debug version of the generation logic
STACK_DEPTH=2
THREADS=1
CACHE_DIR="build/cache_s${STACK_DEPTH}_t${THREADS}"

echo "Setting up debug test..."
rm -rf build
mkdir -p "$CACHE_DIR"

generate_thread_classes() {
    local thread_id=$1
    local stack_depth=$2
    local cache_dir=$3

    echo "DEBUG: generate_thread_classes called with: thread_id=$thread_id, stack_depth=$stack_depth, cache_dir=$cache_dir" >&2

    for ((i=1; i<=stack_depth; i++)); do
        local class_name="DynamicStressClass${thread_id}_${i}"
        local java_file="$cache_dir/${class_name}.java"

        echo "DEBUG: Creating $java_file" >&2

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

    echo "DEBUG: generate_thread_classes completed for thread $thread_id" >&2
}

echo "Starting generation loop..."
for ((t=0; t<THREADS; t++)); do
    echo "DEBUG: Starting background job for thread $t"
    generate_thread_classes $t $STACK_DEPTH "$CACHE_DIR" &
    echo "DEBUG: Background job $t started, PID: $!"
done

echo "DEBUG: Waiting for background jobs..."
wait
echo "DEBUG: All background jobs completed"

echo "Checking generated files:"
ls -la "$CACHE_DIR"
