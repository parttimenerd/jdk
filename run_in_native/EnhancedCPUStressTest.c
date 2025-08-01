#include <jni.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration for recursive native function
static void recursive_work(int depth, int max_depth, volatile double *x);

/**
 * Enhanced native work function that takes stack depth into account
 */
JNIEXPORT void JNICALL Java_EnhancedCPUStressTest_work(JNIEnv *env, jobject obj, jint id, jint duration, jint stackDepth) {
    time_t end = time(NULL) + duration;
    volatile double x = id + 1.0;

    // Allocate some stack-local data to make frames more interesting
    char stack_buffer[1024];
    snprintf(stack_buffer, sizeof(stack_buffer), "Thread_%d_Depth_%d", id, stackDepth);

    while (time(NULL) < end) {
        // Do some intensive computation
        for (int i = 0; i < 50000; i++) {
            x = sin(x) + cos(x * 1.1) + sqrt(fabs(x));
            if (x > 1000) x = 1.0;

            // Occasionally do recursive native calls to build more native stack depth
            if (i % 10000 == 0 && stackDepth > 0) {
                recursive_work(0, stackDepth / 2, &x);
            }
        }

        // Use the stack buffer to prevent optimization
        stack_buffer[0] = (char)(((int)x) % 256);
    }
}

/**
 * Recursive native function to build additional native stack depth
 */
static void recursive_work(int depth, int max_depth, volatile double *x) {
    if (depth >= max_depth) {
        // Base case: do some computation
        for (int i = 0; i < 1000; i++) {
            *x = sin(*x) + cos(*x * 1.3) + log(fabs(*x) + 1.0);
            if (*x > 1000) *x = 1.0;
        }
        return;
    }

    // Allocate some stack space to make the frame more substantial
    volatile double local_array[100];
    for (int i = 0; i < 100; i++) {
        local_array[i] = *x + i * 0.01;
    }

    // Some computation at this level
    for (int i = 0; i < 100; i++) {
        *x += sin(local_array[i % 100]) * 0.001;
    }

    // Recurse deeper
    recursive_work(depth + 1, max_depth, x);

    // More computation on the way back up
    for (int i = 0; i < 50; i++) {
        *x += cos(local_array[i % 100]) * 0.001;
    }
}

// Keep the original work function for backward compatibility
JNIEXPORT void JNICALL Java_CPUStressTest_work(JNIEnv *env, jobject obj, jint id, jint duration) {
    // Call the enhanced version with default stack depth
    Java_EnhancedCPUStressTest_work(env, obj, id, duration, 5);
}
