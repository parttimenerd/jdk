#include <jni.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

/**
 * Native implementation for DynamicStackCPUStressTest
 * This performs intensive CPU work to stress the system
 */
JNIEXPORT void JNICALL Java_DynamicStackCPUStressTest_work(JNIEnv *env, jobject obj, jint threadId, jint duration) {
    time_t startTime = time(NULL);
    time_t endTime = startTime + duration;
    volatile double x = threadId + 1.0;
    volatile double y = threadId * 2.0 + 1.0;
    volatile long iterations = 0;

    printf("Thread %d: Starting native work for %d seconds\n", threadId, duration);
    fflush(stdout);

    while (time(NULL) < endTime) {
        // More intensive mathematical operations to stress CPU
        for (int i = 0; i < 100000; i++) {  // Increased iterations for more work
            x = sin(x) + cos(x * 1.1) + sqrt(fabs(x));
            y = tan(y * 0.9) + log(fabs(y) + 1.0) + exp(y * 0.01);

            // Prevent overflow
            if (x > 1000.0) x = 1.0;
            if (y > 1000.0) y = 1.0;
            if (x < -1000.0) x = -1.0;
            if (y < -1000.0) y = -1.0;

            // Some additional work to make stack sampling more interesting
            volatile double temp = x * y;
            temp = pow(fabs(temp), 0.1);
            x += temp * 0.001;

            iterations++;
        }

        // Check time more frequently to ensure precise duration
        if (iterations % 10 == 0) {
            time_t currentTime = time(NULL);
            if (currentTime >= endTime) {
                break;
            }
        }
    }

    printf("Thread %d: Completed native work after %ld iterations\n", threadId, iterations);
    fflush(stdout);
}
