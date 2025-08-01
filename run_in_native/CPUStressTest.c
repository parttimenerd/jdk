#include <jni.h>
#include <time.h>
#include <math.h>

JNIEXPORT void JNICALL Java_CPUStressTest_work(JNIEnv *env, jobject obj, jint id, jint duration) {
    time_t end = time(NULL) + duration;
    volatile double x = id + 1.0;

    while (time(NULL) < end) {
        for (int i = 0; i < 100000; i++) {
            x = sin(x) + cos(x * 1.1) + sqrt(fabs(x));
            if (x > 1000) x = 1.0;
        }
    }
}