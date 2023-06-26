/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) 2019, Google and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

#include <assert.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include "jni.h"
#include "jvmti.h"
#include "util.hpp"
#include <atomic>

#include "profile2.h"

#ifdef DEBUG
const int INTERVAL_NS = 20000; // 20us, 50 000 times per second
                               // 1000 times more than in a normal profiling run
#else
const int INTERVAL_NS = 1000; // 1us, 1 0000 000 times per second
                              // 20000 times more than in a normal profiling run
#endif

// AsyncGetStackTrace needs class loading events to be turned on!
static void JNICALL OnClassLoad(jvmtiEnv *jvmti, JNIEnv *jni_env,
                                jthread thread, jclass klass) {
}

static void JNICALL OnClassPrepare(jvmtiEnv *jvmti, JNIEnv *jni_env,
                                   jthread thread, jclass klass) {
  // We need to do this to "prime the pump" and get jmethodIDs primed.
  GetJMethodIDs(klass);
}

thread_local ASGST_Queue *queue;

static void asgstHandler(ASGST_Iterator* iterator, void* queueArg, void* arg) {
  printf("Hi from inside the queue with state %d\n", ASGST_State(iterator));
  ASGST_Frame frame;
  ASGST_NextFrame(iterator, &frame);
}

thread_local JNIEnv* _jni_env;

static void JNICALL OnThreadStart(jvmtiEnv *jvmti, JNIEnv *jni_env, jthread thread) {
  queue = ASGST_RegisterQueue(jni_env, 100, 0, &asgstHandler, nullptr);
  _jni_env = jni_env;
}

static SigAction installSignalHandler(int signo, SigAction action, SigHandler handler = NULL) {
    struct sigaction sa;
    struct sigaction oldsa;
    sigemptyset(&sa.sa_mask);

    if (handler != NULL) {
        sa.sa_handler = handler;
        sa.sa_flags = 0;
    } else {
        sa.sa_sigaction = action;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
    }

    sigaction(signo, &sa, &oldsa);
    return oldsa.sa_sigaction;
}

std::atomic<int> counter = {0};

static void signalHandler(int signo, siginfo_t* siginfo, void* ucontext) {
  if (_jni_env == nullptr) {
    return;
  }
  ASGCT_CallFrame frames[10];
  ASGCT_CallTrace trace;
  trace.frames = frames;
  trace.env_id = _jni_env;
  asgct(&trace, 2, ucontext, false);
  if (trace.num_frames <= 0) {
    return;
  }
  ASGST_Enqueue(queue, ucontext, (void*)trace.env_id);
}

static bool startITimerSampler(long interval_ns) {
  time_t sec = interval_ns / 1000000000;
  suseconds_t usec = (interval_ns % 1000000000) / 1000;
  struct itimerval tv = {{sec, usec}, {sec, usec}};

  installSignalHandler(SIGPROF, signalHandler);

  if (setitimer(ITIMER_PROF, &tv, NULL) != 0) {
    return false;
  }
  fprintf(stderr, "=== asgst sampler initialized ===\n");
  return true;
}

static void JNICALL OnVMInit(jvmtiEnv *jvmti, JNIEnv *jni_env, jthread thread) {
  fprintf(stderr, "=== asgst OnVMInit ===\n");
  jint class_count = 0;

  // Get any previously loaded classes that won't have gone through the
  // OnClassPrepare callback to prime the jmethods for AsyncGetStackTrace.
  JvmtiDeallocator<jclass*> classes;
  jvmtiError err = jvmti->GetLoadedClasses(&class_count, classes.get_addr());
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "OnVMInit: Error in GetLoadedClasses: %d\n", err);
    return;
  }

  // Prime any class already loaded and try to get the jmethodIDs set up.
  jclass *classList = classes.get();
  for (int i = 0; i < class_count; ++i) {
    GetJMethodIDs(classList[i]);
  }

  startITimerSampler(INTERVAL_NS);
}

void ensureSuccess(jvmtiError err, const char* message) {
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "%s: %d\n", message, err);
    exit(1);
  }
}

extern "C" {

static
jint Agent_Initialize(JavaVM *jvm, char *options, void *reserved) {
  printf("Agent_Initialize\n");
  jint res = jvm->GetEnv((void **) &jvmti, JVMTI_VERSION);
  if (res != JNI_OK || jvmti == NULL) {
    fprintf(stderr, "Error: wrong result of a valid call to GetEnv!\n");
    return JNI_ERR;
  }

  jvmtiError err;
  jvmtiCapabilities caps;
  memset(&caps, 0, sizeof(caps));
  caps.can_get_line_numbers = 1;
  caps.can_get_source_file_name = 1;

  err = jvmti->AddCapabilities(&caps);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "AgentInitialize: Error in AddCapabilities: %d\n", err);
    return JNI_ERR;
  }

  jvmtiEventCallbacks callbacks;
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.ClassLoad = &OnClassLoad;
  callbacks.VMInit = &OnVMInit;
  callbacks.ClassPrepare = &OnClassPrepare;
  callbacks.ThreadStart = &OnThreadStart;

  ensureSuccess(jvmti->SetEventCallbacks(&callbacks, sizeof(jvmtiEventCallbacks)),
      "AgentInitialize: Error in SetEventCallbacks");

  ensureSuccess(jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_CLASS_LOAD, NULL),
      "AgentInitialize: Error in SetEventNotificationMode for CLASS_LOAD");

  ensureSuccess(jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_CLASS_PREPARE, NULL),
      "AgentInitialize: Error in SetEventNotificationMode for CLASS_PREPARE");

  ensureSuccess(jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_VM_INIT, NULL),
      "AgentInitialize: Error in SetEventNotificationMode for VM_INIT");

  ensureSuccess(jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_THREAD_START, NULL),
      "AgentInitialize: Error in SetEventNotificationMode for THREAD_START");
  initASGCT();
  return JNI_OK;
}

JNIEXPORT
jint JNICALL Agent_OnLoad(JavaVM *jvm, char *options, void *reserved) {
  return Agent_Initialize(jvm, options, reserved);
}

JNIEXPORT
jint JNICALL Agent_OnAttach(JavaVM *jvm, char *options, void *reserved) {
  return Agent_Initialize(jvm, options, reserved);
}

JNIEXPORT
jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
  return JNI_VERSION_1_8;
}

}
