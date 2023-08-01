/*
 * Copyright (c) 2022, SAP SE. All rights reserved.
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

#include "profile2.h"
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <array>
#include <chrono>
#include <cerrno>
#include <signal.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ucontext.h>
#include <ucontext.h>
#include <utility>
#include "jni.h"
#include "jvmti.h"

#ifdef _GNU_SOURCE
#include <dlfcn.h>
#endif

#ifdef DEBUG
// more space for debug info
const int METHOD_HEADER_SIZE = 0x200;
const int METHOD_PRE_HEADER_SIZE = 0x20;
#else
const int METHOD_HEADER_SIZE = 0x100;
const int METHOD_PRE_HEADER_SIZE = 0x10;
#endif

static jvmtiEnv* jvmti;

typedef void (*SigAction)(int, siginfo_t*, void*);
typedef void (*SigHandler)(int);
typedef void (*TimerCallback)(void*);


template <class T>
class JvmtiDeallocator {
 public:
  JvmtiDeallocator() {
    elem_ = NULL;
  }

  ~JvmtiDeallocator() {
    jvmti->Deallocate(reinterpret_cast<unsigned char*>(elem_));
  }

  T* get_addr() {
    return &elem_;
  }

  T get() {
    return elem_;
  }

 private:
  T elem_;
};

static void GetJMethodIDs(jclass klass) {
  jint method_count = 0;
  JvmtiDeallocator<jmethodID*> methods;
  jvmtiError err = jvmti->GetClassMethods(klass, &method_count, methods.get_addr());

  // If ever the GetClassMethods fails, just ignore it, it was worth a try.
  if (err != JVMTI_ERROR_NONE && err != JVMTI_ERROR_CLASS_NOT_PREPARED) {
    fprintf(stderr, "GetJMethodIDs: Error in GetClassMethods: %d\n", err);
  }
}

void printMethod(FILE* stream, jmethodID method) {
  JvmtiDeallocator<char*> name;
  JvmtiDeallocator<char*> signature;
  jvmtiError err = jvmti->GetMethodName(method, name.get_addr(), signature.get_addr(), NULL);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stream, "Error in GetMethodName: %d", err);
    return;
  }
  jclass klass;
  JvmtiDeallocator<char*> className;
  jvmti->GetMethodDeclaringClass(method, &klass);
  jvmti->GetClassSignature(klass, className.get_addr(), NULL);
  fprintf(stream, "%s.%s%s", className.get(), name.get(), signature.get());
}

void printMethod(FILE *stream, ASGST_Method method) {
  if (method == NULL) {
    fprintf(stream, "NULL");
    return;
  }
  char method_name[100];
  char signature[100];
  ASGST_MethodInfo info;
  info.method_name = (char*)method_name;
  info.method_name_length = 100;
  info.signature = (char*)signature;
  info.signature_length = 100;
  info.generic_signature = nullptr;
  ASGST_GetMethodInfo(method, &info);
  ASGST_ClassInfo class_info;
  char class_name[100];
  class_info.class_name = (char*)class_name;
  class_info.class_name_length = 100;
  class_info.generic_class_name = nullptr;
  ASGST_GetClassInfo(info.klass, &class_info);
  fprintf(stream, "%s.%s%s", class_name, method_name, signature);
}

long getSecondsSinceEpoch(){
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

typedef struct {
    jint lineno;                      // line number in the source file
    jmethodID method_id;              // method executed in this frame
} ASGCT_CallFrame;

typedef struct {
    JNIEnv *env_id;                   // Env where trace was recorded
    jint num_frames;                  // number of frames in this trace
    ASGCT_CallFrame *frames;          // frames
} ASGCT_CallTrace;

typedef void (*ASGCTType)(ASGCT_CallTrace *, jint, void *, bool);

static ASGCTType asgct = nullptr;


bool isASGCTNativeFrame(ASGCT_CallFrame frame) {
  return frame.lineno == -3;
}

void printASGCTFrame(FILE* stream, ASGCT_CallFrame frame) {
  JvmtiDeallocator<char*> name;
  jvmtiError err = jvmti->GetMethodName(frame.method_id, name.get_addr(), NULL, NULL);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stream, "=== asgst sampler failed: Error in GetMethodName: %d", err);
    return;
  }
  if (isASGCTNativeFrame(frame)) {
    fprintf(stream, "Native frame ");
    printMethod(stream, frame.method_id);
  } else {
    fprintf(stream, "Java frame   ");
    printMethod(stream, frame.method_id);
    fprintf(stream, ": %d", frame.lineno);
  }
}

void printASGCTFrames(FILE* stream, ASGCT_CallFrame *frames, int length) {
  for (int i = 0; i < length; i++) {
    fprintf(stream, "Frame %d: ", i);
    printASGCTFrame(stream, frames[i]);
    fprintf(stream, "\n");
  }
}

void printASGCTTrace(FILE* stream, ASGCT_CallTrace trace) {
  fprintf(stream, "ASGCT Trace length: %d\n", trace.num_frames);
  if (trace.num_frames > 0) {
    printASGCTFrames(stream, trace.frames, trace.num_frames);
  }
  fprintf(stream, "ASGCT Trace end\n");
}

void printGSTFrame(FILE* stream, jvmtiFrameInfo frame) {
  if (frame.location == -1) {
    fprintf(stream, "Native frame");
    printMethod(stream, frame.method);
  } else {
    fprintf(stream, "Java frame   ");
    printMethod(stream, frame.method);
    fprintf(stream, ": %d", frame.location);
  }
}


void printGSTTrace(FILE* stream, jvmtiFrameInfo* frames, int length) {
  fprintf(stream, "GST Trace length: %d\n", length);
  for (int i = 0; i < length; i++) {
    fprintf(stream, "Frame %d: ", i);
    printGSTFrame(stream, frames[i]);
    fprintf(stream, "\n");
  }
  fprintf(stream, "GST Trace end\n");
}

/** should be called in the agent load method*/
void initASGCT() {
  if (asgct != nullptr) {
    return;
  }
  void *mptr = dlsym((void*)-2, "AsyncGetCallTrace");
  if (mptr == nullptr) {
    fprintf(stderr, "Error: could not find AsyncGetCallTrace!\n");
    exit(0);
  }
  asgct = reinterpret_cast<ASGCTType>(mptr);
}

JNIEnv* env;

template<int max_depth> void printASGCT(void* ucontext, JNIEnv* oenv = nullptr) {
  assert(env != nullptr || oenv != nullptr);
  ASGCT_CallTrace asgct_trace;
  static ASGCT_CallFrame asgct_frames[max_depth];
  asgct_trace.frames = asgct_frames;
  asgct_trace.env_id = oenv == nullptr ? env : oenv;
  asgct_trace.num_frames = 0;

  asgct(&asgct_trace, max_depth, ucontext, false);
  printASGCTTrace(stderr, asgct_trace);
}

template<int max_depth> bool printASGCTOnSuccess(void* ucontext, JNIEnv* oenv = nullptr) {
  assert(env != nullptr || oenv != nullptr);
  ASGCT_CallTrace asgct_trace;
  static ASGCT_CallFrame asgct_frames[max_depth];
  asgct_trace.frames = asgct_frames;
  asgct_trace.env_id = oenv == nullptr ? env : oenv;
  asgct_trace.num_frames = 0;

  asgct(&asgct_trace, max_depth, ucontext, false);
  if (asgct_trace.num_frames <= 0) {
    return false;
  }
  printASGCTTrace(stderr, asgct_trace);
  return true;
}

template<int max_depth> void printGST() {
  jthread thread;
  jvmti->GetCurrentThread(&thread);
  jvmtiFrameInfo gstFrames[max_depth];
  jint gstCount = 0;
  jvmti->GetStackTrace(thread, 0, max_depth, gstFrames, &gstCount);
  printGSTTrace(stderr, gstFrames, gstCount);
}