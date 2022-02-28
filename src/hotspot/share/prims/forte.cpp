/*
 * Copyright (c) 2003, 2022, Oracle and/or its affiliates. All rights reserved.
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
 *
 */

#include "precompiled.hpp"
#include "code/debugInfoRec.hpp"
#include "code/pcDesc.hpp"
#include "gc/shared/collectedHeap.inline.hpp"
#include "memory/universe.hpp"
#include "oops/oop.inline.hpp"
#include "prims/forte.hpp"
#include "prims/jvmtiExport.hpp"
#include "runtime/frame.inline.hpp"
#include "runtime/javaCalls.hpp"
#include "runtime/thread.inline.hpp"
#include "runtime/vframe.inline.hpp"
#include "runtime/vframeArray.hpp"
#include "runtime/vframe_hp.hpp"
#include "compiler/compilerDefinitions.hpp"
#include "jfr/recorder/stacktrace/stackWalker.hpp"

// call frame copied from old .h file and renamed
typedef struct {
    jint lineno;                      // line number in the source file
    jmethodID method_id;              // method executed in this frame
} ASGCT_CallFrame;

// call trace copied from old .h file and renamed
typedef struct {
    JNIEnv *env_id;                   // Env where trace was recorded
    jint num_frames;                  // number of frames in this trace
    ASGCT_CallFrame *frames;          // frames
} ASGCT_CallTrace;

// These name match the names reported by the forte quality kit
enum {
  ticks_no_Java_frame         =  0,
  ticks_no_class_load         = -1,
  ticks_GC_active             = -2,
  ticks_unknown_not_Java      = -3,
  ticks_not_walkable_not_Java = -4,
  ticks_unknown_Java          = -5,
  ticks_not_walkable_Java     = -6,
  ticks_unknown_state         = -7,
  ticks_thread_exit           = -8,
  ticks_deopt                 = -9,
  ticks_safepoint             = -10
};

#if INCLUDE_JVMTI

//-------------------------------------------------------

// Native interfaces for use by Forte tools.


#if !defined(IA64)

class ForteStackWalker : private StackWalker {

  ASGCT_CallTrace* trace;
  const int max_depth;
  int frame_count;

  void handle_compiled_frame(const frame& base_frame, Method* method, int bci, bool inlined) {
    handle(method, bci);
  }

  void handle_interpreted_frame(const frame& frame, Method* method, int bci) {
    handle(method, bci);
  }

  void handle(Method* method, int bci) {
    trace->frames[frame_count].method_id = method->find_jmethod_id_or_null();
    if (!method->is_native()) {
      trace->frames[frame_count].lineno = bci;
    } else {
      trace->frames[frame_count].lineno = -3;
    }
    frame_count++;
  }

  bool abort() const { return frame_count < max_depth; }

public:
  ForteStackWalker(JavaThread* thd, ASGCT_CallTrace* trace, int max_depth):
    StackWalker(thd), trace(trace), max_depth(max_depth), frame_count(0) {}

  void fill_given_top(frame top_frame) {
    NoHandleMark nhm;
    int initial_frame_err = 0;
    frame initial_Java_frame = find_top_java_frame(top_frame, max_depth * 2, &initial_frame_err);

    assert(frame_count == 0, "do not call this method twice on the same object");
    assert(trace->frames != NULL, "trace->frames must be non-NULL");

    if (initial_frame_err != 0) {
      trace->num_frames = initial_frame_err; // either GC active or no_class_load
      return;
    }

    int ret = walk(top_frame);
    if (ret < 0) {
      trace->num_frames = ret;
    }
  }
};


// Forte Analyzer AsyncGetCallTrace() entry point. Currently supported
// on Linux X86, Solaris SPARC and Solaris X86.
//
// Async-safe version of GetCallTrace being called from a signal handler
// when a LWP gets interrupted by SIGPROF but the stack traces are filled
// with different content (see below).
//
// This function must only be called when JVM/TI
// CLASS_LOAD events have been enabled since agent startup. The enabled
// event will cause the jmethodIDs to be allocated at class load time.
// The jmethodIDs cannot be allocated in a signal handler because locks
// cannot be grabbed in a signal handler safely.
//
// void (*AsyncGetCallTrace)(ASGCT_CallTrace *trace, jint depth, void* ucontext)
//
// Called by the profiler to obtain the current method call stack trace for
// a given thread. The thread is identified by the env_id field in the
// ASGCT_CallTrace structure. The profiler agent should allocate a ASGCT_CallTrace
// structure with enough memory for the requested stack depth. The VM fills in
// the frames buffer and the num_frames field.
//
// Arguments:
//
//   trace    - trace data structure to be filled by the VM.
//   depth    - depth of the call stack trace.
//   ucontext - ucontext_t of the LWP
//
// ASGCT_CallTrace:
//   typedef struct {
//       JNIEnv *env_id;
//       jint num_frames;
//       ASGCT_CallFrame *frames;
//   } ASGCT_CallTrace;
//
// Fields:
//   env_id     - ID of thread which executed this trace.
//   num_frames - number of frames in the trace.
//                (< 0 indicates the frame is not walkable).
//   frames     - the ASGCT_CallFrames that make up this trace. Callee followed by callers.
//
//  ASGCT_CallFrame:
//    typedef struct {
//        jint lineno;
//        jmethodID method_id;
//    } ASGCT_CallFrame;
//
//  Fields:
//    1) For Java frame (interpreted and compiled),
//       lineno    - bci of the method being executed or -1 if bci is not available
//       method_id - jmethodID of the method being executed
//    2) For native method
//       lineno    - (-3)
//       method_id - jmethodID of the method being executed

extern "C" {
JNIEXPORT
void AsyncGetCallTrace(ASGCT_CallTrace *trace, jint depth, void* ucontext) {
    if (trace->env_id == NULL || JavaThread::is_thread_from_jni_environment_terminated(trace->env_id)) {
    // bad env_id, thread has exited or thread is exiting
    trace->num_frames = ticks_thread_exit; // -8
    return;
  }

  JavaThread* thread = JavaThread::thread_from_jni_environment(trace->env_id);


  if (thread->in_deopt_handler()) {
    // thread is in the deoptimization handler so return no frames
    trace->num_frames = ticks_deopt; // -9
    return;
  }

  assert(JavaThread::current() == thread,
         "AsyncGetCallTrace must be called by the current interrupted thread");

  if (!JvmtiExport::should_post_class_load()) {
    trace->num_frames = ticks_no_class_load; // -1
    return;
  }

  if (Universe::heap()->is_gc_active()) {
    trace->num_frames = ticks_GC_active; // -2
    return;
  }

  switch (thread->thread_state()) {
  case _thread_new:
  case _thread_uninitialized:
  case _thread_new_trans:
    // We found the thread on the threads list above, but it is too
    // young to be useful so return that there are no Java frames.
    trace->num_frames = 0;
    break;
  case _thread_in_native:
  case _thread_in_native_trans:
  case _thread_blocked:
  case _thread_blocked_trans:
  case _thread_in_vm:
  case _thread_in_vm_trans:
    {
      frame fr;

      // param isInJava == false - indicate we aren't in Java code
      if (!thread->pd_get_top_frame_for_signal_handler(&fr, ucontext, false)) {
        trace->num_frames = ticks_unknown_not_Java;  // -3 unknown frame
      } else {
        if (!thread->has_last_Java_frame()) {
          trace->num_frames = 0; // No Java frames
        } else {
          trace->num_frames = ticks_not_walkable_not_Java;    // -4 non walkable frame by default
          ForteStackWalker(thread, trace, depth).fill_given_top(fr);

          // This assert would seem to be valid but it is not.
          // It would be valid if we weren't possibly racing a gc
          // thread. A gc thread can make a valid interpreted frame
          // look invalid. It's a small window but it does happen.
          // The assert is left here commented out as a reminder.
          // assert(trace->num_frames != ticks_not_walkable_not_Java, "should always be walkable");

        }
      }
    }
    break;
  case _thread_in_Java:
  case _thread_in_Java_trans:
    {
      frame fr;

      // param isInJava == true - indicate we are in Java code
      if (!thread->pd_get_top_frame_for_signal_handler(&fr, ucontext, true)) {
        trace->num_frames = ticks_unknown_Java;  // -5 unknown frame
      } else {
        trace->num_frames = ticks_not_walkable_Java;  // -6, non walkable frame by default
        ForteStackWalker(thread, trace, depth).fill_given_top(fr);
      }
    }
    break;
  default:
    // Unknown thread state
    trace->num_frames = ticks_unknown_state; // -7
    break;
  }
}


#ifndef _WINDOWS
// Support for the Forte(TM) Peformance Tools collector.
//
// The method prototype is derived from libcollector.h. For more
// information, please see the libcollect man page.

// Method to let libcollector know about a dynamically loaded function.
// Because it is weakly bound, the calls become NOP's when the library
// isn't present.
#ifdef __APPLE__
// XXXDARWIN: Link errors occur even when __attribute__((weak_import))
// is added
#define collector_func_load(x0,x1,x2,x3,x4,x5,x6) ((void) 0)
#else
void    collector_func_load(char* name,
                            void* null_argument_1,
                            void* null_argument_2,
                            void *vaddr,
                            int size,
                            int zero_argument,
                            void* null_argument_3);
#pragma weak collector_func_load
#define collector_func_load(x0,x1,x2,x3,x4,x5,x6) \
        ( collector_func_load ? collector_func_load(x0,x1,x2,x3,x4,x5,x6),(void)0 : (void)0 )
#endif // __APPLE__
#endif // !_WINDOWS

} // end extern "C"
#endif // !IA64

void Forte::register_stub(const char* name, address start, address end) {
#if !defined(_WINDOWS) && !defined(IA64)
  assert(pointer_delta(end, start, sizeof(jbyte)) < INT_MAX,
         "Code size exceeds maximum range");

  collector_func_load((char*)name, NULL, NULL, start,
    pointer_delta(end, start, sizeof(jbyte)), 0, NULL);
#endif // !_WINDOWS && !IA64
}

#else // INCLUDE_JVMTI
extern "C" {
  JNIEXPORT
  void AsyncGetCallTrace(ASGCT_CallTrace *trace, jint depth, void* ucontext) {
    trace->num_frames = ticks_no_class_load; // -1
  }
}
#endif // INCLUDE_JVMTI
