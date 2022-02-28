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

#include "gc/shared/collectedHeap.inline.hpp"
#include "memory/universe.hpp"
#include "oops/oop.inline.hpp"
#include "prims/jvmtiExport.hpp"
#include "runtime/frame.inline.hpp"
#include "runtime/thread.inline.hpp"
#include "runtime/vframe.inline.hpp"
#include "runtime/vframeArray.hpp"
#include "runtime/vframe_hp.hpp"
#include "compiler/compilerDefinitions.hpp"
#include "jfr/recorder/stacktrace/stackWalker.hpp"

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

typedef struct {
   jint bci;
   jmethodID method_id;
   void *machinepc;
  // first one bytes: FrameTypeId, then one byte for the CompLevel
   int16_t type;
} ASGCT_CallFrame2;

enum FrameTypeId {
    FRAME_INTERPRETED  = 0,
    FRAME_JIT_COMPILED = 1,
    FRAME_INLINED      = 2,
    FRAME_NATIVE       = 3,
    FRAME_CPP          = 4,
    FRAME_KERNEL       = 5
};

// Enumeration to distinguish tiers of compilation, only >= 0 are used
/*enum CompLevel {
  CompLevel_any               = -1,        // Used for querying the state
  CompLevel_all               = -1,        // Used for changing the state
  CompLevel_none              = 0,         // Interpreter
  CompLevel_simple            = 1,         // C1
  CompLevel_limited_profile   = 2,         // C1, invocation & backedge counters
  CompLevel_full_profile      = 3,         // C1, invocation & backedge counters + mdo
  CompLevel_full_optimization = 4          // C2 or JVMCI
};*/

int16_t encode_type(int frame_type, int comp_level) {
  return frame_type + (comp_level << 8);
}


typedef struct {
    JNIEnv *env_id;                   // Env where trace was recorded
    jint num_frames;                  // number of frames in this trace
    ASGCT_CallFrame2 *frames;          // frames
} ASGCT_CallTrace2;


class ASGCT2StackWalker : private StackWalker {

  ASGCT_CallTrace2* trace;
  const int max_depth;
  int frame_count;

  void handle_interpreted_frame(const frame& frame, Method* method, int bci) {
    //printf("interpreted frame %s\n", method->name_and_sig_as_C_string());
    trace->frames[frame_count] = {
      bci,
      method->find_jmethod_id_or_null(),
      0,
      encode_type(FRAME_INTERPRETED, CompLevel_none)
    };
    frame_count ++;
  }

  void handle_compiled_frame(const frame& base_frame, Method* method, int bci, bool inlined) {
    //printf("compiled frame %s %s\n", method->name_and_sig_as_C_string(), inlined ? "inlined" : "");
    int16_t type = encode_type(inlined ? FRAME_INLINED : FRAME_JIT_COMPILED, method->highest_comp_level());
    trace->frames[frame_count] = {
      bci,
      method->find_jmethod_id_or_null(),
      0,
      type
    };
    frame_count++;
  }

  void handle_misc_frame(const frame& frame) {
    //printf("misc frame\n");
    bool is_native_frame = frame.is_native_frame();
    trace->frames[frame_count] = {
      -4,
      0,
      frame.pc(),
      encode_type(is_native_frame ? FRAME_NATIVE : FRAME_CPP, frame.is_stub_frame() ? CompLevel_all : CompLevel_none)
    };
  }

  bool abort() const { return frame_count < max_depth; }

public:
  ASGCT2StackWalker(JavaThread* thd, ASGCT_CallTrace2* trace, int max_depth):
    StackWalker(thd), trace(trace), max_depth(max_depth), frame_count(0) {}

  void fill_given_top(frame top_frame) {
    NoHandleMark nhm;
    assert(frame_count == 0, "do not call this method twice on the same object");
    assert(trace->frames != NULL, "trace->frames must be non-NULL");

    trace->num_frames = walk(top_frame);
  }
};

// AsyncGetCallTrace2() entry point.
//
// Extension of AsyncGetCallTrace()
//
// void (*AsyncGetCallTrace2)(ASGCT_CallTrace2 *trace, jint depth, void* ucontext)
//
// Called by the profiler to obtain the current method call stack trace for
// a given thread. The thread is identified by the env_id field in the
// ASGCT_CallTrace2 structure. The profiler agent should allocate a ASGCT_CallTrace2
// structure with enough memory for the requested stack depth. The VM fills in
// the frames buffer and the num_frames field.
//
// Arguments:
//
//   trace    - trace data structure to be filled by the VM.
//   depth    - depth of the call stack trace.
//   ucontext - ucontext_t of the LWP
//
// ASGCT_CallTrace2:
//   typedef struct {
//       JNIEnv *env_id;
//       jint num_frames;
//       ASGCT_CallFrame2 *frames;
//   } ASGCT_CallTrace2;
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
//        void *machinepc;
//        jint type;
//        void *nativeFrameAddress;
//    } ASGCT_CallFrame2;
//
//  Fields:
//    1) For Java frame (interpreted and compiled),
//       lineno    - bci of the method being executed or -1 if bci is not available
//       method_id - jmethodID of the method being executed
//    2) For native method
//       lineno    - (-3)
//       method_id - jmethodID of the method being executed
//       type      - (0)
//    3) For native frame
//       lineno    - (-4)
//       method_id - (0)
//       type      - (0)
//    4) For all frames
//       machinepc - pc of the called method
//       nativeFrameAddress - frame pointer

extern "C" {
JNIEXPORT
void AsyncGetCallTrace2(ASGCT_CallTrace2 *trace, jint depth, void* ucontext) {

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
         "AsyncGetCallTrace2 must be called by the current interrupted thread");

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
  case _thread_in_Java:
  case _thread_in_Java_trans:
    {
      frame ret_frame;
      //printf("--- try\n");
      if (!thread->pd_get_top_frame_for_signal_handler(&ret_frame, ucontext, true)) {
        trace->num_frames = ticks_unknown_not_Java;  // -3 unknown frame
        return;
      }
      ASGCT2StackWalker(thread, trace, depth).fill_given_top(ret_frame);
    }
    break;
  default:
    // Unknown thread state
    trace->num_frames = ticks_unknown_state; // -7
    break;
  }
}
}
