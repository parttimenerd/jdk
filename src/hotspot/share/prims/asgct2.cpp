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

bool print_function_and_library_name(address addr,
                                         char* buf, int buflen) {
  // If no scratch buffer given, allocate one here on stack.
  // (used during error handling; its a coin toss, really, if on-stack allocation
  //  is worse than (raw) C-heap allocation in that case).
  char* p = buf;
  if (p == NULL) {
    p = (char*)::alloca(O_BUFLEN);
    buflen = O_BUFLEN;
  }
  int offset = 0;
  bool have_function_name = os::dll_address_to_function_name(addr, p, buflen,
                                                         &offset, true);
  bool is_function_descriptor = false;
#ifdef HAVE_FUNCTION_DESCRIPTORS
  // When we deal with a function descriptor instead of a real code pointer, try to
  // resolve it. There is a small chance that a random pointer given to this function
  // may just happen to look like a valid descriptor, but this is rare and worth the
  // risk to see resolved function names. But we will print a little suffix to mark
  // this as a function descriptor for the reader (see below).
  if (!have_function_name && os::is_readable_pointer(addr)) {
    address addr2 = (address)os::resolve_function_descriptor(addr);
    if (have_function_name = is_function_descriptor =
        dll_address_to_function_name(addr2, p, buflen, &offset, demangle)) {
      addr = addr2;
    }
  }
#endif // HAVE_FUNCTION_DESCRIPTORS

  if (have_function_name) {
    // Print function name, optionally demangled
    if (true && true) {
      char* args_start = strchr(p, '(');
      if (args_start != NULL) {
        *args_start = '\0';
      }
    }
    // Print offset. Omit printing if offset is zero, which makes the output
    // more readable if we print function pointers.
    if (offset == 0) {
      printf("%s", p);
    } else {
      printf("%s+%d", p, offset);
    }
  } else {
    printf(PTR_FORMAT, p2i(addr));
  }
  offset = 0;

  const bool have_library_name = os::dll_address_to_library_name(addr, p, buflen, &offset);
  if (have_library_name) {
    // Cut path parts
    if (true) {
      char* p2 = strrchr(p, os::file_separator()[0]);
      if (p2 != NULL) {
        p = p2 + 1;
      }
    }
    printf(" in %s", p);
    if (!have_function_name) { // Omit offset if we already printed the function offset
      printf("+%d", offset);
    }
  }

  // Write a trailing marker if this was a function descriptor
  if (have_function_name && is_function_descriptor) {
    printf(" (FD)");
  }

  return have_function_name || have_library_name;
}

#define LOG 0
#define ST_LOG(...) LOG ? printf(__VA_ARGS__) : 1


ASGCT_CallFrame2 process_c_frame(const frame *fr) {
  char cs[100];
  if (LOG) print_function_and_library_name(fr->pc(), cs, 100);
  if (fr->cb() != NULL) {
    ST_LOG("cb %s ", fr->cb()->name());
  }
  if (fr->is_native_frame()) {
    ST_LOG("native method %s ", fr->cb()->name());
  }
  ST_LOG("c frame stub frame %d  native frame %d safe point frame %d compiled frame %d java frame %d\n", fr->is_stub_frame(), fr->is_native_frame(), fr->is_safepoint_blob_frame(), fr->is_compiled_frame(), fr->is_java_frame());
  return {
    -4,
    0,
    fr->pc(),
    encode_type(FRAME_CPP, fr->is_stub_frame() ? CompLevel_all : CompLevel_none)
  };
}


static void fill_call_trace_given_top(JavaThread* thd,
                                      ASGCT_CallTrace2* trace,
                                      int depth,
                                      frame top_frame) {
  NoHandleMark nhm;
  //printf("start\n");
  assert(trace->frames != NULL, "trace->frames must be non-NULL");

  StackWalker st(thd, top_frame, false /* do not skip c frames */);

  int count = 0;
  for (; count < depth && !st.at_end(); st.next(), count++) {
    if (st.at_error()) {
      trace->num_frames = st.state();
      return;
    }
    ST_LOG("state in switch %d\n", st.state());
    switch (st.state()) {
      case STACKWALKER_INTERPRETED_FRAME:
      ST_LOG("1: %s\n", st.method()->name_and_sig_as_C_string());
        trace->frames[count] = {
          st.bci(),
          st.method()->find_jmethod_id_or_null(),
          0,
          encode_type(FRAME_INTERPRETED, CompLevel_none)
        };
        break;
      case STACKWALKER_COMPILED_FRAME:
        ST_LOG("2: %s\n", st.method()->name_and_sig_as_C_string());
        trace->frames[count] = {
          st.bci(),
          st.method()->find_jmethod_id_or_null(),
          0,
          encode_type(st.is_inlined() ? FRAME_INLINED : FRAME_JIT_COMPILED, st.method()->highest_comp_level())
        };
        break;
      case STACKWALKER_NATIVE_FRAME:
        ST_LOG("native %s\n", st.method()->name_and_sig_as_C_string());
        trace->frames[count] = {
          -3,
          st.method()->find_jmethod_id_or_null(),
          0,
          FRAME_NATIVE
        };
        break;
      case STACKWALKER_C_FRAME:
        trace->frames[count] = process_c_frame(st.base_frame());
        break;
      default:
        ST_LOG("unknown state %d\n", st.state());
        assert(false, "should never happen");
        trace->num_frames = ticks_unknown_Java;
    }
  }
  //printf("count: %d\n", count);
  trace->num_frames = count;
}

frame next_c_frame(frame fr) {
  frame invalid;
  if (os::is_first_C_frame(&fr)) {
    return invalid;
  }
  return os::get_sender_for_C_frame(&fr);
}

// we just walk the c stack...
void process_c_trace(ASGCT_CallTrace2 *trace, jint depth, void* ucontext) {
  //printf("process c trace\n");
  if (trace->env_id != NULL && JavaThread::is_thread_from_jni_environment_terminated(trace->env_id)) {
    trace->num_frames = ticks_thread_exit; // -8
    return;
  }
  intptr_t* ret_fp;
  intptr_t* ret_sp;
  address addr = os::fetch_frame_from_context(ucontext, &ret_sp, &ret_fp);
  if (addr == NULL || ret_sp == NULL ) {
    // ucontext wasn't useful
    trace->num_frames = ticks_not_walkable_not_Java;
    return;
  }
  frame ret_frame(ret_sp, ret_fp, addr);
  frame invalid;
  int count = 0;
  do {
    trace->frames[count] = process_c_frame(&ret_frame);
    count++;
  } while ((ret_frame = next_c_frame(ret_frame)).pc() != 0);
  trace->num_frames = count;
  //printf("count = %d\n", count);
}


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
  bool c_traces = false; // os::is_first_C_frame is currently not safe enough
  if (trace->env_id == NULL || JavaThread::is_thread_from_jni_environment_terminated(trace->env_id)) {
    // bad env_id, thread has exited or thread is exiting
    if (c_traces) {
      process_c_trace(trace, depth, ucontext);
    } else {
      trace->num_frames = ticks_thread_exit; // -8;
    }
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
    if (c_traces && false) {
      process_c_trace(trace, depth, ucontext);
    } else {
      trace->num_frames = ticks_no_class_load; // -1
    }
    return;
  }

  if (Universe::heap()->is_gc_active()) {
    if (c_traces) {
      process_c_trace(trace, depth, ucontext);
    } else {
      trace->num_frames = ticks_GC_active; // -2
    }
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
        if (c_traces) {
          process_c_trace(trace, depth, ucontext);
        } else {
          trace->num_frames = ticks_unknown_not_Java;  // -3 unknown frame
        }
        return;
      }
      fill_call_trace_given_top(thread, trace, depth, ret_frame);
    }
    break;
  default:
    // Unknown thread state
    printf("unknown thread state\n");
    trace->num_frames = ticks_unknown_state; // -7
    break;
  }
}
}
