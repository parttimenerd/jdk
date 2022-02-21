/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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

class vframeStreamForte : public vframeStreamCommon {
  bool forte_next_into_inlined = false;
 public:
  // constructor that starts with sender of frame fr (top_frame)
  vframeStreamForte(JavaThread *jt, frame fr, bool stop_at_java_call_stub);
  void forte_next();
  bool forte_next_did_go_into_inlined() { return forte_next_into_inlined; }
  bool inlined() { return _sender_decode_offset != 0; }
};


static bool is_decipherable_compiled_frame(JavaThread* thread, frame* fr, CompiledMethod* nm);
static bool is_decipherable_interpreted_frame(JavaThread* thread,
                                              frame* fr,
                                              Method** method_p,
                                              int* bci_p);




vframeStreamForte::vframeStreamForte(JavaThread *jt,
                                     frame fr,
                                     bool stop_at_java_call_stub) : vframeStreamCommon(jt, false /* process_frames */) {

  _stop_at_java_call_stub = stop_at_java_call_stub;
  _frame = fr;

  // We must always have a valid frame to start filling

  bool filled_in = fill_from_frame();

  assert(filled_in, "invariant");

}


// Solaris SPARC Compiler1 needs an additional check on the grandparent
// of the top_frame when the parent of the top_frame is interpreted and
// the grandparent is compiled. However, in this method we do not know
// the relationship of the current _frame relative to the top_frame so
// we implement a more broad sanity check. When the previous callee is
// interpreted and the current sender is compiled, we verify that the
// current sender is also walkable. If it is not walkable, then we mark
// the current vframeStream as at the end.
//
// returns true if inlined
void vframeStreamForte::forte_next() {
  // handle frames with inlining
  forte_next_into_inlined = false;
  if (_mode == compiled_mode &&
      vframeStreamCommon::fill_in_compiled_inlined_sender()) {
    forte_next_into_inlined = true;
    return;
  }

  // handle general case

  int loop_count = 0;
  int loop_max = MaxJavaStackTraceDepth * 2;


  do {

    loop_count++;

    // By the time we get here we should never see unsafe but better
    // safe then segv'd

    if ((loop_max != 0 && loop_count > loop_max) || !_frame.safe_for_sender(_thread)) {
      _mode = at_end_mode;
      return;
    }

    _frame = _frame.sender(&_reg_map);

  } while (!fill_from_frame());
}

// Determine if 'fr' is a decipherable compiled frame. We are already
// assured that fr is for a java compiled method.

static bool is_decipherable_compiled_frame(JavaThread* thread, frame* fr, CompiledMethod* nm) {
  assert(nm->is_java_method(), "invariant");

  if (thread->has_last_Java_frame() && thread->last_Java_pc() == fr->pc()) {
    // We're stopped at a call into the JVM so look for a PcDesc with
    // the actual pc reported by the frame.
    PcDesc* pc_desc = nm->pc_desc_at(fr->pc());

    // Did we find a useful PcDesc?
    if (pc_desc != NULL &&
        pc_desc->scope_decode_offset() != DebugInformationRecorder::serialized_null) {
      return true;
    }
  }

  // We're at some random pc in the compiled method so search for the PcDesc
  // whose pc is greater than the current PC.  It's done this way
  // because the extra PcDescs that are recorded for improved debug
  // info record the end of the region covered by the ScopeDesc
  // instead of the beginning.
  PcDesc* pc_desc = nm->pc_desc_near(fr->pc() + 1);

  // Now do we have a useful PcDesc?
  if (pc_desc == NULL ||
      pc_desc->scope_decode_offset() == DebugInformationRecorder::serialized_null) {
    // No debug information is available for this PC.
    //
    // vframeStreamCommon::fill_from_frame() will decode the frame depending
    // on the state of the thread.
    //
    // Case #1: If the thread is in Java (state == _thread_in_Java), then
    // the vframeStreamCommon object will be filled as if the frame were a native
    // compiled frame. Therefore, no debug information is needed.
    //
    // Case #2: If the thread is in any other state, then two steps will be performed:
    // - if asserts are enabled, found_bad_method_frame() will be called and
    //   the assert in found_bad_method_frame() will be triggered;
    // - if asserts are disabled, the vframeStreamCommon object will be filled
    //   as if it were a native compiled frame.
    //
    // Case (2) is similar to the way interpreter frames are processed in
    // vframeStreamCommon::fill_from_interpreter_frame in case no valid BCI
    // was found for an interpreted frame. If asserts are enabled, the assert
    // in found_bad_method_frame() will be triggered. If asserts are disabled,
    // the vframeStreamCommon object will be filled afterwards as if the
    // interpreter were at the point of entering into the method.
    return false;
  }

  // This PcDesc is useful however we must adjust the frame's pc
  // so that the vframeStream lookups will use this same pc
  fr->set_pc(pc_desc->real_pc(nm));
  return true;
}


// Determine if 'fr' is a walkable interpreted frame. Returns false
// if it is not. *method_p, and *bci_p are not set when false is
// returned. *method_p is non-NULL if frame was executing a Java
// method. *bci_p is != -1 if a valid BCI in the Java method could
// be found.
// Note: this method returns true when a valid Java method is found
// even if a valid BCI cannot be found.

static bool is_decipherable_interpreted_frame(JavaThread* thread,
                                              frame* fr,
                                              Method** method_p,
                                              int* bci_p) {
  assert(fr->is_interpreted_frame(), "just checking");

  // top frame is an interpreted frame
  // check if it is walkable (i.e. valid Method* and valid bci)

  // Because we may be racing a gc thread the method and/or bci
  // of a valid interpreter frame may look bad causing us to
  // fail the is_interpreted_frame_valid test. If the thread
  // is in any of the following states we are assured that the
  // frame is in fact valid and we must have hit the race.

  JavaThreadState state = thread->thread_state();
  bool known_valid = (state == _thread_in_native ||
                      state == _thread_in_vm ||
                      state == _thread_blocked );

  if (known_valid || fr->is_interpreted_frame_valid(thread)) {

    // The frame code should completely validate the frame so that
    // references to Method* and bci are completely safe to access
    // If they aren't the frame code should be fixed not this
    // code. However since gc isn't locked out the values could be
    // stale. This is a race we can never completely win since we can't
    // lock out gc so do one last check after retrieving their values
    // from the frame for additional safety

    Method* method = fr->interpreter_frame_method();

    // We've at least found a method.
    // NOTE: there is something to be said for the approach that
    // if we don't find a valid bci then the method is not likely
    // a valid method. Then again we may have caught an interpreter
    // frame in the middle of construction and the bci field is
    // not yet valid.
    if (!Method::is_valid_method(method)) return false;
    *method_p = method; // If the Method* found is invalid, it is
                        // ignored by forte_fill_call_trace_given_top().
                        // So set method_p only if the Method is valid.

    address bcp = fr->interpreter_frame_bcp();
    int bci = method->validate_bci_from_bcp(bcp);

    // note: bci is set to -1 if not a valid bci
    *bci_p = bci;
    return true;
  }

  return false;
}


// Determine if a Java frame can be found starting with the frame 'fr'.
//
// Check the return value of find_initial_Java_frame and the value of
// 'method_p' to decide on how use the results returned by this method.
//
// If 'method_p' is not NULL, an initial Java frame has been found and
// the stack can be walked starting from that initial frame. In this case,
// 'method_p' points to the Method that the initial frame belongs to and
// the initial Java frame is returned in initial_frame_p.
//
// find_initial_Java_frame() returns true if a Method has been found (i.e.,
// 'method_p' is not NULL) and the initial frame that belongs to that Method
// is decipherable.
//
// A frame is considered to be decipherable:
//
// - if the frame is a compiled frame and a PCDesc is available;
//
// - if the frame is an interpreter frame that is valid or the thread is
//   state (_thread_in_native || state == _thread_in_vm || state == _thread_blocked).
//
// Note that find_initial_Java_frame() can return false even if an initial
// Java method was found (e.g., there is no PCDesc available for the method).
//
// If 'method_p' is NULL, it was not possible to find a Java frame when
// walking the stack starting from 'fr'. In this case find_initial_Java_frame
// returns false.

static bool find_initial_Java_frame(JavaThread* thread,
                                    frame* fr,
                                    frame* initial_frame_p,
                                    Method** method_p,
                                    int* bci_p) {

  // It is possible that for a frame containing a compiled method
  // we can capture the method but no bci. If we get no
  // bci the frame isn't walkable but the method is usable.
  // Therefore we init the returned Method* to NULL so the
  // caller can make the distinction.

  *method_p = NULL;

  // On the initial call to this method the frame we get may not be
  // recognizable to us. This should only happen if we are in a JRT_LEAF
  // or something called by a JRT_LEAF method.

  frame candidate = *fr;

#ifdef ZERO
  // Zero has no frames with code blobs, so the generic code fails.
  // Instead, try to do Zero-specific search for Java frame.

  {
    RegisterMap map(thread, false, false);

    while (true) {
      // Cannot walk this frame? Cannot do anything anymore.
      if (!candidate.safe_for_sender(thread)) {
        return false;
      }

      if (candidate.is_entry_frame()) {
        // jcw is NULL if the java call wrapper could not be found
        JavaCallWrapper* jcw = candidate.entry_frame_call_wrapper_if_safe(thread);
        // If initial frame is frame from StubGenerator and there is no
        // previous anchor, there are no java frames associated with a method
        if (jcw == NULL || jcw->is_first_frame()) {
          return false;
        }
      }

      // If we find a decipherable interpreted frame, this is our initial frame.
      if (candidate.is_interpreted_frame()) {
        if (is_decipherable_interpreted_frame(thread, &candidate, method_p, bci_p)) {
          *initial_frame_p = candidate;
          return true;
        }
      }

      // Walk some more.
      candidate = candidate.sender(&map);
    }

    // No dice, report no initial frames.
    return false;
  }
#endif

  // If the starting frame we were given has no codeBlob associated with
  // it see if we can find such a frame because only frames with codeBlobs
  // are possible Java frames.

  if (fr->cb() == NULL) {

    // See if we can find a useful frame
    int loop_count;
    int loop_max = MaxJavaStackTraceDepth * 2;
    RegisterMap map(thread, false, false);

    for (loop_count = 0; loop_max == 0 || loop_count < loop_max; loop_count++) {
      if (!candidate.safe_for_sender(thread)) return false;
      candidate = candidate.sender(&map);
      if (candidate.cb() != NULL) break;
    }
    if (candidate.cb() == NULL) return false;
  }

  // We have a frame known to be in the codeCache
  // We will hopefully be able to figure out something to do with it.
  int loop_count;
  int loop_max = MaxJavaStackTraceDepth * 2;
  RegisterMap map(thread, false, false);

  for (loop_count = 0; loop_max == 0 || loop_count < loop_max; loop_count++) {

    if (candidate.is_entry_frame()) {
      // jcw is NULL if the java call wrapper couldn't be found
      JavaCallWrapper *jcw = candidate.entry_frame_call_wrapper_if_safe(thread);
      // If initial frame is frame from StubGenerator and there is no
      // previous anchor, there are no java frames associated with a method
      if (jcw == NULL || jcw->is_first_frame()) {
        return false;
      }
    }

    if (candidate.is_interpreted_frame()) {
      if (is_decipherable_interpreted_frame(thread, &candidate, method_p, bci_p)) {
        *initial_frame_p = candidate;
        return true;
      }

      // Hopefully we got some data
      return false;
    }

    if (candidate.cb()->is_compiled()) {

      CompiledMethod* nm = candidate.cb()->as_compiled_method();
      *method_p = nm->method();

      // If the frame is not decipherable, then the value of -1
      // for the BCI is used to signal that no BCI is available.
      // Furthermore, the method returns false in this case.
      //
      // If a decipherable frame is available, the BCI value will
      // not be used.

      *bci_p = -1;

      *initial_frame_p = candidate;

      // Native wrapper code is trivial to decode by vframeStream

      if (nm->is_native_method()) return true;

      // If the frame is not decipherable, then a PC was found
      // that does not have a PCDesc from which a BCI can be obtained.
      // Nevertheless, a Method was found.

      if (!is_decipherable_compiled_frame(thread, &candidate, nm)) {
        return false;
      }

      // is_decipherable_compiled_frame may modify candidate's pc
      *initial_frame_p = candidate;

      assert(nm->pc_desc_at(candidate.pc()) != NULL, "debug information must be available if the frame is decipherable");

      return true;
    }

    // Must be some stub frame that we don't care about

    if (!candidate.safe_for_sender(thread)) return false;
    candidate = candidate.sender(&map);

    // If it isn't in the code cache something is wrong
    // since once we find a frame in the code cache they
    // all should be there.

    if (candidate.cb() == NULL) return false;

  }

  return false;

}

static void forte_fill_call_trace_given_top(JavaThread* thd,
                                            ASGCT_CallTrace* trace,
                                            int depth,
                                            frame top_frame) {
  NoHandleMark nhm;

  frame initial_Java_frame;
  Method* method;
  int bci = -1; // assume BCI is not available for method
                // update with correct information if available
  int count;

  count = 0;
  assert(trace->frames != NULL, "trace->frames must be non-NULL");

  // Walk the stack starting from 'top_frame' and search for an initial Java frame.
  find_initial_Java_frame(thd, &top_frame, &initial_Java_frame, &method, &bci);

  // Check if a Java Method has been found.
  if (method == NULL) return;

  if (!Method::is_valid_method(method)) {
    trace->num_frames = ticks_GC_active; // -2
    return;
  }

  vframeStreamForte st(thd, initial_Java_frame, false);

  for (; !st.at_end() && count < depth; st.forte_next(), count++) {
    bci = st.bci();
    method = st.method();

    if (!Method::is_valid_method(method)) {
      // we throw away everything we've gathered in this sample since
      // none of it is safe
      trace->num_frames = ticks_GC_active; // -2
      return;
    }

    trace->frames[count].method_id = method->find_jmethod_id_or_null();
    if (!method->is_native()) {
      trace->frames[count].lineno = bci;
    } else {
      trace->frames[count].lineno = -3;
    }
  }
  trace->num_frames = count;
  return;
}


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
  JavaThread* thread;

  if (trace->env_id == NULL ||
    (thread = JavaThread::thread_from_jni_environment(trace->env_id)) == NULL ||
    thread->is_exiting()) {

    // bad env_id, thread has exited or thread is exiting
    trace->num_frames = ticks_thread_exit; // -8
    return;
  }

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
          forte_fill_call_trace_given_top(thread, trace, depth, fr);

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
        forte_fill_call_trace_given_top(thread, trace, depth, fr);
      }
    }
    break;
  default:
    // Unknown thread state
    trace->num_frames = ticks_unknown_state; // -7
    break;
  }
}

}

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

/**
 *
 * Gets the caller frame of `fr`.
 *
 * based on the next_frame method from vmError.cpp aka pns gdb command
 *
 * only usable when we are sure to not have any compiled frames afterwards,
 * as this method might trip up
 *
 * Problem: leads to "invalid bci or invalid scope error" in vframestream
 */
static frame next_frame(JavaThread* t, frame fr, bool supports_os_get_frame) {
  // Compiled code may use EBP register on x86 so it looks like
  // non-walkable C frame. Use frame.sender() for java frames.
  frame invalid;
  // Catch very first native frame by using stack address.
  // For JavaThread stack_base and stack_size should be set.
  if (!t->is_in_full_stack((address)(fr.real_fp() + 1))) {
    return invalid;
  }
  if (fr.is_java_frame() || fr.is_native_frame() || fr.is_runtime_frame() || !supports_os_get_frame) {
    if (!fr.safe_for_sender(t)) {
      return invalid;
    }
    RegisterMap map(t, false, false); // No update
    return fr.sender(&map);
  } else {
    // is_first_C_frame() does only simple checks for frame pointer,
    // it will pass if java compiled code has a pointer in EBP.
    if (os::is_first_C_frame(&fr)) return invalid;
    return os::get_sender_for_C_frame(&fr);
  }
}

// -1: problems somewhere, -2 if gc seems to be active, otherwise the count of non ignored frames
template<typename T = int>
static int get_frame_count(JavaThread* thread,
frame top_frame,
void (*raw_frame_func)(frame&, T),
void (*interp_frame_func)(frame&, Method*, int, T),
void (*compiled_frame_func)(Method*, int, bool, T),
void (*misc_frame_func)(frame&, bool, T),
T args = 0) {
  RegisterMap map(thread, false, false);
  int count = 0;
  int raw_count = 0;
  do {
    if (!top_frame.safe_for_sender(thread)) {
      //printf("unsafe\n");
      if (next_frame(thread, top_frame, true).pc() != NULL) {
        // we are still able to walk the C stack
        break;
      }
      return -1;
    }
    top_frame = top_frame.sender(&map);
    raw_frame_func(top_frame, args);
    count++;

    if (top_frame.is_java_frame()) { // another validity check
      Method *method_p = NULL;
      int bci_p = 0;

      if (top_frame.is_interpreted_frame()) {
        if (!top_frame.is_interpreted_frame_valid(thread) || !is_decipherable_interpreted_frame(thread, &top_frame, &method_p, &bci_p)) {
          printf("interpret not decipherable\n");
          return -1;
        }
        if (!Method::is_valid_method(method_p)) {
          // we throw away everything we've gathered in this sample since
          // none of it is safe
          printf("gc_active\n");
          return ticks_GC_active; // -2
        }
        interp_frame_func(top_frame, method_p, bci_p, args);
      } else if (top_frame.is_compiled_frame()) {
        frame enclosing_frame = top_frame;
        CompiledMethod* nm = enclosing_frame.cb()->as_compiled_method();
        method_p = nm->method();
        bci_p = -1;
        if (!is_decipherable_compiled_frame(thread, &enclosing_frame, nm)){
          return -1;
        }

        vframeStreamForte st(thread, top_frame, false);

        for (; !st.at_end(); st.forte_next()) {
          Method* method = st.method();
          int bci = st.bci();

          if (!Method::is_valid_method(method)) {
            // we throw away everything we've gathered in this sample since
            // none of it is safe
            return ticks_GC_active;
          }
          if (method->find_jmethod_id_or_null() != NULL) {
            compiled_frame_func(method, bci, st.inlined(), args);
          } else {  // there has to be something wrong here
            return -1;
          }
          if (!st.inlined()) {
            break;
          }
        }
      } else {
        misc_frame_func(top_frame, false, args);
      }
    } else {
      misc_frame_func(top_frame, false, args);
    }
  } while (!top_frame.is_first_frame());
  while ((top_frame = next_frame(thread, top_frame, true)).pc() != NULL) {
    misc_frame_func(top_frame, true, args);
    count++;
  }
  return count;
}

template <typename... T>
static void nil_function(T... args) {}

class frameState {
  ASGCT_CallTrace2* trace;
  int depth;
  int current_frame;
  int index;
public:
  frameState(ASGCT_CallTrace2* trace, int depth, int actual_frames): trace(trace), depth(depth) {
    current_frame = actual_frames - 1;
    index = 0;
  }

  void register_current_frame(ASGCT_CallFrame2 frame) {
    if (current_frame < depth) {
      trace->frames[index] = frame;
      index++;
    }
    current_frame--;
  }
};

void handle_interpreted_frame(frame& frame, Method* method, int bci, frameState &state) {
  //printf("interpreted frame %s\n", method->name_and_sig_as_C_string());
  state.register_current_frame({
    bci,
    method->find_jmethod_id_or_null(),
    0,
    encode_type(FRAME_INTERPRETED, CompLevel_none)
  });
}

void handle_compiled_frame(Method* method, int bci, bool inlined, frameState &state) {
  //printf("compiled frame %s %s\n", method->name_and_sig_as_C_string(), inlined ? "inlined" : "");
  int16_t type = encode_type(inlined ? FRAME_INLINED : FRAME_JIT_COMPILED, method->highest_comp_level());
  state.register_current_frame({
    bci,
    method->find_jmethod_id_or_null(),
    0,
    type
  });
}

void handle_misc_frame(frame& frame, bool after_end, frameState &state) {
  //printf("misc frame\n");
  bool is_native_frame = frame.is_native_frame();
  state.register_current_frame({
      -4,
      0,
      frame.pc(),
      encode_type(is_native_frame ? FRAME_NATIVE : FRAME_CPP, CompLevel_none)
    });
}


static void forte_fill_call_trace_given_top2(JavaThread* thd,
                                            ASGCT_CallTrace2* trace,
                                            int depth,
                                            frame top_frame) {
  assert(trace->frames != NULL, "trace->frames must be non-NULL");
  NoHandleMark nhm;
  int rec_frame_count = get_frame_count(thd, top_frame, nil_function, nil_function, nil_function, nil_function);
  if (rec_frame_count == -1) {
    trace->num_frames = 0;
    return;
  }
  if (rec_frame_count < -1) { // some other error
    trace->num_frames = rec_frame_count;
    return;
  }

  frameState state(trace, depth, rec_frame_count);
  get_frame_count<frameState&>(thd, top_frame, nil_function,
    handle_interpreted_frame,
    handle_compiled_frame,
    handle_misc_frame,
    state);
  trace->num_frames = rec_frame_count;
  return;
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
  JavaThread* thread;

  if (trace->env_id == NULL ||
    (thread = JavaThread::thread_from_jni_environment(trace->env_id)) == NULL ||
    thread->is_exiting()) {
    // bad env_id, thread has exited or thread is exiting
    trace->num_frames = ticks_thread_exit; // -8
    return;
  }

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
      frame ret_frame;
      if (!thread->pd_get_top_frame_for_signal_handler(&ret_frame, ucontext, false)) {
        trace->num_frames = ticks_unknown_not_Java;  // -3 unknown frame
        return;
      }
      forte_fill_call_trace_given_top2(thread, trace, depth, ret_frame);
    }
    break;
  case _thread_in_Java:
  case _thread_in_Java_trans:
    {
      frame ret_frame;
      if (!thread->pd_get_top_frame_for_signal_handler(&ret_frame, ucontext, true)) {
        trace->num_frames = ticks_unknown_not_Java;  // -3 unknown frame
        return;
      }
      forte_fill_call_trace_given_top2(thread, trace, depth, ret_frame);
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
