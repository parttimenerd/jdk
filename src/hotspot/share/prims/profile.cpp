/*
 * Copyright (c) 2024, SAP SE or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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

#include "profile.h"

#include "jni.h"
#include "precompiled.hpp"
#include "code/debugInfoRec.hpp"
#include "code/pcDesc.hpp"
#include "gc/shared/collectedHeap.inline.hpp"
#include "memory/universe.hpp"
#include "oops/oop.inline.hpp"
#include "prims/forte.hpp"
#include "prims/jvmtiExport.hpp"
#include "runtime/frame.hpp"
#include "runtime/frame.inline.hpp"
#include "runtime/javaCalls.hpp"
#include "runtime/javaThread.inline.hpp"
#include "runtime/stackFrameStream.inline.hpp"
#include "runtime/vframe.inline.hpp"
#include "runtime/vframeArray.hpp"
#include "utilities/checkedCast.hpp"
#include "utilities/globalDefinitions.hpp"
#include <cstdint>

// the following is copied from forte.cpp

class vframeStreamASGST : public vframeStreamCommon {
 public:
  // constructor that starts with sender of frame fr (top_frame)
  vframeStreamASGST(JavaThread *jt, frame fr, bool stop_at_java_call_stub, bool walk_loom);
  void forte_next();

  frame current_frame() { return _frame; }

  bool inlined() { return _sender_decode_offset == DebugInformationRecorder::serialized_null; }
};


static bool is_decipherable_compiled_frame(JavaThread* thread, frame* fr, CompiledMethod* nm);
static bool is_decipherable_interpreted_frame(JavaThread* thread,
                                              frame* fr,
                                              Method** method_p,
                                              int* bci_p);




vframeStreamASGST::vframeStreamASGST(JavaThread *jt,
                                     frame fr,
                                     bool stop_at_java_call_stub,
                                     bool walk_loom)
    : vframeStreamCommon(RegisterMap(jt,
                                     RegisterMap::UpdateMap::skip,
                                     RegisterMap::ProcessFrames::skip,
                                     walk_loom ? RegisterMap::WalkContinuation::skip : RegisterMap::WalkContinuation::include)) {
  _reg_map.set_async(true);
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
void vframeStreamASGST::forte_next() {
  // handle frames with inlining
  if (_mode == compiled_mode &&
      vframeStreamCommon::fill_in_compiled_inlined_sender()) {
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
    if (pc_desc != nullptr &&
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
  if (pc_desc == nullptr ||
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
// returned. *method_p is non-null if frame was executing a Java
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
// If 'method_p' is not null, an initial Java frame has been found and
// the stack can be walked starting from that initial frame. In this case,
// 'method_p' points to the Method that the initial frame belongs to and
// the initial Java frame is returned in initial_frame_p.
//
// find_initial_Java_frame() returns true if a Method has been found (i.e.,
// 'method_p' is not null) and the initial frame that belongs to that Method
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
// If 'method_p' is null, it was not possible to find a Java frame when
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
  // Therefore we init the returned Method* to null so the
  // caller can make the distinction.

  *method_p = nullptr;

  // On the initial call to this method the frame we get may not be
  // recognizable to us. This should only happen if we are in a JRT_LEAF
  // or something called by a JRT_LEAF method.

  frame candidate = *fr;

#ifdef ZERO
  // Zero has no frames with code blobs, so the generic code fails.
  // Instead, try to do Zero-specific search for Java frame.

  {
    RegisterMap map(thread,
                    RegisterMap::UpdateMap::skip,
                    RegisterMap::ProcessFrames::skip,
                    RegisterMap::WalkContinuation::skip);

    while (true) {
      // Cannot walk this frame? Cannot do anything anymore.
      if (!candidate.safe_for_sender(thread)) {
        return false;
      }

      if (candidate.is_entry_frame()) {
        // jcw is null if the java call wrapper could not be found
        JavaCallWrapper* jcw = candidate.entry_frame_call_wrapper_if_safe(thread);
        // If initial frame is frame from StubGenerator and there is no
        // previous anchor, there are no java frames associated with a method
        if (jcw == nullptr || jcw->is_first_frame()) {
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

  if (fr->cb() == nullptr) {

    // See if we can find a useful frame
    int loop_count;
    int loop_max = MaxJavaStackTraceDepth * 2;
    RegisterMap map(thread,
                    RegisterMap::UpdateMap::skip,
                    RegisterMap::ProcessFrames::skip,
                    RegisterMap::WalkContinuation::skip);

    for (loop_count = 0; loop_max == 0 || loop_count < loop_max; loop_count++) {
      if (!candidate.safe_for_sender(thread)) return false;
      candidate = candidate.sender(&map);
      if (candidate.cb() != nullptr) break;
    }
    if (candidate.cb() == nullptr) return false;
  }

  // We have a frame known to be in the codeCache
  // We will hopefully be able to figure out something to do with it.
  int loop_count;
  int loop_max = MaxJavaStackTraceDepth * 2;
  RegisterMap map(thread,
                  RegisterMap::UpdateMap::skip,
                  RegisterMap::ProcessFrames::skip,
                  RegisterMap::WalkContinuation::skip);

  for (loop_count = 0; loop_max == 0 || loop_count < loop_max; loop_count++) {

    if (candidate.is_entry_frame()) {
      // jcw is null if the java call wrapper couldn't be found
      JavaCallWrapper *jcw = candidate.entry_frame_call_wrapper_if_safe(thread);
      // If initial frame is frame from StubGenerator and there is no
      // previous anchor, there are no java frames associated with a method
      if (jcw == nullptr || jcw->is_first_frame()) {
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

      assert(nm->pc_desc_at(candidate.pc()) != nullptr, "debug information must be available if the frame is decipherable");

      return true;
    }

    // Must be some stub frame that we don't care about

    if (!candidate.safe_for_sender(thread)) return false;
    candidate = candidate.sender(&map);

    // If it isn't in the code cache something is wrong
    // since once we find a frame in the code cache they
    // all should be there.

    if (candidate.cb() == nullptr) return false;

  }

  return false;

}

// based on forte_fill_call_trace_given_top
int walk_stack(JavaThread* thread, frame top_frame, ASGST_WalkStackCallback javaCallback,
  ASGST_CFrameCallback nonJavaCallback, void *arg, int options) {

  if (options & ASGST_RECONSTITUTE) {
    StackWatermarkSet::start_processing(thread, StackWatermarkKind::gc);
  }

  bool walk_loom_threads = (options & ASGST_WALK_LOOM_THREADS) != 0;

  NoHandleMark nhm;

  frame initial_Java_frame;
  Method* method;
  int bci = -1; // assume BCI is not available for method
                // update with correct information if available

  // Walk the stack starting from 'top_frame' and search for an initial Java frame.
  find_initial_Java_frame(thread, &top_frame, &initial_Java_frame, &method, &bci);

  // Check if a Java Method has been found.
  if (method == nullptr) return 0;

  if (!Method::is_valid_method(method)) {
    return ASGST_GC_active; // -2
  }

  vframeStreamASGST st(thread, initial_Java_frame, false, walk_loom_threads);

  for (; !st.at_end(); st.forte_next()) {
    method = st.method();

    if (!Method::is_valid_method(method)) {
      // we throw away everything we've gathered in this sample since
      // none of it is safe
      return ASGST_GC_active; // -2
    }

    jmethodID method_id = method->find_jmethod_id_or_null();
    bci = method->is_native() ? -3 : st.bci();

    frame fr = st.current_frame();

    uint8_t type = 0;
    if (method->is_native()) {
      type = ASGST_FRAME_JAVA_NATIVE;
    } else if (st.inlined()) {
      type = ASGST_FRAME_JAVA_INLINED;
    } else {
      type = ASGST_FRAME_JAVA;
    }

    ASGST_FrameInfo frame_info{
      .type = type,
      .comp_level = st.cb()->is_compiled() ? st.cb()->as_nmethod()->comp_level() : 0,
      .bci = bci,
      .method = method_id,
      .frame = ASGST_Frame{fr.pc(), fr.sp(), fr.fp()}
      };
    if (javaCallback(&frame_info, arg) != 1) {
      break;
    }

    if (nonJavaCallback != nullptr) {
      frame fr = st.current_frame();
      if (!os::is_first_C_frame(&fr)) {
        frame parent = os::get_sender_for_C_frame(&fr);
        if (!parent.is_java_frame()) {
          ASGST_Frame nonJavaFrame{fr.pc(), fr.sp(), fr.fp()};
          nonJavaCallback(&nonJavaFrame, arg);
        }
      }
    }
  }
  return 1;
}

JavaThread* get_thread() {
  Thread* raw_thread = Thread::current_or_null_safe();
  JavaThread* thread;

  if (raw_thread == nullptr || !raw_thread->is_Java_thread() ||
      (thread = JavaThread::cast(raw_thread))->is_exiting()) {
    // bad env_id, thread has exited or thread is exiting
    return nullptr;
  }
  return thread;
}

frame create_frame(void* sp, void* fp, void* pc) {
  frame fr = frame();
  fr.init((intptr_t*)sp, (intptr_t*)fp, (address)pc);
  return fr;
}

void checkEnabled() {
  if (EnableMinimalASGST == 0) {
    fatal("Minimal ASGST is not enabled, pass -XX:+EnableMinimalASGST to enable it");
  }
}

int ASGST_WalkStackFromFrame(ASGST_Frame fr,
  ASGST_WalkStackCallback javaCallback,
  ASGST_CFrameCallback nonJavaCallback,
  void *arg, int options) {
  checkEnabled();

  assert(javaCallback != nullptr, "invariant");

  if (fr.pc == nullptr) {
    return ASGST_no_Java_frame;
  }

  JavaThread* thread = get_thread();
  if (thread == nullptr) {
    return ASGST_thread_exit;
  }

  if (thread->in_deopt_handler()) {
    // We are in a deopt handler, so we can't walk the stack.
    return ASGST_deopt;
  }

  if (!JvmtiExport::should_post_class_load()) {
    return ASGST_no_class_load;
  }

  if (Universe::heap()->is_gc_active()) {
    return ASGST_GC_active;
  }

  // signify to other code in the VM that we're in ASGCT
  ThreadInAsgct tia(thread);

  frame top_frame = create_frame(fr.sp, fr.fp, fr.pc);

  switch (thread->thread_state()) {
  case _thread_new:
  case _thread_uninitialized:
  case _thread_new_trans:
    // We found the thread on the threads list above, but it is too
    // young to be useful so return that there are no Java frames.
    return 0;
  case _thread_in_native:
  case _thread_in_native_trans:
  case _thread_blocked:
  case _thread_blocked_trans:
  case _thread_in_vm:
  case _thread_in_vm_trans:
    {
      if (!thread->has_last_Java_frame()) {
        return 0; // No Java frames
      } else {
        int ret = walk_stack(thread, top_frame, javaCallback, nonJavaCallback, arg, options);
        if (ret == 0) {
          return ASGST_not_walkable_not_Java;
        }
        return ret;
      }
    }
    break;
  case _thread_in_Java:
  case _thread_in_Java_trans:
    {
      int ret = walk_stack(thread, top_frame, javaCallback, nonJavaCallback, arg, options);
      if (ret == 0) {
        return ASGST_not_walkable_Java;
      }
      return ret;
    }
    break;
  default:
    // Unknown thread state
    return ASGST_unknown_state;
  }
}


int ASGST_IsJavaFrame(ASGST_Frame fr) {
  checkEnabled();
  return create_frame(fr.fp, fr.sp, fr.pc).is_java_frame();
}

ASGST_Frame ASGST_GetFrame(void* ucontext, bool focusOnJava) {
  checkEnabled();
  JavaThread* thread = get_thread();
  ASGST_Frame empty{nullptr, nullptr, nullptr};
  if (thread == nullptr) {
    return empty;
  }
  if (focusOnJava) {
    frame ret_frame;
    if (thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, true)) {
      return ASGST_Frame{ret_frame.pc(), ret_frame.sp(), ret_frame.fp()};
    } else if (thread->frame_anchor()->last_Java_sp() != nullptr) {
      address last_Java_pc = thread->last_Java_pc();
      if (last_Java_pc == nullptr) {
        last_Java_pc = (address)thread->last_Java_sp()[-1];
      }
      return ASGST_Frame{last_Java_pc, thread->last_Java_sp(), thread->frame_anchor()->last_Java_fp()};
    }
    frame last_frame = os::fetch_frame_from_context(ucontext);
    if (os::is_first_C_frame(&last_frame)) {
      return empty;
    }
    return ASGST_Frame{last_frame.pc(), last_frame.sp(), last_frame.fp()};
  }
  intptr_t* ret_sp = nullptr;
  intptr_t* ret_fp = nullptr;
  void* ret_pc = os::fetch_frame_from_context(ucontext, &ret_sp, &ret_fp);
  if (ret_pc == nullptr || ret_sp == nullptr ) {
    // ucontext wasn't useful
    return empty;
  }
  frame ret_frame = create_frame(ret_sp, ret_fp, ret_pc);
  if (!ret_frame.safe_for_sender(thread)) {
#if COMPILER2_OR_JVMCI
    // C2 and JVMCI use ebp as a general register see if null fp helps
    frame ret_frame2 = create_frame(ret_sp, nullptr, ret_pc);
    if (!ret_frame2.safe_for_sender(thread)) {
      // nothing else to try if the frame isn't good
      return empty;
    }
    ret_frame = ret_frame2;
#else
    return empty;
#endif
  }
  return ASGST_Frame{ret_frame.pc(), ret_frame.sp(), ret_frame.fp()};
}


void ASGST_SetSafepointCallback(ASGST_SafepointCallback callback, void *arg) {
  checkEnabled();
  JavaThread* thread = get_thread();
  if (thread == nullptr) {
    return;
  }
  thread->set_asgst_safepoint_callback(callback, arg);
}

void ASGST_GetSafepointCallback(ASGST_SafepointCallback *callback, void **arg) {
  checkEnabled();
  JavaThread* thread = get_thread();
  if (thread == nullptr) {
    *callback = nullptr;
    *arg = nullptr;
    return;
  }
  thread->get_asgst_safepoint_callback(callback, arg);
}

void ASGST_TriggerSafePoint() {
  checkEnabled();
  JavaThread* thread = get_thread();
  if (thread == nullptr) {
    return;
  }
  thread->trigger_asgst_safepoint();
}

// based on compute_top_java_frame from
// https://github.com/openjdk/jdk/compare/master...fisk:jdk:jfr_safe_trace_v1
// by Erik Österlund
bool compute_top_java_frame(JavaThread* thread, frame request, frame* top_frame) {
  if (!thread->has_last_Java_frame()) {
    return false;
  }

  void* sampled_sp = request.sp();
  void* sampled_pc = request.pc();
  const char* sampler = (thread == Thread::current()) ? "self" : "remote";

  CodeBlob* sampled_cb = CodeCache::find_blob(sampled_pc);

  if (sampled_cb == nullptr) {
    // No code blob... probably native code. Perform a biased sample
    *top_frame = thread->last_frame();
    return true;
  }

  if (!sampled_cb->is_nmethod() &&
      !sampled_cb->is_vtable_blob() &&
      !sampled_cb->is_adapter_blob() &&
      !sampled_cb->is_method_handles_adapter_blob()) {
    // Cold code blob... perform a biased sample
    *top_frame = thread->last_frame();
    return true;
  }

  // For nmethods, vtable stubs, itable stubs, adapter blobs and method handle intrinsic blobs,
  // want to perform an accurate unbiased sample
  nmethod* sampled_nm = sampled_cb->as_nmethod_or_null();

  // We sampled an nmethod. Let's find the frame it came from.
  RegisterMap map(thread,
                  RegisterMap::UpdateMap::skip,
                  RegisterMap::ProcessFrames::skip,
                  RegisterMap::WalkContinuation::skip);

  // Search the first frame that is above the sampled sp
  for (StackFrameStream frame_stream(thread, false /* update_registers */, false /* process_frames */);
       !frame_stream.is_done();
       frame_stream.next()) {
    frame* f = frame_stream.current();

    if (f->is_safepoint_blob_frame() || f->is_runtime_frame()) {
      // Skip runtime stubs
      continue;
    }

    // Seek the first matching frame
    if (f->real_fp() <= sampled_sp) {
      // Continue searching the matching frame or its caller
      continue;
    }

    if (sampled_nm == nullptr) {
      // The sample didn't have an nmethod; we decided to trace from its caller
      *top_frame = *f;
      return true;
    }

    // We might have a matching frame; check it
    if (f->cb()->as_nmethod_or_null() == sampled_nm) {
      // We found the sampled nmethod! Let's correct the safepoint bias
      PcDesc* pc_desc = sampled_nm->pc_desc_near(address(sampled_pc) + 1);
      if (pc_desc == nullptr || pc_desc->scope_decode_offset() == DebugInformationRecorder::serialized_null) {
        // Bogus PC at frame boundary; we are close enough to the caller; trace from there
        continue;
      }
      f->set_pc(pc_desc->real_pc(sampled_nm));
      assert(sampled_nm->pc_desc_at(f->pc()) != nullptr, "invalid pc");

      *top_frame = *f;
      return true;
    } else {
      // Frame not matching... possibly due to polling after unwinding.
      address saved_exception_pc = thread->saved_exception_pc();
      nmethod* exception_nm = saved_exception_pc == nullptr ? nullptr : CodeCache::find_blob(saved_exception_pc)->as_nmethod_or_null();

      if (exception_nm == sampled_nm && sampled_nm->is_at_poll_return(saved_exception_pc)) {
        // We have polled at an unwind site in the compiled method. Let's reconstruct what the frame
        // would have looked like before unwinding. This will point into garbage stack memory, but
        // is safe, as the stack sampling only cares about PCs, and not the content of the stack.
        intptr_t* previous_sp = f->sp() - sampled_nm->frame_size();

        // We found the sampled nmethod! Let's correct the safepoint bias
        PcDesc* pc_desc = sampled_nm->pc_desc_near(address(sampled_pc) + 1);
        if (pc_desc == nullptr || pc_desc->scope_decode_offset() == DebugInformationRecorder::serialized_null) {
          // Bogus PC at frame boundary; we are close enough to the caller; trace from there
          *top_frame = *f;
        } else {
          *top_frame = frame(previous_sp, previous_sp, (intptr_t*)f->sp(), (address)pc_desc->real_pc(sampled_nm), sampled_nm);
        }
      } else {
        // Mismatched sample; trace from caller
        *top_frame = *f;
      }

      return true;
    }
  }

  // No frame found
  return false;
}

ASGST_Frame ASGST_ComputeTopFrameAtSafepoint(ASGST_Frame captured) {
  checkEnabled();
  JavaThread* thread = get_thread();
  ASGST_Frame empty{nullptr, nullptr, nullptr};
  if (thread == nullptr) {
    return empty;
  }

  frame top_frame;
  if (!compute_top_java_frame(thread, create_frame(captured.sp, captured.fp, captured.pc), &top_frame)) {
    return empty;
  }

  return ASGST_Frame{top_frame.pc(), top_frame.sp(), top_frame.fp()};
}