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

#include "stackWalker.hpp"
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

static bool is_decipherable_compiled_frame(JavaThread* thread, frame* fr, CompiledMethod* nm);
static bool is_decipherable_interpreted_frame(JavaThread* thread,
                                              frame* fr,
                                              Method** method_p,
                                              int* bci_p);


class compiledFrameStreamForte : public vframeStreamCommon {
  bool forte_next_into_inlined = false;
 public:
  // constructor that starts with sender of frame fr (top_frame)
  compiledFrameStreamForte(JavaThread *jt, frame fr, bool stop_at_java_call_stub);
  void forte_next();
  bool forte_next_did_go_into_inlined() { return forte_next_into_inlined; }
  bool inlined() { return _sender_decode_offset != 0; }
};

// the following code was originally present in the forte.cpp file
// it is moved in to this file to allow reuse in JFR


compiledFrameStreamForte::compiledFrameStreamForte(JavaThread *jt,
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
void compiledFrameStreamForte::forte_next() {
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


StackWalker::StackWalker(JavaThread* thread): thread(thread),
  supports_os_get_frame(os::current_frame().pc() != NULL) {
}

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
 frame StackWalker::next_c_frame(frame fr) {
  // Compiled code may use EBP register on x86 so it looks like
  // non-walkable C frame. Use frame.sender() for java frames.
  frame invalid;
  // Catch very first native frame by using stack address.
  // For JavaThread stack_base and stack_size should be set.
  if (!thread->is_in_full_stack((address)(fr.real_fp() + 1))) {
    return invalid;
  }
  if (fr.is_java_frame() || fr.is_native_frame() || fr.is_runtime_frame() || !supports_os_get_frame) {
    if (!fr.safe_for_sender(thread)) {
      return invalid;
    }
    RegisterMap map(thread, false, false); // No update
    return fr.sender(&map);
  } else {
    // is_first_C_frame() does only simple checks for frame pointer,
    // it will pass if java compiled code has a pointer in EBP.
    if (os::is_first_C_frame(&fr)) {
      return invalid;
    }
    return os::get_sender_for_C_frame(&fr);
  }
}

int StackWalker::walk(frame top_frame) {
  RegisterMap map(thread, false, false);
  int count = 0;
  int raw_count = 0;
  do {
    if (abort()) {
      return count;
    }
    if (!top_frame.safe_for_sender(thread)) {
      if (next_c_frame(top_frame).pc() != NULL) {
        // we are still able to walk the C stack
        break;
      }
      if (count > 0) {
        return count;
      } else {
        return STACKWALKER_INDECIPHERABLE_FRAME;
      }
    }
    top_frame = top_frame.sender(&map);
    this->handle_frame(top_frame);
    count++;

    if (top_frame.is_java_frame()) { // another validity check
      Method *method_p = NULL;
      int bci_p = 0;

      if (top_frame.is_interpreted_frame()) {
        if (!top_frame.is_interpreted_frame_valid(thread) || !is_decipherable_interpreted_frame(thread, &top_frame, &method_p, &bci_p)) {
          return STACKWALKER_INDECIPHERABLE_FRAME;
        }
        if (!Method::is_valid_method(method_p)) {
          // we throw away everything we've gathered in this sample since
          // none of it is safe
          return STACKWALKER_GC_ACTIVE; // -2
        }
        this->handle_java_frame(top_frame);
        this->handle_interpreted_frame(top_frame, method_p, bci_p);
      } else if (top_frame.is_compiled_frame()) {
        frame enclosing_frame = top_frame;
        CompiledMethod* nm = enclosing_frame.cb()->as_compiled_method();
        method_p = nm->method();
        bci_p = -1;
        if (!is_decipherable_compiled_frame(thread, &enclosing_frame, nm)){
          return STACKWALKER_INDECIPHERABLE_FRAME;
        }
        this->handle_java_frame(top_frame);

        compiledFrameStreamForte st(thread, top_frame, false);

        for (; !st.at_end(); st.forte_next()) {
          Method* method = st.method();
          int bci = st.bci();

          if (!Method::is_valid_method(method)) {
            // we throw away everything we've gathered in this sample since
            // none of it is safe
            return STACKWALKER_GC_ACTIVE;
          }
          this->handle_compiled_frame(top_frame, method, bci, st.inlined());
          if (!st.inlined()) {
            break;
          }
        }
      } else {
        this->handle_misc_frame(top_frame);
      }
    } else {
      this->handle_misc_frame(top_frame);
    }
  } while (!top_frame.is_first_frame());
  while ((top_frame = next_c_frame(top_frame)).pc() != NULL) {
    this->handle_misc_frame(top_frame);
    count++;
  }
  return count;
}

// finds the top java frame using the stackwalker
class TopJavaFrameFinder : private StackWalker {
  frame top_java_frame;
  const int max_depth;
  int cur_depth;
public:

TopJavaFrameFinder(JavaThread* thread, int max_depth = -1): StackWalker(thread),
  max_depth(max_depth), cur_depth(0) {}

  void handle_frame(const frame& frame) { cur_depth++; }

  void handle_java_frame(const frame& frame) {
    top_java_frame = frame;
  }

  bool abort() const { return top_java_frame.pc() != NULL || (max_depth != -1 && cur_depth > max_depth); }

  frame find(frame top_frame, int *err_code = NULL) {
    int ret = walk(top_frame);
    if (err_code != NULL) {
      *err_code = ret < 0 ? ret : 0;
    }
    return top_java_frame;
  }
};

frame StackWalker::find_top_java_frame(frame top_frame, int max_depth, int* err_code) {
  return TopJavaFrameFinder(thread).find(top_frame, err_code);
}