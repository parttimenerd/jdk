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

#include "asgct2.hpp"
#include "gc/shared/collectedHeap.inline.hpp"
#include "runtime/frame.inline.hpp"
#include "runtime/thread.inline.hpp"
#include "runtime/vframe.inline.hpp"
#include "runtime/vframeArray.hpp"
#include "runtime/vframe_hp.hpp"
#include "jfr/recorder/stacktrace/stackWalker.hpp"
namespace asgct2 {

static void fill_call_trace_given_top(JavaThread* thd,
                                      CallTrace* trace,
                                      int depth,
                                      frame top_frame) {
  NoHandleMark nhm;
  assert(trace->frames != NULL, "trace->frames must be non-NULL");
  trace->frame_info = NULL;

  StackWalker st(thd, top_frame, false /* do not skip c frames */);

  int count = 0;
  for (; count < depth && !st.at_end(); st.next(), count++) {
    if (st.at_error()) {
      trace->num_frames = st.state();
      return;
    }
    switch (st.state()) {
      case STACKWALKER_INTERPRETED_FRAME:
        trace->frames[count] = {
          0,
          (uint8_t)FRAME_INTERPRETED,
          (uint8_t)CompLevel_none,
          st.bci(),
          st.method()->find_jmethod_id_or_null()
        };
        break;
      case STACKWALKER_COMPILED_FRAME:
        trace->frames[count] = {
          0,
          (uint8_t)(st.is_inlined() ? FRAME_INLINE : FRAME_JIT),
          (uint8_t)(st.method()->highest_comp_level()),
          st.bci(),
          st.method()->find_jmethod_id_or_null()
        };
        break;
      case STACKWALKER_NATIVE_FRAME:
        trace->frames[count] = {
          0,
          (uint8_t)FRAME_NATIVE,
          (uint8_t)CompLevel_none,
          -3,
          st.method()->find_jmethod_id_or_null()
        };
        break;
      case STACKWALKER_C_FRAME:
        trace->frames[count] = {
          st.base_frame()->pc(),
          (uint8_t)FRAME_CPP,
          (uint8_t)(st.base_frame()->is_stub_frame() ? CompLevel_all : CompLevel_none),
          -4,
          0
        };
        break;
      default:
        assert(false, "should never happen");
        trace->num_frames = static_cast<jint>(Error::UNKNOWN_JAVA);
    }
  }
  trace->num_frames = count;
}
}

extern "C" {
JNIEXPORT
void AsyncGetCallTrace2(asgct2::CallTrace *trace, jint depth, void* ucontext) {
  if (trace->env_id == NULL || JavaThread::is_thread_from_jni_environment_terminated(trace->env_id)) {
    // bad env_id, thread has exited or thread is exiting
    trace->num_frames = static_cast<jint>(asgct2::Error::THREAD_EXIT); // -8;
    return;
  }

  JavaThread* thread = JavaThread::thread_from_jni_environment(trace->env_id);

  if (thread->in_deopt_handler()) {
    // thread is in the deoptimization handler so return no frames
    trace->num_frames = static_cast<jint>(asgct2::Error::DEOPT); // -9
    return;
  }

  assert(JavaThread::current() == thread,
         "AsyncGetCallTrace2 must be called by the current interrupted thread");

  if (!JvmtiExport::should_post_class_load()) {
    trace->num_frames = static_cast<jint>(asgct2::Error::NO_CLASS_LOAD); // -1
    return;
  }

  if (Universe::heap()->is_gc_active()) {
    trace->num_frames = static_cast<jint>(asgct2::Error::GC_ACTIVE); // -2
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
      if (!thread->pd_get_top_frame_for_signal_handler(&ret_frame, ucontext, true)) {
        trace->num_frames = static_cast<jint>(asgct2::Error::UNKNOWN_NOT_JAVA);  // -3 unknown frame
        return;
      }
      fill_call_trace_given_top(thread, trace, depth, ret_frame);
    }
    break;
  default:
    // Unknown thread state
    trace->num_frames = static_cast<jint>(asgct2::Error::UNKNOWN_STATE); // -7
    break;
  }
}
}
