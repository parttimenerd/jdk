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

#include "prims/stackWalker.hpp"
#include "prims/jvmtiExport.hpp"
#include "runtime/javaThread.hpp"
#include "runtime/os.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "profile2.h"

int ASGST_Capabilities() {
  return ASGST_REGISTER_QUEUE | ASGST_MARK_FRAME;
}

struct _ASGST_Iterator {
  StackWalker walker;
  JavaThread *thread;
  int32_t options;
};

static int initStackWalker(_ASGST_Iterator *iterator, void *ucontext, int options, frame &ret_frame) {
  if (iterator->thread != nullptr) {
      iterator->thread->set_in_async_stack_walking(true);
  }
  iterator->walker = StackWalker(iterator->thread, ret_frame, (options & ASGST_INCLUDE_NON_JAVA_FRAMES) == 0);
  return ASGST_JAVA_TRACE;
}

// check if the frame has at least valid pointers
static bool is_c_frame_safe(frame fr) {
  return os::is_readable_pointer(fr.pc()) && os::is_readable_pointer(fr.sp()) && os::is_readable_pointer(fr.fp());
}

// like pd_fetch_frame_from_context but whithout using the JavaThread, only using os methods
static bool frame_from_context(frame* fr, void* ucontext) {
  ucontext_t* uc = (ucontext_t*) ucontext;
  frame ret_frame = os::fetch_frame_from_context(ucontext);
  if (!is_c_frame_safe(ret_frame)) {
#if COMPILER2_OR_JVMCI
    // C2 and JVMCI use ebp as a general register see if NULL fp helps
    frame ret_frame2(ret_frame.sp(), NULL, ret_frame.pc());
    if (!is_c_frame_safe(ret_frame2)) {
      // nothing else to try if the frame isn't good
      return false;
    }
    ret_frame = ret_frame2;
#else
    // nothing else to try if the frame isn't good
    return false;
#endif // COMPILER2_OR_JVMCI
  }
  *fr = ret_frame;
  return true;
}

static int initNonJavaStackWalker(_ASGST_Iterator* iter, void* ucontext, int32_t options) {
  bool include_non_java_frames = (options & ASGST_INCLUDE_NON_JAVA_FRAMES) != 0;

  frame ret_frame;
  if (!include_non_java_frames || !frame_from_context(&ret_frame, ucontext)) {
    return ASGST_NO_FRAME;
  }
  iter->thread = nullptr;
  int ret = initStackWalker(iter, ucontext, options, ret_frame);
  return ret > 0 ? ASGST_NON_JAVA_TRACE : ret;
}

// check current thread, return error or kind, set thread if available
static int ASGST_Check(JavaThread** thread) {
  Thread* raw_thread = Thread::current_or_null_safe();
  if (raw_thread == nullptr || !raw_thread->is_Java_thread()) {
    return ASGST_NON_JAVA_TRACE;
  }
  if ((*thread = JavaThread::cast(raw_thread))->is_exiting()) {
    return ASGST_THREAD_EXIT;
  }
  if ((*thread)->is_at_poll_safepoint()) {
    return ASGST_UNSAFE_STATE;
  }
  if ((*thread)->in_deopt_handler() || Universe::heap()->is_gc_active()) {
    return ASGST_NON_JAVA_TRACE;
  }
  if (!JvmtiExport::should_post_class_load()) {
    return ASGST_NO_CLASS_LOAD;
  }
  return ASGST_JAVA_TRACE;
}

int ASGST_CreateIter(_ASGST_Iterator* iterator, void* ucontext, int32_t options) {

  iterator->options = options;
  bool include_non_java_frames = (options & ASGST_INCLUDE_NON_JAVA_FRAMES) != 0;


  int kindOrError = ASGST_Check(&iterator->thread);
  // handle error case
  if (kindOrError <= 0) {
    return kindOrError;
  }

  // handle non-java case
  if (kindOrError > ASGST_JAVA_TRACE) {
    return initNonJavaStackWalker(iterator, ucontext, options);
  }

  // handle java case

  JavaThread* thread = iterator->thread;

  switch (thread->thread_state()) {
  case _thread_new:
  case _thread_uninitialized:
  case _thread_new_trans:
    // We found the thread on the threads list above, but it is too
    // young to be useful so return that there are no Java frames.
    return initNonJavaStackWalker(iterator, ucontext, options);
  case _thread_in_native:
  case _thread_in_native_trans:
  case _thread_blocked:
  case _thread_blocked_trans:
  case _thread_in_vm:
  case _thread_in_vm_trans:
    {
      frame ret_frame;
      // param isInJava == false - indicate we aren't in Java code
      if (!thread->pd_get_top_frame_for_signal_handler(&ret_frame, ucontext, false)) {
        if (!include_non_java_frames || !thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, false, true)) {
          return ASGST_NO_FRAME;
        }
      } else {
        if (!thread->has_last_Java_frame()) {
          if (!include_non_java_frames) {
            return ASGST_NO_TOP_JAVA_FRAME;
          }
        }
      }
      return initStackWalker(iterator, ucontext, options, ret_frame);
    }
    break;
  case _thread_in_Java:
  case _thread_in_Java_trans:
    {
      frame ret_frame;
      if (!thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, true, include_non_java_frames)) {
        // check without forced ucontext again
        if (!include_non_java_frames || !thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, true, false)) {
          return ASGST_NO_TOP_JAVA_FRAME;
        }
      }
      return initStackWalker(iterator, ucontext, options, ret_frame);
    }
    break;
  default:
    // Unknown thread state
    return ASGST_NO_FRAME;
  }
  return 0;
}

int ASGST_NextFrame(ASGST_Iterator *iter, ASGST_Frame *frame) {
  int state = ASGST_State(iter);
  if (state <= 0) {
    frame->type = 0;
    return state;
  }
  frame->comp_level = -1;
  frame->bci = -1;
  auto m = iter->walker.method();
  frame->method_id = m != nullptr ? m->find_jmethod_id_or_null() : nullptr;
  auto f = iter->walker.base_frame();
  frame->pc = f->pc();
  frame->sp = f->sp();
  frame->fp = f->fp();
  if (iter->walker.is_bytecode_based_frame()) {
    frame->comp_level = iter->walker.compilation_level(),
    frame->bci = iter->walker.bci();
  }

  if (iter->walker.is_bytecode_based_frame()) {
    frame->type = iter->walker.is_inlined() ? ASGST_FRAME_JAVA_INLINED : ASGST_FRAME_JAVA;
  } else if (iter->walker.is_native_frame()) {
    frame->type = ASGST_FRAME_NATIVE;
  } else {
    frame->type = ASGST_FRAME_CPP;
  }
  iter->walker.next();
  return 1;
}

int ASGST_State(ASGST_Iterator *iter) {
  if (iter == nullptr) {
    return ASGST_NO_FRAME;
  }
  if (iter->walker.at_end()) {
    return ASGST_NO_FRAME;
  }
  return std::min(iter->walker.state(), 1);
}

void ASGST_DestroyIter(ASGST_Iterator* iter) {
  if (iter->thread != nullptr) {
    iter->thread->set_in_async_stack_walking(false);
  }
}

class IterRAII {
  ASGST_Iterator* iter;
public:
  IterRAII(ASGST_Iterator* iter) : iter(iter) {}
  ~IterRAII() {
    ASGST_DestroyIter(iter);
  }
};

int ASGST_RunWithIterator(void* ucontext, int options, void (*fun)(ASGST_Iterator*, void*), void* argument) {
  int8_t iter[sizeof(ASGST_Iterator)]; // no need for default constructor
  ASGST_Iterator* iterator = (ASGST_Iterator*) iter;
  IterRAII raii(iterator); // destroy iterator on exit
  int ret = ASGST_CreateIter(iterator, ucontext, options);
  if (ret <= 0) {
    return ret;
  }
  fun(iterator, argument);
  return 1;
}

// state or -1
// no JVMTI_THREAD_STATE_INTERRUPTED, limited JVMTI_THREAD_STATE_SUSPENDED
int ASGST_ThreadState() {
  JavaThread* thread;
  if (ASGST_Check(&thread) <= 0) {
    return -1;
  }
  int state = JVMTI_THREAD_STATE_ALIVE;

  if (thread->is_suspended()) {
    state |= JVMTI_THREAD_STATE_SUSPENDED;
  }
    switch (thread->thread_state()) {
    case _thread_in_native:
    case _thread_in_native_trans:
      state |= JVMTI_THREAD_STATE_IN_NATIVE;
      break;
    case _thread_blocked:
    case _thread_blocked_trans:
      state |= JVMTI_THREAD_STATE_BLOCKED_ON_MONITOR_ENTER;
      break;
    case _thread_in_vm:
    case _thread_in_Java:
    case _thread_new:
      {
      state = JVMTI_THREAD_STATE_ALIVE;
      if (thread->is_carrier_thread_suspended() || thread->is_suspended()) {
        // Suspended non-virtual thread.
        state |= JVMTI_THREAD_STATE_SUSPENDED;
      }
      break;
      }
    default:
      break;
  }
  return state;
}