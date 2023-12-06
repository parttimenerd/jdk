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

#include "jni.h"
#include "memory/allocation.hpp"
#include "memory/resourceArea.hpp"
#include "oops/instanceKlass.hpp"
#include "oops/method.hpp"
#include "precompiled.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include "prims/jvmtiEnvBase.hpp"
#include "prims/stackWalker.hpp"
#include "prims/jvmtiExport.hpp"
#include "profile2.h"
#include "runtime/atomic.hpp"
#include "runtime/frame.hpp"
#include "runtime/handshake.hpp"
#include "runtime/interfaceSupport.inline.hpp"
#include "runtime/javaThread.hpp"
#include "runtime/jniHandles.hpp"
#include "runtime/os.hpp"
#include "runtime/safefetch.hpp"
#include "runtime/safepointMechanism.hpp"
#include "runtime/stackWatermark.hpp"
#include "runtime/stackWatermarkKind.hpp"
#include "utilities/globalDefinitions.hpp"
#include "code/compiledMethod.hpp"


int JFRLL_Capabilities() {
  return JFRLL_MARK_FRAME;
}

struct _JFRLL_InitialIteratorArgs {
  frame _frame;
  bool _allowThreadLastFrameUse;
  bool _invalidSpAndFp;
  bool _switchToThreadBased;
  StackWalkerMiscArguments _misc;

  void reset() {
    _frame = {};
    _allowThreadLastFrameUse = true;
    _invalidSpAndFp = false;
    _switchToThreadBased = false;
    _misc = {};
  }
};

struct _JFRLL_Iterator {
  StackWalker walker;
  JavaThread *thread;
  frame top_frame;
  int options = options;
  bool invalidSpAndFp = false;
  bool switchToThreadBased = false;
  _JFRLL_InitialIteratorArgs initialArgs;

  void reset() {
    thread = nullptr;
    top_frame = frame();
    options = 0;
    invalidSpAndFp = false;
    switchToThreadBased = false;
    initialArgs.reset();
  }
};

static int initStackWalker(_JFRLL_Iterator *iterator, int options, frame frame, bool allow_thread_last_frame_use = true, StackWalkerMiscArguments misc = {}) {
  iterator->walker = StackWalker(iterator->thread, frame,
    (options & JFRLL_INCLUDE_NON_JAVA_FRAMES) == 0,
    (options & JFRLL_END_ON_FIRST_JAVA_FRAME) != 0,
    allow_thread_last_frame_use, misc);
  iterator->options = options;
  return JFRLL_JAVA_TRACE;
}

// check if the frame has at least valid pointers
static bool is_c_frame_safe(frame fr) {
  return os::is_readable_pointer2(fr.pc()) && os::is_readable_pointer2(fr.sp()) && os::is_readable_pointer2(fr.fp());
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

static int initNonJavaStackWalker(_JFRLL_Iterator* iter, void* ucontext, int32_t options) {
  bool include_non_java_frames = (options & JFRLL_INCLUDE_NON_JAVA_FRAMES) != 0;

  frame ret_frame;
  if (!include_non_java_frames || !frame_from_context(&ret_frame, ucontext)) {
    return JFRLL_NO_FRAME;
  }
  iter->thread = nullptr;
  iter->initialArgs._frame = ret_frame;
  int ret = initStackWalker(iter, options, ret_frame);
  return ret > 0 ? JFRLL_NON_JAVA_TRACE : ret;
}

// check current thread, return error or kind, set thread if available
static int JFRLL_Check(JavaThread** thread, bool allow_safepoints) {
  Thread* raw_thread = Thread::current_or_null_safe();
  if (raw_thread == nullptr || !raw_thread->is_Java_thread()) {
    return JFRLL_NON_JAVA_TRACE;
  }
  if ((*thread = JavaThread::cast(raw_thread))->is_exiting()) {
    return JFRLL_THREAD_EXIT;
  }
  if (!allow_safepoints && (*thread)->is_at_safepoint()) {
    return JFRLL_UNSAFE_STATE;
  }
  if ((*thread)->in_deopt_handler() || Universe::heap()->is_gc_active()) {
    return JFRLL_NON_JAVA_TRACE;
  }
  return JFRLL_JAVA_TRACE;
}

// @return error or kind
int JFRLL_CreateIter(_JFRLL_Iterator* iterator, void* ucontext, int32_t options, bool allow_safepoints) {
  bool include_non_java_frames = (options & JFRLL_INCLUDE_NON_JAVA_FRAMES) != 0;


  int kindOrError = JFRLL_Check(&iterator->thread, allow_safepoints);
  // handle error case
  if (kindOrError <= 0) {
    return kindOrError;
  }

  // handle non-java case
  if (kindOrError > JFRLL_JAVA_TRACE) {
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
      if (!thread->pd_get_top_frame_for_signal_handler(&ret_frame, ucontext, true)) {
        if (!include_non_java_frames || !thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, true, true)) {
          if (!thread->has_last_Java_frame()) {
            if (!include_non_java_frames) {
              return JFRLL_NO_TOP_JAVA_FRAME;
            }
          }
          return JFRLL_NO_FRAME;
        }
      }
      iterator->initialArgs._frame = ret_frame;
      return initStackWalker(iterator, options, ret_frame);
    }
    break;
  case _thread_in_Java:
  case _thread_in_Java_trans:
    {
      frame ret_frame;
      if (!thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, true, include_non_java_frames)) {
        // check without forced ucontext again
        if (!include_non_java_frames || !thread->pd_get_top_frame_for_profiling(&ret_frame, ucontext, true, false)) {
          return JFRLL_NO_TOP_JAVA_FRAME;
        }
      }
      iterator->initialArgs._frame = ret_frame;
      return initStackWalker(iterator, options, ret_frame);
    }
    break;
  default:
    // Unknown thread state
    return JFRLL_NO_FRAME;
  }
  return 0;
}

int JFRLL_CreateIterFromFrame(_JFRLL_Iterator* iterator, void* sp, void* fp, void* pc, int32_t options, bool allow_safepoints) {

  bool include_non_java_frames = (options & JFRLL_INCLUDE_NON_JAVA_FRAMES) != 0;


  int kindOrError = JFRLL_Check(&iterator->thread, allow_safepoints);
  // handle error case
  if (kindOrError <= 0) {
    return kindOrError;
  }

  frame f{sp, fp, pc};
  iterator->initialArgs._frame = f;
  iterator->thread = JavaThread::current_or_null();
  // handle non-java case
  if (kindOrError > JFRLL_JAVA_TRACE) {
    int ret = initStackWalker(iterator, options, f);
    return ret < 1 ? ret : JFRLL_NON_JAVA_TRACE;
  }
  int ret = initStackWalker(iterator, options, f);
  return ret < 1 ? ret : JFRLL_JAVA_TRACE;
}


void resetFrame(JFRLL_Frame* frame) {
  frame->pc = nullptr;
  frame->sp = nullptr;
  frame->fp = nullptr;
  frame->bci = -1;
  frame->method = nullptr;
  frame->comp_level = -1;
  frame->type = 0;
}

void setFrame(JFRLL_Iterator *iter, JFRLL_Frame *frame, bool invalidSpAndFp = false) {
  auto* f = iter->walker.base_frame();
  frame->pc = f->pc();
  frame->sp = invalidSpAndFp ? nullptr : f->sp();
  frame->fp = invalidSpAndFp ? nullptr : f->fp();
}

int JFRLL_NextFrame(JFRLL_Iterator *iter, JFRLL_Frame *frame) {
  resetFrame(frame); // just to be on the safe side
  int state = JFRLL_State(iter);
  if (state <= 0) {
    return state;
  }
  if (iter->walker.is_at_first_bc_java_frame()) {
    // handle JFRLL_END_ON_FIRST_JAVA_FRAME
    if (state < 0) {
      return state;
    }
    frame->type = JFRLL_FRAME_JAVA;
    setFrame(iter, frame);
    iter->walker.next();
    return 0;
  }
  auto m = iter->walker.method();
  frame->method = (JFRLL_Method)m;
  setFrame(iter, frame);
  if (iter->walker.is_bytecode_based_frame()) {
    frame->comp_level = iter->walker.compilation_level(),
    frame->bci = iter->walker.bci();
  }
  if (iter->walker.is_bytecode_based_frame()) {
    frame->type = iter->walker.is_inlined() ? JFRLL_FRAME_JAVA_INLINED : JFRLL_FRAME_JAVA;
  } else if (iter->walker.is_native_frame()) {
    frame->type = JFRLL_FRAME_JAVA_NATIVE;
  } else {
    frame->type = JFRLL_FRAME_NON_JAVA;
  }
  if (iter->switchToThreadBased && !iter->walker.is_inlined() && iter->top_frame.pc() != nullptr && !iter->walker.at_first_java_frame() && (iter->top_frame.sp() >= frame->sp || iter->top_frame.fp() >= frame->fp)) {
    // we take switching to the iterator at safepoint frame into account,
    // if the current frame is not inlined (and the top frame pc is not null) and the walker
    // is not at its first bytecode backed java frame
    iter->switchToThreadBased = false;
    iter->invalidSpAndFp = false;
    int8_t walkerBytes[sizeof(StackWalker)]; // backup the old walker
    StackWalker* walker = (StackWalker*) walkerBytes;
    *walker = iter->walker;
    assert(iter->top_frame.pc() != nullptr, "non nulllll");
    initStackWalker(iter, iter->options,
      {iter->top_frame.sp(), iter->top_frame.fp(), iter->top_frame.pc()}, false);
    if ((Method*)frame->method == iter->walker.method() && frame->sp == nullptr) {
      // last frame from old walker is similar to current frame from new walker
      // therefor skip the old walker frame (contains less information)
      // typical for poll-at-return handshakes
      return JFRLL_NextFrame(iter, frame);
    }
    if (iter->walker.at_end_or_error() && !walker->at_end_or_error()) {
      // use the old walker
      iter->walker = *walker;
      if ((Method*)frame->method == iter->walker.method() && frame->sp == nullptr) {
        // last frame from old walker is similar to current frame from new walker
        // therefor skip the old walker frame (contains less information)
        // typical for poll-at-return handshakes
        return JFRLL_NextFrame(iter, frame);
      }
      return 1;
    }
  } else {
    iter->walker.next();
  }
  return 1;
}

int JFRLL_State(JFRLL_Iterator *iter) {
  if (iter == nullptr) {
    return JFRLL_NO_FRAME;
  }
  if (iter->walker.at_end()) {
    return JFRLL_NO_FRAME;
  }
  if (iter->walker.at_error()) {
    // the error code start at -1
    // custom error codes start at -17
    return iter->walker.error() - 16;
  }
  return 1;
}

class IterRAII {
  JavaThread* thread;
  bool old;
public:
  IterRAII() {
    Thread* t = Thread::current_or_null_safe();
    thread = t == nullptr || !t->is_Java_thread() ? nullptr : JavaThread::cast(t);
    if (thread != nullptr) {
      old = thread->in_async_stack_walking();
      thread->set_in_async_stack_walking(true);
    }
  }
  ~IterRAII() {
    if (thread != nullptr && !old) {
      thread->set_in_async_stack_walking(old);
    }
  }
};

int JFRLL_RunWithIterator(void* ucontext, int options, JFRLL_IteratorHandler fun, void* argument) {
  int8_t iter[sizeof(JFRLL_Iterator)]; // no need for default constructor
  JFRLL_Iterator* iterator = (JFRLL_Iterator*) iter;
  iterator->reset();
  IterRAII raii; // destroy iterator on exit
  int ret = JFRLL_CreateIter(iterator, ucontext, options, false);
  if (ret <= 0) {
    return ret;
  }
  fun(iterator, argument);
  return ret;
}

int JFRLL_RunWithIteratorFromFrame(void* sp, void* fp, void* pc, int options, JFRLL_IteratorHandler fun, void* argument) {
  int8_t iter[sizeof(JFRLL_Iterator)]; // no need for default constructor
  JFRLL_Iterator* iterator = (JFRLL_Iterator*) iter;
  iterator->reset();
  IterRAII raii; // destroy iterator on exit
  int ret = JFRLL_CreateIterFromFrame(iterator, sp, fp, pc, options, false);
  if (ret <= 0) {
    return ret;
  }
  fun(iterator, argument);
  return ret;
}

void JFRLL_RewindIterator(JFRLL_Iterator* iterator) {
  iterator->switchToThreadBased = iterator->initialArgs._switchToThreadBased;
  iterator->invalidSpAndFp = iterator->initialArgs._invalidSpAndFp;
  initStackWalker(iterator, iterator->options, iterator->initialArgs._frame,
    iterator->initialArgs._allowThreadLastFrameUse, iterator->initialArgs._misc);
}

// state or -1
// no JVMTI_THREAD_STATE_INTERRUPTED, limited JVMTI_THREAD_STATE_SUSPENDED
int JFRLL_ThreadState() {
  JavaThread* thread;
  if (JFRLL_Check(&thread, true) <= 0) {
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

const char* typeToStr(int type) {
  switch (type) {
    case JFRLL_FRAME_JAVA:
      return "java";
    case JFRLL_FRAME_JAVA_INLINED:
      return "java_inlined";
    case JFRLL_FRAME_JAVA_NATIVE:
      return "java_native";
    case JFRLL_FRAME_NON_JAVA:
      return "non_java";
    case 0:
      return "error";
    default:
      return "unknown";
  }
}

class JFRLLQueueElementHandlerImpl : public JFRLLQueueElementHandler {
  int options;
  JFRLL_Handler fun;
  void* queue_arg;
public:
  JFRLLQueueElementHandlerImpl(int options, JFRLL_Handler fun, void* queue_arg) : options(options), fun(fun), queue_arg(queue_arg) {}
  void operator()(JFRLLQueueElement* element, frame top_frame, CompiledMethod* cm) override {

    int8_t iter[sizeof(JFRLL_Iterator)]; // no need for default constructor
    JFRLL_Iterator* iterator = (JFRLL_Iterator*) iter;
    iterator->reset();
    iterator->thread = JavaThread::current();
    IterRAII raii; // destroy iterator on exit

    assert(element != nullptr, "element is null");
    if (element->pc() == nullptr && top_frame.pc() == nullptr) {
      return;
    }
    // fExtended is the current frame + info from the element
    frame fExtended = frame(element->sp(), element->fp(),
      element->pc() != nullptr ? (address)element->pc() : top_frame.pc());
    StackWalkerMiscArguments misc{cm != nullptr ? cm->method() : nullptr, (address)element->bcp()};
    initStackWalker(iterator, options, fExtended, false, misc);
    iterator->initialArgs._allowThreadLastFrameUse = false;
    iterator->initialArgs._frame = fExtended;
    iterator->initialArgs._misc = misc;

    if (element->pc() != nullptr && fExtended.pc() != top_frame.pc() && top_frame.pc() != nullptr) {
      iterator->switchToThreadBased = true;
      iterator->invalidSpAndFp = true;
      iterator->top_frame = top_frame;
      iterator->initialArgs._invalidSpAndFp = true;
      iterator->initialArgs._switchToThreadBased = true;
    }
    // we ignore top native frames
    while (iterator->walker.is_native_frame()) {
      iterator->walker.next();
    }
    // call the handler
    fun(iterator, queue_arg, element->argument());
    return;
  }
};

JFRLL_Queue* JFRLL_RegisterQueue(JNIEnv* env, int size, int options, JFRLL_Handler fun, void* argument) {
  JavaThread* thread = env == nullptr ? JavaThread::current_or_null() : JavaThread::thread_from_jni_environment(env);
  if (!os::is_readable_pointer2(thread->handshake_state()) || thread->is_terminated()) {
    return nullptr;
  }
  return (JFRLL_Queue*)thread->handshake_state()->register_jfrll_queue(thread, size, new JFRLLQueueElementHandlerImpl(options, fun, argument));
}

bool JFRLL_DeregisterQueue(JNIEnv* env, JFRLL_Queue* queue) {
  JavaThread* thread = env == nullptr ? JavaThread::current_or_null() : JavaThread::thread_from_jni_environment(env);
  if (thread == nullptr || !os::is_readable_pointer2(thread->handshake_state()) || thread->is_terminated()) {
    return false;
  }
  return (JFRLL_Queue*)thread->handshake_state()->remove_jfrll_queue((JFRLLQueue*)queue);
}


class JFRLLQueueOnSafepointHandlerImpl : public JFRLLQueueOnSafepointHandler {
  int options;
  bool offer_iterator;
  JFRLL_OnQueueSafepointHandler fun;
  void* on_queue_arg;
public:
  JFRLLQueueOnSafepointHandlerImpl(int options, bool offer_iterator,
    JFRLL_OnQueueSafepointHandler fun, void* on_queue_arg) :
    options(options), offer_iterator(offer_iterator), fun(fun), on_queue_arg(on_queue_arg) {}
  void operator()(JFRLLQueue* queue, frame top_frame, CompiledMethod* cm) override {
    if (offer_iterator) {
      int8_t iter[sizeof(JFRLL_Iterator)]; // no need for default constructor
      JFRLL_Iterator* iterator = (JFRLL_Iterator*) iter;
      iterator->reset();
      iterator->thread = JavaThread::current();
      IterRAII raii; // destroy iterator on exit
      if (top_frame.pc() == nullptr) {
        return;
      }
      frame f = top_frame;
      StackWalkerMiscArguments misc{cm != nullptr ? cm->method() : nullptr,
        (address)top_frame.potential_interpreter_frame_bcp_safe()};
      initStackWalker(iterator, options, f, false, misc);
      iterator->initialArgs._frame = f;
      iterator->initialArgs._allowThreadLastFrameUse = false;
      iterator->initialArgs._misc = misc;
      fun((JFRLL_Queue*)queue, iterator, on_queue_arg);
    } else {
      fun((JFRLL_Queue*)queue, nullptr, on_queue_arg);
    }
  }
};

void JFRLL_SetOnQueueProcessingStart(JFRLL_Queue* queue, int options, bool offerIterator, JFRLL_OnQueueSafepointHandler before, void* arg) {
  JFRLLQueue* q = (JFRLLQueue*)queue;
  if (before == nullptr) {
    q->set_before(nullptr);
  } else {
    q->set_before(new JFRLLQueueOnSafepointHandlerImpl(options, offerIterator, before, arg));
  }
}

void JFRLL_SetOnQueueProcessingEnd(JFRLL_Queue* queue, int options, bool offerIterator, JFRLL_OnQueueSafepointHandler after, void* arg) {
  JFRLLQueue* q = (JFRLLQueue*)queue;
  if (after == nullptr) {
    q->set_after(nullptr);
  } else {
    q->set_after(new JFRLLQueueOnSafepointHandlerImpl(options, offerIterator, after, arg));
  }
}

struct EnqueueFindFirstJavaFrameStruct {
  int kindOrError; // 1, no error, 2 no Java frame, < 0 error
  JFRLLQueueElement elem;
  JavaThread* thread;
};

void enqueueFindFirstJavaFrame(JFRLL_Iterator* iterator, void* arg) {
  auto argument = (EnqueueFindFirstJavaFrameStruct*)arg;
  if (!iterator->walker.walk_till_first_java_frame()) {
    argument->kindOrError = iterator->walker.at_error() ? iterator->walker.error() : JFRLL_NON_JAVA_TRACE;
    return;
  }
  argument->kindOrError = 1;
  const frame* base_frame = iterator->walker.base_frame();
  argument->elem = {base_frame->pc(), base_frame->fp(), base_frame->sp(), nullptr, base_frame->potential_interpreter_frame_bcp_safe()};
  argument->thread = iterator->thread;
}

int JFRLL_Enqueue(JFRLL_Queue* queue, void* ucontext, void* argument) {
  if (queue == nullptr) {
    return JFRLL_ENQUEUE_NO_QUEUE;
  }
  JFRLLQueue *q = (JFRLLQueue*)queue;
  if (q->is_full()) {
    return JFRLL_ENQUEUE_FULL_QUEUE;
  }
  EnqueueFindFirstJavaFrameStruct runArgument;
  int kind = JFRLL_RunWithIterator(ucontext, JFRLL_END_ON_FIRST_JAVA_FRAME, &enqueueFindFirstJavaFrame, &runArgument);
  if (kind != JFRLL_JAVA_TRACE) {
    return kind;
  }
  if (runArgument.kindOrError != JFRLL_JAVA_TRACE) {
    return 0;
  }
  assert(q->in_current_thread(), "must be called from the same thread");
  if (q->is_full()) {
    return JFRLL_ENQUEUE_FULL_QUEUE;
  }
  JFRLLQueueElement elem = runArgument.elem.set_argument(argument);
  JFRLLQueuePushResult worked = JavaThread::current()->handshake_state()->jfrll_enqueue(q, elem);
  switch (worked) {
    case JFRLL_QUEUE_PUSH_SUCCESS:
      return JFRLL_JAVA_TRACE;
    case JFRLL_QUEUE_PUSH_CLOSED:
      return JFRLL_ENQUEUE_OTHER_ERROR;
    case JFRLL_QUEUE_PUSH_FULL:
      return JFRLL_ENQUEUE_FULL_QUEUE;
    default:
      assert(false, "unknown result");
      return -1;
  }
}

int JFRLL_GetEnqueuableElement(void* ucontext, JFRLL_QueueElement* element) {
  EnqueueFindFirstJavaFrameStruct runArgument;
  int kind = JFRLL_RunWithIterator(ucontext, JFRLL_END_ON_FIRST_JAVA_FRAME, &enqueueFindFirstJavaFrame, &runArgument);
  if (kind != JFRLL_JAVA_TRACE) {
    return kind;
  }
  if (runArgument.kindOrError != JFRLL_JAVA_TRACE) {
    return runArgument.kindOrError;
  }
  *element = {runArgument.elem.pc(), runArgument.elem.fp(), runArgument.elem.sp(), nullptr};
  return 1;
}

JFRLL_QueueElement* JFRLL_GetQueueElement(JFRLL_Queue* queue, int n) {
  JFRLLQueue* q = (JFRLLQueue*)queue;
  auto elem = q->get(n < 0 ? (q->length() + n) : n);
  return (JFRLL_QueueElement*)elem;
}

JFRLL_QueueSizeInfo JFRLL_GetQueueSizeInfo(JFRLL_Queue* queue) {
  JFRLLQueue* q = (JFRLLQueue*)queue;
  return {q->length(), q->capacity(), q->attempts()};
}

void JFRLL_ResizeQueue(JFRLL_Queue* queue, int size) {
  JFRLLQueue* q = (JFRLLQueue*)queue;
  q->trigger_resize(size);
}

static bool isClassValid(JFRLL_Class klass) {
  Klass* k = (Klass*)klass;
  return os::is_readable_pointer2(k);
}

// checks that a method and it's constant pool are readable
static bool isMethodValid(JFRLL_Method method) {
  Method* m = (Method*)method;
  return os::is_readable_pointer2(m) && Method::is_valid_method(m) &&
      os::is_readable_pointer2(m->constMethod()) &&
      os::is_readable_pointer2(m->constMethod()->constants()) &&
      isClassValid((JFRLL_Class) m->method_holder());
}

jmethodID JFRLL_MethodToJMethodID(JFRLL_Method method) {
  if (!isMethodValid(method)) {
    return nullptr;
  }
  return (jmethodID)((Method*)method)->find_jmethod_id_or_null();
}

JFRLL_Method JFRLL_JMethodIDToMethod(jmethodID methodID) {
  Method** m = (Method**)methodID;
  if (methodID == nullptr || *m == nullptr) {
    return nullptr;
  }
  return (JFRLL_Method)*m;
}

void writeField(Symbol* symbol, char* char_field, jint* length_field) {
  int& length = *length_field;
  if (length == 0 || char_field == nullptr) {
    length = 0;
    return;
  }
  if (symbol == nullptr) {
    if (length > 0) {
      *char_field = 0;
      length = 0;
    }
  } else {
    symbol->as_C_string(char_field, length);
    length = symbol->utf8_length();
  }
}

void nullField(char* char_field, jint* length_field) {
  if (*length_field > 0 && char_field != nullptr) {
    *char_field = '\0';
    *length_field = 0;
  } else {
    *length_field = 0;
  }
}

void JFRLL_GetMethodInfo(JFRLL_Method method, JFRLL_MethodInfo* info) {
  if (!isMethodValid(method)) {
    nullField(info->method_name, &info->method_name_length);
    nullField(info->signature, &info->signature_length);
    nullField(info->generic_signature, &info->generic_signature_length);
    info->modifiers = 0;
    info->klass = nullptr;
    info->idnum = 0;
    info->class_idnum = 0;
    return;
  }
  Method *m = (Method*)method;
  auto cm = m->constMethod();

  auto constants = cm->constants();

  InstanceKlass *klass = constants->pool_holder();
  info->klass = (JFRLL_Class)klass;
  info->idnum = cm->orig_method_idnum();
  info->class_idnum = klass->orig_idnum();
  writeField(m->name(), info->method_name, &info->method_name_length);
  writeField(m->signature(), info->signature, &info->signature_length);
  writeField(m->generic_signature(), info->generic_signature, &info->generic_signature_length);
  info->modifiers = m->access_flags().get_flags();
}

jint JFRLL_GetMethodIdNum(JFRLL_Method method) {
  if (!isMethodValid(method)) {
    return 0;
  }
  return ((Method*)method)->orig_method_idnum();
}

int JFRLL_GetMethodLineNumberTable(JFRLL_Method method, JFRLL_MethodLineNumberEntry* entries, int length) {
  int num_entries = 0;
  Method* m = (Method*)method;
  CompressedLineNumberReadStream stream(m->compressed_linenumber_table());
  while (stream.read_pair()) {
    if (num_entries < length) {
      *entries = {.start_bci = stream.line(), .line_number = stream.bci()};
      entries++;
    }
    num_entries++;
  }
  return num_entries;
}

jint JFRLL_GetMethodLineNumber(JFRLL_Method method, jint bci) {
  if (!isMethodValid(method) || bci == -1) {
    return -1;
  }
  Method* m = (Method*)method;
  CompressedLineNumberReadStream stream(m->compressed_linenumber_table());
  int last_line = -1;
  while (stream.read_pair()) {
    if (stream.bci()) {
      if (stream.bci() > bci) {
        break;
      }
      last_line = stream.line();
    }
  }
  return last_line;
}

void JFRLL_GetClassInfo(JFRLL_Class klass, JFRLL_ClassInfo *info) {
  if (!isClassValid(klass)) {
    nullField(info->class_name, &info->class_name_length);
    nullField(info->generic_class_name, &info->generic_class_name_length);
    info->modifiers = 0;
    info->idnum = 0;
    return;
  }
  Klass *k = (Klass*)klass;
  writeField(k->name(), info->class_name, &info->class_name_length);
  if (k->is_instance_klass()) {
    InstanceKlass *ik = (InstanceKlass*)k;
    writeField(ik->generic_signature(), info->generic_class_name, &info->generic_class_name_length);
    info->idnum = ik->orig_idnum();
  } else {
    nullField(info->generic_class_name, &info->generic_class_name_length);
    info->idnum = 0;
  }
  info->modifiers = k->access_flags().get_flags();
}

jlong JFRLL_GetClassIdNum(JFRLL_Class klass) {
  return !isClassValid(klass) || !((Klass*)klass)->is_instance_klass() ? 0 : ((InstanceKlass*)klass)->orig_idnum();
}

JFRLL_Class JFRLL_GetClass(JFRLL_Method method) {
  if (!isMethodValid(method)) {
    return nullptr;
  }
  Method *m = (Method*)method;
  InstanceKlass *klass = m->method_holder();
  return (JFRLL_Class)klass;
}

JFRLL_Class JFRLL_JClassToClass(jclass klass) {
  return (JFRLL_Class)klass;
}

jclass JFRLL_ClassToJClass(JFRLL_Class klass) {
  JavaThread* thread = JavaThread::current_or_null();
  return isClassValid(klass) && thread != nullptr ?
    (jclass)JNIHandles::make_local(thread, ((Klass*)klass)->java_mirror()) : nullptr;
}

class SmallKlassDeallocationHandler : public CHeapObj<mtServiceability> {
  JFRLL_ClassUnloadHandler _handler;
  void* _arg;
  SmallKlassDeallocationHandler* _next;
public:
  SmallKlassDeallocationHandler(JFRLL_ClassUnloadHandler handler, void* arg):
    _handler(handler), _arg(arg) {}
  SmallKlassDeallocationHandler(): _handler(nullptr), _arg(nullptr) {}

  void call(InstanceKlass* klass, bool redefined) {
    _handler((JFRLL_Class)klass, (JFRLL_Method*)klass->methods()->data(), klass->methods()->size(), redefined, _arg);
  }

  bool has_handler(JFRLL_ClassUnloadHandler handler) const {
    return _handler == handler;
  }

  bool has_argument(void* arg) const { return _arg == arg; }

  SmallKlassDeallocationHandler* next() { return _next; }

  void set_next(SmallKlassDeallocationHandler* next) { _next = next; }

  ~SmallKlassDeallocationHandler() {
    if (_next != nullptr) {
      delete _next;
    }
  }
};

class KlassDeallocationHandlerImpl : public KlassDeallocationHandler {
  SmallKlassDeallocationHandler* handlers;
  bool registered = false;
  void register_if_needed() {
    InstanceKlass::add_deallocation_handler(this);
  }
public:
  KlassDeallocationHandlerImpl() {}

  void add(JFRLL_ClassUnloadHandler handler, void* arg) {
    register_if_needed();
    auto* n = new SmallKlassDeallocationHandler(handler, arg);
    n->set_next(handlers);
    handlers = n;
  }

  void call(InstanceKlass* klass, bool redefined) {
    SmallKlassDeallocationHandler* cur = handlers;
    while (cur != nullptr) {
      cur->call(klass, redefined);
      cur = cur->next();
    }
  }

  bool remove(JFRLL_ClassUnloadHandler handler, void* arg) {
    SmallKlassDeallocationHandler* prev = nullptr;
    auto* cur = handlers;
    bool found = false;
    while (cur != nullptr) {
      if (cur->has_handler(handler) && cur->has_argument(arg)) {
        auto next = cur->next();
        delete cur;
        found = true;
        if (prev == nullptr) {
          handlers = next;
        } else {
          prev->set_next(next);
        }
        cur = next;
      } else {
        prev = cur;
        cur = cur->next();
      }
    }
    return found;
  }

  ~KlassDeallocationHandlerImpl() {
    delete handlers;
  }
};

std::mutex klassDeallocationHandlersMutex;
KlassDeallocationHandlerImpl klassDealloc;

void JFRLL_RegisterClassUnloadHandler(JFRLL_ClassUnloadHandler handler, void* arg) {
  std::lock_guard<std::mutex> lock(klassDeallocationHandlersMutex);
  klassDealloc.add(handler, arg);
}

bool JFRLL_DeregisterClassUnloadHandler(JFRLL_ClassUnloadHandler handler, void* arg) {
  std::lock_guard<std::mutex> lock(klassDeallocationHandlersMutex);
  return klassDealloc.remove(handler, arg);
}

class JFRLLFrameMark : public CHeapObj<mtServiceability> {
  JavaThread* _thread = nullptr;
  JFRLL_FrameMarkHandler _handler;
  int _options;
  void* _argument;
  std::atomic<void*> _sp;
  JFRLLFrameMark* _next = nullptr;
public:
  JFRLLFrameMark(JavaThread* thread, JFRLL_FrameMarkHandler handler, int options, void* argument, void* sp):
    _thread(thread),_handler(handler), _options(options), _argument(argument), _sp(sp) {}

  void call(JFRLL_Iterator* iterator) {
    _handler((JFRLL_FrameMark*)this, iterator, _argument);
  }

  void call(frame fr) {
    int8_t iter[sizeof(JFRLL_Iterator)]; // no need for default constructor
    JFRLL_Iterator* iterator = (JFRLL_Iterator*) iter;
    iterator->reset();
    IterRAII raii; // destroy iterator on exit
    int ret = JFRLL_CreateIterFromFrame(iterator, fr.sp(), fr.fp(), fr.pc(), _options, true);
    if (ret == 0) {
      return;
    }
    assert(ret == 1, "Expect Java trace");
    call(iterator);
  }

  JFRLLFrameMark* next() const { return _next; }

  void set_next(JFRLLFrameMark* next) { _next = next; }

  bool applicable(void* sp) const { return this->_sp != nullptr && this->_sp <= sp; }

  void* sp() const { return _sp; }

  void update(void* sp) {
    _sp = sp;
  }
  JavaThread* thread() const { return _thread; }
};

// Watermark for JFRLL, handling multiple JFRLL frame marks
class JFRLLStackWatermark : public StackWatermark {

  std::recursive_mutex _mutex;
  JFRLLFrameMark* _mark_list = nullptr;

  // helper methods

  // find smallest sp and set it
  void recompute_and_set_watermark() {
    JFRLLFrameMark *current = _mark_list;
    void* smallest = nullptr;
    bool set = false;
    while (current != nullptr) {
      if ((!set || current->sp() < smallest) && current->sp() != nullptr) {
        // ignore nulls as they disable
        smallest = current->sp();
        set = true;
      }
      current = current->next();
    }
    void* sp = set ? smallest : nullptr;
    if (sp != (void*)watermark()) {
      update_watermark((uintptr_t)sp);
    }
  }

  bool contains(JFRLLFrameMark* mark) {
    JFRLLFrameMark* current = _mark_list;
    while (current != nullptr) {
      if (current == mark) {
        return true;
      }
      current = current->next();
    }
    return false;
  }

  // deletes frame mark and update watermark if necessary
  void remove_directly(JFRLLFrameMark* mark) {
    // remove mark from list
    JFRLLFrameMark* current = _mark_list;
    while (current != nullptr) {
      if (current == mark) {
        current->set_next(mark->next());
        break;
      }
      current = current->next();
    }
    delete mark;
    recompute_and_set_watermark();
  }

  // implementation of StackWatermark methods

  virtual uint32_t epoch_id() const { return 0; }

  virtual bool process_on_iteration() { return false; }

  virtual void process(const frame& fr, RegisterMap& register_map, void* context) {
    printf("wm::process for sp %p\n", fr.sp());
  }

  virtual void trigger_before_unwind(const frame& fr) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    JFRLLFrameMark* current = _mark_list;
    while (current != nullptr) {
      if (current->applicable(fr.sp())) {
        current->call(fr);
      }
      current = current->next();
    }
  }

public:

  JFRLLStackWatermark(JavaThread* jt): StackWatermark(jt, StackWatermarkKind::jfrll, 0, false) {
  }

  void add(JFRLLFrameMark* mark) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    mark->set_next(_mark_list);
    _mark_list = mark;
    recompute_and_set_watermark();
  }

  // update of frame mark and update watermark if necessary
  void update(JFRLLFrameMark *mark, void* sp) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    assert(contains(mark), "mark does not exist");
    mark->update(sp);
    recompute_and_set_watermark();
  }

  void remove(JFRLLFrameMark* mark) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    assert(contains(mark), "mark does not exist");
    remove_directly(mark);
  }
};

// init if needed, signal safe
JFRLLStackWatermark* initWatermark(JavaThread* current = nullptr) {
  JavaThread* const jt = current == nullptr ? JavaThread::current_or_null() : current;
  assert(jt != nullptr, "Thread is null");
  if (jt->jfrll_watermark() == nullptr) {
    JFRLLStackWatermark* watermark = new JFRLLStackWatermark(jt);
    if (!jt->set_jfrll_watermark(watermark)) {
      // a watermark has been added between
      // the call to jfrll_watermark and the call to set_jfrll_watermark
      delete watermark;
    }
    StackWatermarkSet::add_watermark(jt, jt->jfrll_watermark());
  }
  return (JFRLLStackWatermark*)jt->jfrll_watermark();
}

JFRLL_FrameMark* JFRLL_RegisterFrameMark(JNIEnv* env, JFRLL_FrameMarkHandler handler, int options, void* arg) {
  JavaThread* thread = env == nullptr ? JavaThread::current_or_null() : JavaThread::thread_from_jni_environment(env);
  assert(thread != nullptr, "Thread is null");
  JFRLLFrameMark* mark = new JFRLLFrameMark(thread, handler, options, arg, nullptr);
  auto wm = initWatermark(thread);
  wm->add(mark);
  return (JFRLL_FrameMark*)mark;
}

void JFRLL_MoveFrameMark(JFRLL_FrameMark* mark, void* sp) {
  initWatermark()->update((JFRLLFrameMark*)mark, sp);
}

void* JFRLL_GetFrameMarkStackPointer(JFRLL_FrameMark* mark) {
  return ((JFRLLFrameMark*)mark)->sp();
}

void JFRLL_RemoveFrameMark(JFRLL_FrameMark* mark) {
  initWatermark(((JFRLLFrameMark*)mark)->thread())->remove((JFRLLFrameMark*)mark);
}

class JFRLLCustomSampler : public virtual JfrCustomSampler {
  JFRLL_StartSampler _start;
  JFRLL_StopSampler _stop;
  JFRLL_OnConfigChange _on_config;
  JFRLL_OnJFRCheckpoint _on_checkpoint;

  JFRLLCustomSampler(char* name, JFRLL_StartSampler start, JFRLL_StopSampler stop, JFRLL_OnConfigChange on_config, JFRLL_OnJFRCheckpoint on_checkpoint) : JfrCustomSampler(name), _start(start), _stop(stop), _on_config(on_config), _on_checkpoint(on_checkpoint) {}

  virtual void start(int64_t java_period_millis, int64_t native_period_millis, uint32_t stack_depth) {
    assert(java_period_millis >= 0, "java_period_millis must be >= 0");
    assert(native_period_millis >= 0, "native_period_millis must be >= 0");
    _java_period_millis = java_period_millis;
    _native_period_millis = native_period_millis;
    JFRLL_JFRConfig config = {java_period_millis, native_period_millis};
    _start(&config, stack_depth);
    _is_started = true;
  }

  virtual void update() {
    assert(is_started(), "JFRLLCustomSampler must be started");
    JFRLL_JFRConfig config = { _java_period_millis, _native_period_millis };
    _on_config(&config);
  }

  virtual void stop() {
    assert(is_started(), "JFRLLCustomSampler must be started");
    _stop();
    _is_started = false;
  }

  // to do: implement
  virtual void on_new_epoch() {
    _on_checkpoint();
  }
};