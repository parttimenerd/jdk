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
#include "oops/method.hpp"
#include "precompiled.hpp"

#include <algorithm>
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
#include "runtime/javaFrameAnchor.hpp"
#include "runtime/javaThread.hpp"
#include "runtime/jniHandles.hpp"
#include "runtime/os.hpp"
#include "runtime/safefetch.hpp"
#include "runtime/safepointMechanism.hpp"
#include "runtime/stackWatermark.hpp"
#include "runtime/stackWatermarkKind.hpp"
#include "utilities/globalDefinitions.hpp"
#include "code/compiledMethod.hpp"


int ASGST_Capabilities() {
  return ASGST_REGISTER_QUEUE | ASGST_MARK_FRAME;
}

struct _ASGST_Iterator {
  StackWalker walker;
  JavaThread *thread;
  JavaFrameAnchor anchor;
  int options = options;
  bool invalidSpAndFp = false;
  bool switchToThreadBased = false;

  void reset() {
    thread = nullptr;
    anchor = JavaFrameAnchor();
    options = 0;
    invalidSpAndFp = false;
    switchToThreadBased = false;
  }
};

static int initStackWalker(_ASGST_Iterator *iterator, int options, frame frame, bool allow_thread_last_frame_use = true, bool set_async = true) {
  if (set_async && iterator->thread != nullptr) {
      iterator->thread->set_in_async_stack_walking(true);
  }
  iterator->walker = StackWalker(iterator->thread, frame,
    (options & ASGST_INCLUDE_NON_JAVA_FRAMES) == 0,
    (options & ASGST_END_ON_FIRST_JAVA_FRAME) != 0,
    allow_thread_last_frame_use);
  iterator->options = options;
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
  int ret = initStackWalker(iter, options, ret_frame);
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

// @return error or kind
int ASGST_CreateIter(_ASGST_Iterator* iterator, void* ucontext, int32_t options) {

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
          return ASGST_NO_TOP_JAVA_FRAME;
        }
      }
      return initStackWalker(iterator, options, ret_frame);
    }
    break;
  default:
    // Unknown thread state
    return ASGST_NO_FRAME;
  }
  return 0;
}

int ASGST_CreateIterFromFrame(_ASGST_Iterator* iterator, void* sp, void* fp, void* pc, int32_t options) {

  bool include_non_java_frames = (options & ASGST_INCLUDE_NON_JAVA_FRAMES) != 0;


  int kindOrError = ASGST_Check(&iterator->thread);
  // handle error case
  if (kindOrError <= 0) {
    return kindOrError;
  }

  frame f{sp, fp, pc};

  // handle non-java case
  if (kindOrError > ASGST_JAVA_TRACE) {
    int ret = initStackWalker(iterator, options, f);
    return ret < 1 ? ret : ASGST_NON_JAVA_TRACE;
  }
  int ret = initStackWalker(iterator, options, f);
  return ret < 1 ? ret : ASGST_JAVA_TRACE;
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
  frame->method = (ASGST_Method)m;
  auto f = iter->walker.base_frame();
  frame->pc = f->pc();
  frame->sp = iter->invalidSpAndFp ? nullptr : f->sp();
  frame->fp = iter->invalidSpAndFp ? nullptr : f->fp();
  if (iter->walker.is_bytecode_based_frame()) {
    frame->comp_level = iter->walker.compilation_level(),
    frame->bci = iter->walker.bci();
  }
  if (iter->walker.is_bytecode_based_frame()) {
    frame->type = iter->walker.is_inlined() ? ASGST_FRAME_JAVA_INLINED : ASGST_FRAME_JAVA;
  } else if (iter->walker.is_native_frame()) {
    frame->type = ASGST_FRAME_JAVA_NATIVE;
  } else {
    frame->type = ASGST_FRAME_NON_JAVA;
  }
  if (iter->switchToThreadBased && !iter->walker.is_inlined() && iter->anchor.last_Java_pc() != nullptr) {
    iter->switchToThreadBased = false;
    iter->invalidSpAndFp = false;
    int8_t walkerBytes[sizeof(StackWalker)]; // backup the old walker
    StackWalker* walker = (StackWalker*) walkerBytes;
    *walker = iter->walker;
    assert(iter->anchor.last_Java_pc() != nullptr, "non nulllll");
    initStackWalker(iter, iter->options,
      {iter->anchor.last_Java_sp(), iter->anchor.last_Java_fp(), iter->anchor.last_Java_pc()}, false, false);
    if ((Method*)frame->method == iter->walker.method() && frame->sp == nullptr) {
      // last frame from old walker is similar to current frame from new walker
      // therefor skip the old walker frame (contains less information)
      // typical for poll-at-return handshakes
      return ASGST_NextFrame(iter, frame);
    }
    if (iter->walker.at_end_or_error() && !walker->at_end_or_error()) {
      // use the old walker
      iter->walker = *walker;
      if ((Method*)frame->method == iter->walker.method() && frame->sp == nullptr) {
        // last frame from old walker is similar to current frame from new walker
        // therefor skip the old walker frame (contains less information)
        // typical for poll-at-return handshakes
        return ASGST_NextFrame(iter, frame);
      }
      return 1;
    }
  } else  iter->walker.next();
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
  iterator->reset();
  IterRAII raii(iterator); // destroy iterator on exit
  int ret = ASGST_CreateIter(iterator, ucontext, options);
  if (ret <= 0) {
    return ret;
  }
  fun(iterator, argument);
  return ret;
}

int ASGST_RunWithIteratorFromFrame(void* sp, void* fp, void* pc, int options, void (*fun)(ASGST_Iterator*, void*), void* argument) {
  int8_t iter[sizeof(ASGST_Iterator)]; // no need for default constructor
  ASGST_Iterator* iterator = (ASGST_Iterator*) iter;
  iterator->reset();
  IterRAII raii(iterator); // destroy iterator on exit
  int ret = ASGST_CreateIterFromFrame(iterator, sp, fp, pc, options);
  if (ret <= 0) {
    return ret;
  }
  fun(iterator, argument);
  return ret;
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

const char* typeToStr(int type) {
  switch (type) {
    case ASGST_FRAME_JAVA:
      return "java";
    case ASGST_FRAME_JAVA_INLINED:
      return "java_inlined";
    case ASGST_FRAME_JAVA_NATIVE:
      return "java_native";
    case ASGST_FRAME_NON_JAVA:
      return "non_java";
    case 0:
      return "error";
    default:
      return "unknown";
  }
}

class ASGSTQueueElementHandlerImpl : public ASGSTQueueElementHandler {
  int options;
  ASGST_Handler fun;
  void* queue_arg;
public:
  ASGSTQueueElementHandlerImpl(int options, ASGST_Handler fun, void* queue_arg) : options(options), fun(fun), queue_arg(queue_arg) {}
  void operator()(ASGSTQueueElement* element, JavaFrameAnchor* frame_anchor) override {
    int8_t iter[sizeof(ASGST_Iterator)]; // no need for default constructor
    ASGST_Iterator* iterator = (ASGST_Iterator*) iter;
    iterator->reset();
    iterator->thread = JavaThread::current();
    IterRAII raii(iterator); // destroy iterator on exit
    assert(element != nullptr, "element is null");
    if (element->pc() == nullptr && frame_anchor->last_Java_pc() == nullptr) {
      return;
    }
    frame fExtended = frame(frame_anchor->last_Java_sp(), frame_anchor->last_Java_fp(),
      element->pc() != nullptr ? (address)element->pc() : frame_anchor->last_Java_pc());
    initStackWalker(iterator, options, fExtended, false);
    if (element->pc() != nullptr && fExtended.pc() != frame_anchor->last_Java_pc() && frame_anchor->last_Java_pc() != nullptr) {
      iterator->switchToThreadBased = true;
      iterator->invalidSpAndFp = true;
      iterator->anchor = frame_anchor;
    }
    fun(iterator, queue_arg, element->argument());
    return;
  }
};

ASGST_Queue* ASGST_RegisterQueue(JNIEnv* env, int size, int options, ASGST_Handler fun, void* argument) {
  JavaThread* thread = env == nullptr ? JavaThread::current_or_null() : JavaThread::thread_from_jni_environment(env);
  if (thread == nullptr || !os::is_readable_pointer(thread->handshake_state()) || thread->is_terminated()) {
    return nullptr;
  }
  return (ASGST_Queue*)thread->handshake_state()->register_asgst_queue(thread, size, new ASGSTQueueElementHandlerImpl(options, fun, argument));
}

bool ASGST_DeregisterQueue(JNIEnv* env, ASGST_Queue* queue) {
  JavaThread* thread = env == nullptr ? JavaThread::current_or_null() : JavaThread::thread_from_jni_environment(env);
  if (thread == nullptr || !os::is_readable_pointer(thread->handshake_state()) || thread->is_terminated()) {
    return false;
  }
  return (ASGST_Queue*)thread->handshake_state()->remove_asgst_queue((ASGSTQueue*)queue);
}


class ASGSTQueueOnSafepointHandlerImpl : public ASGSTQueueOnSafepointHandler {
  int options;
  ASGST_OnQueueSafepointHandler fun;
  void* on_queue_arg;
public:
  ASGSTQueueOnSafepointHandlerImpl(int options, ASGST_OnQueueSafepointHandler fun, void* on_queue_arg) : options(options), fun(fun), on_queue_arg(on_queue_arg) {}
  void operator()(ASGSTQueue* queue, JavaFrameAnchor* frame_anchor) override {
    int8_t iter[sizeof(ASGST_Iterator)]; // no need for default constructor
    ASGST_Iterator* iterator = (ASGST_Iterator*) iter;
    iterator->reset();
    iterator->thread = JavaThread::current();
    IterRAII raii(iterator); // destroy iterator on exit
    if (frame_anchor->last_Java_pc() == nullptr) {
      return;
    }
    frame f = frame(frame_anchor->last_Java_sp(), frame_anchor->last_Java_fp(), frame_anchor->last_Java_pc());
    initStackWalker(iterator, options, f, false);
    fun((ASGST_Queue*)queue, iterator, on_queue_arg);
  }
};

void ASGST_SetOnQueueProcessingStart(ASGST_Queue* queue, int options, ASGST_OnQueueSafepointHandler before, void* arg) {
  ASGSTQueue* q = (ASGSTQueue*)queue;
  if (before == nullptr) {
    q->set_before(nullptr);
  } else {
    q->set_before(new ASGSTQueueOnSafepointHandlerImpl(options, before, arg));
  }
}

void ASGST_SetOnQueueProcessingEnd(ASGST_Queue* queue, int options, ASGST_OnQueueSafepointHandler after, void* arg) {
  ASGSTQueue* q = (ASGSTQueue*)queue;
  if (after == nullptr) {
    q->set_after(nullptr);
  } else {
    q->set_after(new ASGSTQueueOnSafepointHandlerImpl(options, after, arg));
  }
}

struct EnqueueFindFirstJavaFrameStruct {
  int kindOrError; // 1, no error, 2 no Java frame, < 0 error
  void* fp;
  void* pc;
  JavaThread* thread;
};

void enqueueFindFirstJavaFrame(ASGST_Iterator* iterator, void* arg) {
  auto argument = (EnqueueFindFirstJavaFrameStruct*)arg;
  if (!iterator->walker.walk_till_first_java_frame()) {
    argument->kindOrError = iterator->walker.at_error() ? iterator->walker.error() : ASGST_NON_JAVA_TRACE;
    return;
  }
  argument->kindOrError = 1;
  argument->fp = iterator->walker.base_frame()->fp();
  argument->pc = iterator->walker.base_frame()->pc();
  argument->thread = iterator->thread;
}

int ASGST_Enqueue(ASGST_Queue* queue, void* ucontext, void* argument) {
  if (queue == nullptr) {
    return ASGST_ENQUEUE_NO_QUEUE;
  }
  ASGSTQueue *q = (ASGSTQueue*)queue;
  if (q->is_full()) {
    return ASGST_ENQUEUE_FULL_QUEUE;
  }
  EnqueueFindFirstJavaFrameStruct runArgument;
  int kind = ASGST_RunWithIterator(ucontext, ASGST_END_ON_FIRST_JAVA_FRAME, &enqueueFindFirstJavaFrame, &runArgument);
  if (kind != ASGST_JAVA_TRACE || runArgument.kindOrError != ASGST_JAVA_TRACE) {
    return kind;
  }
  bool worked = runArgument.thread->handshake_state()->asgst_enqueue(q, {runArgument.pc, runArgument.pc});
  return worked ? (int)ASGST_JAVA_TRACE : (int)ASGST_ENQUEUE_FULL_QUEUE;
}

int ASGST_QueueSize(ASGST_Queue* queue) {
  return ((ASGSTQueue*)queue)->length();
}

jmethodID ASGST_MethodToJMethodID(ASGST_Method method) {
  return (jmethodID)((Method*)method)->find_jmethod_id_or_null();
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

void ASGST_GetMethodInfo(ASGST_Method method, ASGST_MethodInfo* info) {
  Method *m = (Method*)method;
  InstanceKlass *klass = m->method_holder();
  info->klass = (ASGST_Class)klass;
  writeField(m->name(), info->method_name, &info->method_name_length);
  writeField(m->signature(), info->signature, &info->signature_length);
  writeField(m->generic_signature(), info->generic_signature, &info->generic_signature_length);
  info->modifiers = m->access_flags().get_flags();
}

void ASGST_GetClassInfo(ASGST_Class klass, ASGST_ClassInfo *info) {
  if (klass == nullptr) {
    *info->class_name = '\0';
    info->class_name_length = 0;
    *info->generic_class_name = '\0';
    info->generic_class_name_length = 0;
    info->modifiers = 0;
    return;
  }
  Klass *k = (Klass*)klass;
  writeField(k->name(), info->class_name, &info->class_name_length);
  if (k->is_instance_klass()) {
    InstanceKlass *ik = (InstanceKlass*)k;
    writeField(ik->generic_signature(), info->generic_class_name, &info->generic_class_name_length);
  }
  info->modifiers = k->access_flags().get_flags();
}

ASGST_Class ASGST_GetClass(ASGST_Method method) {
  Method *m = (Method*)method;
  InstanceKlass *klass = m->method_holder();
  return (ASGST_Class)klass;
}

jclass ASGST_ClassToJClass(ASGST_Class klass) {
  JavaThread* thread = JavaThread::current_or_null();
  return klass != nullptr && thread != nullptr ?
    (jclass)JNIHandles::make_local(thread, ((Klass*)klass)->java_mirror()) : nullptr;
}

class KlassDeallocationHandlerImpl : public KlassDeallocationHandler {
  ASGST_ClassUnloadHandler handler;
public:
  KlassDeallocationHandlerImpl(ASGST_ClassUnloadHandler handler): handler(handler) {}

  void call(InstanceKlass* klass) {
    handler((ASGST_Class)klass, (ASGST_Method*)klass->methods()->data(), klass->methods()->size());
  }
};

void ASGST_RegisterClassUnloadHandler(ASGST_ClassUnloadHandler handler) {
  InstanceKlass::add_deallocation_handler(new KlassDeallocationHandlerImpl(handler));
}

class ASGSTFrameMark : public CHeapObj<mtServiceability> {
  JavaThread* _thread = nullptr;
  ASGST_FrameMarkHandler _handler;
  int _options;
  void* _argument;
  std::atomic<void*> _sp;
  ASGSTFrameMark* _next = nullptr;
public:
  ASGSTFrameMark(JavaThread* thread, ASGST_FrameMarkHandler handler, int options, void* argument, void* sp):
    _thread(thread),_handler(handler), _options(options), _argument(argument), _sp(sp) {}

  void call(ASGST_Iterator* iterator) {
    _handler((ASGST_FrameMark*)this, iterator, _argument);
  }

  void call(frame fr) {
    int8_t iter[sizeof(ASGST_Iterator)]; // no need for default constructor
    ASGST_Iterator* iterator = (ASGST_Iterator*) iter;
    iterator->reset();
    IterRAII raii(iterator); // destroy iterator on exit
    int ret = ASGST_CreateIterFromFrame(iterator, fr.sp(), fr.fp(), fr.pc(), _options);
    assert(ret == 1, "Expect Java trace");
    call(iterator);
  }

  ASGSTFrameMark* next() const { return _next; }

  void set_next(ASGSTFrameMark* next) { _next = next; }

  bool applicable(void* sp) const { return this->_sp != nullptr && this->_sp <= sp; }

  void* sp() const { return _sp; }

  void update(void* sp) {
    _sp = sp;
  }
  JavaThread* thread() const { return _thread; }
};

// Watermark for ASGST, handling multiple ASGST frame marks
class ASGSTStackWatermark : public StackWatermark {

  std::recursive_mutex _mutex;
  ASGSTFrameMark* _mark_list = nullptr;

  // helper methods

  // find smallest sp and set it
  void recompute_and_set_watermark() {
    ASGSTFrameMark *current = _mark_list;
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

  bool contains(ASGSTFrameMark* mark) {
    ASGSTFrameMark* current = _mark_list;
    while (current != nullptr) {
      if (current == mark) {
        return true;
      }
      current = current->next();
    }
    return false;
  }

  // deletes frame mark and update watermark if necessary
  void remove_directly(ASGSTFrameMark* mark) {
    // remove mark from list
    ASGSTFrameMark* current = _mark_list;
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
    ASGSTFrameMark* current = _mark_list;
    while (current != nullptr) {
      if (current->applicable(fr.sp())) {
        current->call(fr);
      }
      current = current->next();
    }
  }

public:

  ASGSTStackWatermark(JavaThread* jt): StackWatermark(jt, StackWatermarkKind::asgst, 0, false) {
  }

  void add(ASGSTFrameMark* mark) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    mark->set_next(_mark_list);
    _mark_list = mark;
    recompute_and_set_watermark();
  }

  // update of frame mark and update watermark if necessary
  void update(ASGSTFrameMark *mark, void* sp) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    assert(contains(mark), "mark does not exist");
    mark->update(sp);
    recompute_and_set_watermark();
  }

  void remove(ASGSTFrameMark* mark) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    assert(contains(mark), "mark does not exist");
    remove_directly(mark);
  }
};

// init if needed, signal safe
ASGSTStackWatermark* initWatermark(JavaThread* current = nullptr) {
  JavaThread* const jt = current == nullptr ? JavaThread::current_or_null() : current;
  assert(jt != nullptr, "Thread is null");
  if (jt->asgst_watermark() == nullptr) {
    ASGSTStackWatermark* watermark = new ASGSTStackWatermark(jt);
    if (!jt->set_asgst_watermark(watermark)) {
      // a watermark has been added between
      // the call to asgst_watermark and the call to set_asgst_watermark
      delete watermark;
    }
    StackWatermarkSet::add_watermark(jt, jt->asgst_watermark());
  }
  return (ASGSTStackWatermark*)jt->asgst_watermark();
}

ASGST_FrameMark* ASGST_RegisterFrameMark(JNIEnv* env, ASGST_FrameMarkHandler handler, int options, void* arg) {
  JavaThread* thread = env == nullptr ? JavaThread::current_or_null() : JavaThread::thread_from_jni_environment(env);
  assert(thread != nullptr, "Thread is null");
  ASGSTFrameMark* mark = new ASGSTFrameMark(thread, handler, options, arg, nullptr);
  auto wm = initWatermark(thread);
  wm->add(mark);
  return (ASGST_FrameMark*)mark;
}

void ASGST_MoveFrameMark(ASGST_FrameMark* mark, void* sp) {
  initWatermark()->update((ASGSTFrameMark*)mark, sp);
}

void* ASGST_GetFrameMarkStackPointer(ASGST_FrameMark* mark) {
  return ((ASGSTFrameMark*)mark)->sp();
}

void ASGST_RemoveFrameMark(ASGST_FrameMark* mark) {
  initWatermark(((ASGSTFrameMark*)mark)->thread())->remove((ASGSTFrameMark*)mark);
}