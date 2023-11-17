/*
 * Copyright (c) 2017, 2022, Oracle and/or its affiliates. All rights reserved.
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

#ifndef SHARE_RUNTIME_HANDSHAKE_HPP
#define SHARE_RUNTIME_HANDSHAKE_HPP

#include "code/relocInfo.hpp"
#include "memory/allStatic.hpp"
#include "memory/allocation.hpp"
#include "memory/iterator.hpp"
#include "runtime/atomic.hpp"
#include "runtime/flags/flagSetting.hpp"
#include "runtime/frame.hpp"
#include "runtime/mutex.hpp"
#include "runtime/orderAccess.hpp"
#include "runtime/os.hpp"
#include "runtime/stackWatermark.hpp"
#include "utilities/filterQueue.hpp"
#include "utilities/globalDefinitions.hpp"
#include "utilities/linkedlist.hpp"

#include "utilities/vmassert_uninstall.hpp"
#include <atomic>
#include <functional>
#include <mutex>
#include "utilities/vmassert_reinstall.hpp"

class HandshakeOperation;
class AsyncHandshakeOperation;
class JavaThread;
class SuspendThreadHandshake;
class ThreadSelfSuspensionHandshake;
class UnsafeAccessErrorHandshake;
class ThreadsListHandle;

// A handshake closure is a callback that is executed for a JavaThread
// while it is in a safepoint/handshake-safe state. Depending on the
// nature of the closure, the callback may be executed by the initiating
// thread, the target thread, or the VMThread. If the callback is not executed
// by the target thread it will remain in a blocked state until the callback completes.
class HandshakeClosure : public ThreadClosure, public CHeapObj<mtThread> {
  const char* const _name;
 public:
  HandshakeClosure(const char* name) : _name(name) {}
  virtual ~HandshakeClosure()                      {}
  const char* name() const                         { return _name; }
  virtual bool is_async()                          { return false; }
  virtual bool is_suspend()                        { return false; }
  virtual bool is_async_exception()                { return false; }
  virtual void do_thread(Thread* thread) = 0;
};

class AsyncHandshakeClosure : public HandshakeClosure {
 public:
   AsyncHandshakeClosure(const char* name) : HandshakeClosure(name) {}
   virtual ~AsyncHandshakeClosure() {}
   virtual bool is_async()          { return true; }
};

class Handshake : public AllStatic {
 public:
  // Execution of handshake operation
  static void execute(HandshakeClosure*       hs_cl);
  // This version of execute() relies on a ThreadListHandle somewhere in
  // the caller's context to protect target (and we sanity check for that).
  static void execute(HandshakeClosure*       hs_cl, JavaThread* target);
  // This version of execute() is used when you have a ThreadListHandle in
  // hand and are using it to protect target. If tlh == nullptr, then we
  // sanity check for a ThreadListHandle somewhere in the caller's context
  // to verify that target is protected.
  static void execute(HandshakeClosure*       hs_cl, ThreadsListHandle* tlh, JavaThread* target);
  // This version of execute() relies on a ThreadListHandle somewhere in
  // the caller's context to protect target (and we sanity check for that).
  static void execute(AsyncHandshakeClosure*  hs_cl, JavaThread* target);
};

class JvmtiRawMonitor;

class JFRLLQueue;

class JFRLLQueueElement {
  void* _pc;
  void* _fp;
  void* _sp;
  void* _argument;
  address _bcp; // might be null for some elements
public:
  JFRLLQueueElement(void* pc, void* fp, void *sp, void* argument, address bcp) : _pc(pc), _fp(fp), _sp(sp), _argument(argument), _bcp(bcp) {}
  JFRLLQueueElement(): JFRLLQueueElement(nullptr, nullptr, nullptr, nullptr, nullptr) {}
  void* pc() const { return _pc; }
  void* fp() const { return _fp; }
  void* sp() const { return _sp; }
  void* argument() const { return _argument; }
  void* bcp() const { return _bcp; }
  JFRLLQueueElement set_argument(void* argument) const { return JFRLLQueueElement(_pc, _fp, _sp, argument, _bcp); }
};

class JFRLLQueueElementHandler : public CHeapObj<mtServiceability> {
 public:
  virtual void operator()(JFRLLQueueElement*, frame top_frame, CompiledMethod* cm = nullptr) = 0;
  virtual ~JFRLLQueueElementHandler() {}
};

class JFRLLQueueOnSafepointHandler : public CHeapObj<mtServiceability> {
 public:
  virtual void operator()(JFRLLQueue*, frame top_frame, CompiledMethod* cm = nullptr) = 0;
  virtual ~JFRLLQueueOnSafepointHandler() {}
};

enum JFRLLQueuePushResult: int8_t {
  JFRLL_QUEUE_PUSH_SUCCESS = 0,
  JFRLL_QUEUE_PUSH_FULL = 1,
  JFRLL_QUEUE_PUSH_CLOSED = 2
};

class JFRLLQueue : public CHeapObj<mtServiceability> {
  int _id;
  JavaThread* _thread;
  JFRLLQueueElement* _elements;
  int _capacity;
  // pop: head increase
  volatile int _head;
  // push: tail increase
  volatile int _tail;
  volatile int _attempts;

  const int STATE_CLOSED             = 0;
  const int STATE_OPEN               = 1;
  const int STATE_CURRENTLY_PUSHING  = 2;

  volatile int _state = STATE_OPEN;

  JFRLLQueueElementHandler* _handler;
  JFRLLQueueOnSafepointHandler* _before_handler = nullptr;
  JFRLLQueueOnSafepointHandler* _after_handler = nullptr;

  std::mutex _handler_lock;

  JFRLLQueue* _next = nullptr;

  // suggested new size, -1 for no resize
  int _new_size;

  // resize if new size is set, drops all elements still enqueued,
  // so clean it
  void resize_if_needed();

  bool transition_to_push_to_jfrll_queue() {
    return Atomic::cmpxchg(&_state, STATE_OPEN, STATE_CURRENTLY_PUSHING) == STATE_OPEN;
  }

  void finished_push_to_jfrll_queue() {
    assert(_state == STATE_CURRENTLY_PUSHING, "wrong state");
    _state = STATE_OPEN;
  }
public:

  // Constructor
  // @param handler pointer to the handler, deleted when the JFRLLQueue is destroyed
  JFRLLQueue(int id, JavaThread *thread, size_t size,
             JFRLLQueueElementHandler *handler);

  int capacity() const { return _capacity; }
  int length() const { return (_tail < _head ? (_tail + _capacity) : _tail) - _head; }
  bool is_full() const { return length() >= capacity(); }
  bool is_empty() const { return length() <= 0; }
  // or null if no such element
  JFRLLQueueElement* get(int n);

  // use the methods in HandshakeState to enqueue/dequeue

  JFRLLQueuePushResult push(JFRLLQueueElement element);

  // element or null if empty
  JFRLLQueueElement *pop();

  ~JFRLLQueue() {
    transition_to_close_jfrll_queue();
    delete _handler;
    os::free(_elements);
    delete _before_handler;
    delete _after_handler;
    // no need to open the queue, as it is removed anyway
  }

  // called directly before the handle method invocations at a safe-point
  //
  // triggers the registered handle
  void before(frame top_frame, CompiledMethod* cm = nullptr);

  // called directly after the handle method invocations at a safe-point
  //
  // triggers the registered handle, resizes the queue if requested,
  // and resets the attempts
  void after(frame top_frame, CompiledMethod* cm = nullptr);

  // sets the before handler and deletes the previous handler if present
  void set_before(JFRLLQueueOnSafepointHandler *handler);

  // sets the after handler and deletes the previous handler if present
  void set_after(JFRLLQueueOnSafepointHandler *handler);

  bool equals(const JFRLLQueue* other) const {
    return other != nullptr && _id == other->_id;
  }

  bool in_current_thread();

  JFRLLQueue* next() const { return _next; }

  void set_next(JFRLLQueue* next) { _next = next; }

  bool has_next() const { return _next != nullptr; }

  // returns the number of push calls since the last finish of a safepoint
  int attempts() const { return _attempts; }

  // trigger resize to happen at next finish of a safepoint / after call
  void trigger_resize(int new_size) {
    assert(in_current_thread(), "only call from current thread");
    _new_size = new_size;
  }

  void handle(JFRLLQueueElement* element, frame top_frame, CompiledMethod* cm = nullptr) const {
    (*_handler)(element, top_frame, cm);
  }

  // e.g. during return safepoint handling
  void transition_to_close_jfrll_queue() {
    while (Atomic::cmpxchg(&_state, STATE_OPEN, STATE_CLOSED) != STATE_OPEN) {
      // spin, the same thread is currently pushing to the queues in a signal handler, or some other method
      // maybe we don't need this, but I'm unsure, so I leave it in to be on the safe side
    }
  }

  // reversal of transition_to_close_jfrll_queues
  void open_jfrll_queue() {
    assert(_state == STATE_CLOSED, "wrong state");
    _state = STATE_OPEN;
  }
};

// The HandshakeState keeps track of an ongoing handshake for this JavaThread.
// VMThread/Handshaker and JavaThread are serialized with _lock making sure the
// operation is only done by either VMThread/Handshaker on behalf of the
// JavaThread or by the target JavaThread itself.
class HandshakeState {
  friend ThreadSelfSuspensionHandshake;
  friend SuspendThreadHandshake;
  friend UnsafeAccessErrorHandshake;
  friend JavaThread;
  // This a back reference to the JavaThread,
  // the target for all operation in the queue.
  JavaThread* _handshakee;
  // The queue containing handshake operations to be performed on _handshakee.
  FilterQueue<HandshakeOperation*> _queue;
  // Provides mutual exclusion to this state and queue. Also used for
  // JavaThread suspend/resume operations.
  Monitor _lock;
  // Set to the thread executing the handshake operation.
  Thread* volatile _active_handshaker;

  bool claim_handshake();
  bool possibly_can_process_handshake();
  bool can_process_handshake();

  bool have_non_self_executable_operation();
  HandshakeOperation* get_op_for_self(bool allow_suspend, bool check_async_exception);
  HandshakeOperation* get_op();
  void remove_op(HandshakeOperation* op);

  void set_active_handshaker(Thread* thread) { Atomic::store(&_active_handshaker, thread); }

  class MatchOp {
    HandshakeOperation* _op;
   public:
    MatchOp(HandshakeOperation* op) : _op(op) {}
    bool operator()(HandshakeOperation* op) {
      return op == _op;
    }
  };
  bool has_operation() { return !_queue.is_empty(); }

  bool has_operation(bool allow_suspend, bool check_async_exception);
 public:

  HandshakeState(JavaThread* thread);
  ~HandshakeState();

  void add_operation(HandshakeOperation* op);


  // does calling process_by_self make sense?
  bool can_run() { return has_operation() || has_jfrll_entries(); }
  bool can_run(bool allow_suspend, bool check_async_exception) {
    return has_operation(allow_suspend, check_async_exception) || has_jfrll_entries();
  }
  bool has_async_exception_operation();
  void clean_async_exception_operation();

  bool operation_pending(HandshakeOperation* op);

  // If the method returns true we need to check for a possible safepoint.
  // This is due to a suspension handshake which put the JavaThread in blocked
  // state so a safepoint may be in-progress.
  bool process_by_self(bool allow_suspend, bool check_async_exception, frame* top_frame = nullptr, CompiledMethod* cm = nullptr);

  enum ProcessResult {
    _no_operation = 0,
    _not_safe,
    _claim_failed,
    _processed,
    _succeeded,
    _number_states
  };
  ProcessResult try_process(HandshakeOperation* match_op);

  Thread* active_handshaker() const { return Atomic::load(&_active_handshaker); }

  JFRLLQueue* register_jfrll_queue(JavaThread* thread, size_t size, JFRLLQueueElementHandler* handler);

  bool remove_jfrll_queue(JFRLLQueue* queue);

  JFRLLQueuePushResult jfrll_enqueue(JFRLLQueue* queue, JFRLLQueueElement element);

  int jfrll_queue_size() const { return _jfrll_queue_size; }


  // Support for asynchronous exceptions
 private:
  bool _async_exceptions_blocked;

  bool async_exceptions_blocked() { return _async_exceptions_blocked; }
  void set_async_exceptions_blocked(bool b) { _async_exceptions_blocked = b; }
  void handle_unsafe_access_error();

  // Suspend/resume support
 private:
  // This flag is true when the thread owning this
  // HandshakeState (the _handshakee) is suspended.
  volatile bool _suspended;
  // This flag is true while there is async handshake (trap)
  // on queue. Since we do only need one, we can reuse it if
  // thread gets suspended again (after a resume)
  // and we have not yet processed it.
  bool _async_suspend_handshake;

  int _current_jfrll_queue_id = 0;
  // linked list
  JFRLLQueue* _jfrll_queue_start = nullptr;
  std::mutex _jfrll_queue_mutex;
  std::atomic<int> _jfrll_queue_size = {0};
  StackWatermark* volatile _jfrll_watermark = 0;

public:


  // e.g. during return safepoint handling
  void transition_to_close_jfrll_queues() {
    for (JFRLLQueue* queue = _jfrll_queue_start; queue != nullptr; queue = queue->next()) {
      queue->transition_to_close_jfrll_queue();
    }
  }

  // reversal of transition_to_close_jfrll_queues
  void open_jfrll_queues() {
    for (JFRLLQueue* queue = _jfrll_queue_start; queue != nullptr; queue = queue->next()) {
      queue->open_jfrll_queue();
    }
  }

  bool has_jfrll_queues() const { return _jfrll_queue_start != nullptr; }
  bool has_jfrll_entries() const { return _jfrll_queue_start != nullptr && _jfrll_queue_size.load() > 0; }

private:
  void process_jfrll_queue(frame top_frame, CompiledMethod* cm = nullptr);

  // Called from the suspend handshake.
  bool suspend_with_handshake();
  // Called from the async handshake (the trap)
  // to stop a thread from continuing execution when suspended.
  void do_self_suspend();

  bool is_suspended()                       { return Atomic::load(&_suspended); }
  void set_suspended(bool to)               { return Atomic::store(&_suspended, to); }
  bool has_async_suspend_handshake()        { return _async_suspend_handshake; }
  void set_async_suspend_handshake(bool to) { _async_suspend_handshake = to; }

  bool suspend();
  bool resume();
};

#endif // SHARE_RUNTIME_HANDSHAKE_HPP
