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

#include "memory/allStatic.hpp"
#include "memory/allocation.hpp"
#include "memory/iterator.hpp"
#include "runtime/atomic.hpp"
#include "runtime/flags/flagSetting.hpp"
#include "runtime/javaFrameAnchor.hpp"
#include "runtime/mutex.hpp"
#include "runtime/orderAccess.hpp"
#include "runtime/os.hpp"
#include "runtime/stackWatermark.hpp"
#include "utilities/filterQueue.hpp"
#include "utilities/globalDefinitions.hpp"
#include "utilities/linkedlist.hpp"

#include <atomic>
#include <functional>
#include <mutex>
#ifdef assert
#undef assert
#endif
#define assert(p, ...) vmassert(p, __VA_ARGS__)

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

class ASGSTQueue;

class ASGSTQueueElement {
  void* _pc;
  void* _argument;
public:
  ASGSTQueueElement(void* pc, void* argument) : _pc(pc), _argument(argument) {}
  ASGSTQueueElement(): ASGSTQueueElement(nullptr, nullptr) {}
  void* pc() const { return _pc; }
  void* argument() const { return _argument; }
};

class ASGSTQueueElementHandler : public CHeapObj<mtServiceability> {
 public:
  //
  virtual void operator()(ASGSTQueueElement*, JavaFrameAnchor*) = 0;
};

class ASGSTQueueOnSafepointHandler : public CHeapObj<mtServiceability> {
 public:
  virtual void operator()(ASGSTQueue*, JavaFrameAnchor*) = 0;
};

class ASGSTQueue : public CHeapObj<mtServiceability> {
  int id;
  JavaThread* thread;
  ASGSTQueueElement* elements;
  int size;
  // pop: head increase
  int head;
  // push: tail increase
  int tail;
  ASGSTQueueElementHandler* handler;
  ASGSTQueueOnSafepointHandler* before_handler = nullptr;
  ASGSTQueueOnSafepointHandler* after_handler = nullptr;

  std::mutex _handler_lock;

  ASGSTQueue* _next = nullptr;

public:

  // Constructor
  // @param handler pointer to the handler, deleted when the ASGSTQueue is destroyed
  ASGSTQueue(int id, JavaThread *thread, size_t size,
             ASGSTQueueElementHandler *handler);

  int max_size() const { return size; }
  int length() const { return (tail < head ? (tail + size) : tail) - head; }
  bool is_full() const { return length() >= max_size(); }
  bool is_empty() const { return length() <= 0; }

  // use the methods in HandshakeState to enqueue/dequeue

  bool push(ASGSTQueueElement element);

  // element or null if empty
  ASGSTQueueElement *pop();

  ~ASGSTQueue() {
    delete handler;
    os::free(elements);
    delete before_handler;
    delete after_handler;
  }

  void handle(ASGSTQueueElement* element, JavaFrameAnchor* frame_anchor) {
    (*handler)(element, frame_anchor);
  }

  // called directly before the handle method invocations at a safe-point
  void before(JavaFrameAnchor *frame_anchor);

  // called directly after the handle method invocations at a safe-point
  void after(JavaFrameAnchor *frame_anchor);

  // sets the before handler and deletes the previous handler if present
  void set_before(ASGSTQueueOnSafepointHandler *handler);

  // sets the after handler and deletes the previous handler if present
  void set_after(ASGSTQueueOnSafepointHandler *handler);

  bool equals(const ASGSTQueue* other) const {
    return other != nullptr && id == other->id;
  }

  bool in_current_thread();

  ASGSTQueue* next() const { return _next; }

  void set_next(ASGSTQueue* next) { _next = next; }

  bool has_next() const { return _next != nullptr; }
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

 public:

  HandshakeState(JavaThread* thread);
  ~HandshakeState();

  void add_operation(HandshakeOperation* op);

  bool has_operation() { return !_queue.is_empty(); }
  // does calling process_by_self make sense?
  bool can_run() { return has_operation() || has_asgst_queues(); }
  bool has_operation(bool allow_suspend, bool check_async_exception);
  bool has_async_exception_operation();
  void clean_async_exception_operation();

  bool operation_pending(HandshakeOperation* op);

  // If the method returns true we need to check for a possible safepoint.
  // This is due to a suspension handshake which put the JavaThread in blocked
  // state so a safepoint may be in-progress.
  bool process_by_self(bool allow_suspend, bool check_async_exception);

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

  ASGSTQueue* register_asgst_queue(JavaThread* thread, size_t size, ASGSTQueueElementHandler* handler);

  bool remove_asgst_queue(ASGSTQueue* queue);

  bool asgst_enqueue(ASGSTQueue* queue, ASGSTQueueElement element);

  int asgst_queue_size() const { return _asgst_queue_size; }


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

  int _current_asgst_queue_id = 0;
  // linked list
  ASGSTQueue* _asgst_queue_start = nullptr;
  std::mutex _asgst_queue_mutex;
  std::atomic<int> _asgst_queue_size = {0};
  StackWatermark* volatile _asgst_watermark = 0;

  void process_asgst_queue(JavaFrameAnchor* frame_anchor);

  bool has_asgst_queues() const { return _asgst_queue_start != nullptr; }

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
