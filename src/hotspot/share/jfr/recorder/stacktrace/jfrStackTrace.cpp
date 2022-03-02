/*
 * Copyright (c) 2011, 2020, Oracle and/or its affiliates. All rights reserved.
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
#include "jfr/recorder/checkpoint/jfrCheckpointWriter.hpp"
#include "jfr/recorder/checkpoint/types/traceid/jfrTraceId.inline.hpp"
#include "jfr/recorder/repository/jfrChunkWriter.hpp"
#include "jfr/recorder/stacktrace/jfrStackTrace.hpp"
#include "jfr/recorder/stacktrace/stackWalker.hpp"
#include "jfr/recorder/storage/jfrBuffer.hpp"
#include "jfr/support/jfrMethodLookup.hpp"
#include "memory/allocation.inline.hpp"
#include "oops/instanceKlass.inline.hpp"
#include "runtime/vframe.inline.hpp"

static void copy_frames(JfrStackFrame** lhs_frames, u4 length, const JfrStackFrame* rhs_frames) {
  assert(lhs_frames != NULL, "invariant");
  assert(rhs_frames != NULL, "invariant");
  if (length > 0) {
    *lhs_frames = NEW_C_HEAP_ARRAY(JfrStackFrame, length, mtTracing);
    memcpy(*lhs_frames, rhs_frames, length * sizeof(JfrStackFrame));
  }
}

JfrStackFrame::JfrStackFrame(const traceid& id, int bci, int type, const InstanceKlass* ik) :
  _klass(ik), _methodid(id), _line(0), _bci(bci), _type(type) {}

JfrStackFrame::JfrStackFrame(const traceid& id, int bci, int type, int lineno, const InstanceKlass* ik) :
  _klass(ik), _methodid(id), _line(lineno), _bci(bci), _type(type) {}

JfrStackTrace::JfrStackTrace(JfrStackFrame* frames, u4 max_frames) :
  _next(NULL),
  _frames(frames),
  _id(0),
  _hash(0),
  _nr_of_frames(0),
  _max_frames(max_frames),
  _frames_ownership(false),
  _reached_root(false),
  _lineno(false),
  _written(false) {}

JfrStackTrace::JfrStackTrace(traceid id, const JfrStackTrace& trace, const JfrStackTrace* next) :
  _next(next),
  _frames(NULL),
  _id(id),
  _hash(trace._hash),
  _nr_of_frames(trace._nr_of_frames),
  _max_frames(trace._max_frames),
  _frames_ownership(true),
  _reached_root(trace._reached_root),
  _lineno(trace._lineno),
  _written(false) {
  copy_frames(&_frames, trace._nr_of_frames, trace._frames);
}

JfrStackTrace::~JfrStackTrace() {
  if (_frames_ownership) {
    FREE_C_HEAP_ARRAY(JfrStackFrame, _frames);
  }
}

template <typename Writer>
static void write_stacktrace(Writer& w, traceid id, bool reached_root, u4 nr_of_frames, const JfrStackFrame* frames) {
  w.write((u8)id);
  w.write((u1)!reached_root);
  w.write(nr_of_frames);
  for (u4 i = 0; i < nr_of_frames; ++i) {
    frames[i].write(w);
  }
}

void JfrStackTrace::write(JfrChunkWriter& sw) const {
  assert(!_written, "invariant");
  write_stacktrace(sw, _id, _reached_root, _nr_of_frames, _frames);
  _written = true;
}

void JfrStackTrace::write(JfrCheckpointWriter& cpw) const {
  write_stacktrace(cpw, _id, _reached_root, _nr_of_frames, _frames);
}

bool JfrStackFrame::equals(const JfrStackFrame& rhs) const {
  return _methodid == rhs._methodid && _bci == rhs._bci && _type == rhs._type;
}

bool JfrStackTrace::equals(const JfrStackTrace& rhs) const {
  if (_reached_root != rhs._reached_root || _nr_of_frames != rhs._nr_of_frames || _hash != rhs._hash) {
    return false;
  }
  for (u4 i = 0; i < _nr_of_frames; ++i) {
    if (!_frames[i].equals(rhs._frames[i])) {
      return false;
    }
  }
  return true;
}

template <typename Writer>
static void write_frame(Writer& w, traceid methodid, int line, int bci, u1 type) {
  w.write((u8)methodid);
  w.write((u4)line);
  w.write((u4)bci);
  w.write((u8)type);
}

void JfrStackFrame::write(JfrChunkWriter& cw) const {
  write_frame(cw, _methodid, _line, _bci, _type);
}

void JfrStackFrame::write(JfrCheckpointWriter& cpw) const {
  write_frame(cpw, _methodid, _line, _bci, _type);
}

static const size_t min_valid_free_size_bytes = 16;

static inline bool is_full(const JfrBuffer* enqueue_buffer) {
  return enqueue_buffer->free_size() < min_valid_free_size_bytes;
}

static void process_stackwalker_frame_state(const StackWalker &st, int* type, int* bci) {
  *bci = 0;
  switch (st.state()) {
  case STACKWALKER_INTERPRETED_FRAME:
    *type = JfrStackFrame::FRAME_INTERPRETER;
    break;
  case STACKWALKER_COMPILED_FRAME:
    *type = st.is_inlined() ? JfrStackFrame::FRAME_INLINE : JfrStackFrame::FRAME_JIT;
    break;
  case STACKWALKER_NATIVE_FRAME:
    *type = JfrStackFrame::FRAME_NATIVE;
    *bci = 0;
  default:
    assert(false, "this case should never be reached");
    break;
  }
}

bool JfrStackTrace::record_thread(JavaThread& thread, frame& frame) {
  // Explicitly monitor the available space of the thread-local buffer used for enqueuing klasses as part of tagging methods.
  // We do this because if space becomes sparse, we cannot rely on the implicit allocation of a new buffer as part of the
  // regular tag mechanism. If the free list is empty, a malloc could result, and the problem with that is that the thread
  // we have suspended could be the holder of the malloc lock. If there is no more available space, the attempt is aborted.
  const JfrBuffer* const enqueue_buffer = JfrTraceIdLoadBarrier::get_enqueue_buffer(Thread::current());
  assert(enqueue_buffer != nullptr, "invariant");
  StackWalker st(&thread, frame, true /* skip c frames */,
    MAX_STACK_DEPTH * 2 /* maximum number of c frames to skip */);
  u4 count = 0;
  _reached_root = true;
  _hash = 1;
  while (!st.at_end()) {
    if (count >= _max_frames) {
      _reached_root = false;
      break;
    }
    if (st.at_error() || is_full(enqueue_buffer)) {
      return false;
    }
    const Method* method = st.method();
    const traceid mid = JfrTraceId::load(method);
    int type, bci;
    process_stackwalker_frame_state(st, &type, &bci);
    const int lineno = method->line_number_from_bci(bci);
    _hash = (_hash * 31) + mid;
    _hash = (_hash * 31) + bci;
    _hash = (_hash * 31) + type;
    _frames[count] = JfrStackFrame(mid, bci, type, lineno, method->method_holder());
    count++;
    st.next();
  }

  _lineno = true;
  _nr_of_frames = count;
  return true;
}

void JfrStackFrame::resolve_lineno() const {
  assert(_klass, "no klass pointer");
  assert(_line == 0, "already have linenumber");
  const Method* const method = JfrMethodLookup::lookup(_klass, _methodid);
  assert(method != NULL, "invariant");
  assert(method->method_holder() == _klass, "invariant");
  _line = method->line_number_from_bci(_bci);
}

void JfrStackTrace::resolve_linenos() const {
  for (unsigned int i = 0; i < _nr_of_frames; i++) {
    _frames[i].resolve_lineno();
  }
  _lineno = true;
}

bool JfrStackTrace::record_safe(JavaThread* thread, int skip) {
  assert(thread == Thread::current(), "Thread stack needs to be walkable");
  StackWalker st(thread, true /* skip native frames */,
    MAX_STACK_DEPTH * 2 /* maximum number of c frames to skip */);
  u4 count = 0;
  _reached_root = true;
  st.skip_frames(skip);

  _hash = 1;
  while (!st.at_end()) {
    if (count >= _max_frames) {
      _reached_root = false;
      break;
    }
    if (st.at_error()) {
      return false;
    }
    const Method* method = st.method();
    const traceid mid = JfrTraceId::load(method);
    int type, bci;
    process_stackwalker_frame_state(st, &type, &bci);
    _hash = (_hash * 31) + mid;
    _hash = (_hash * 31) + bci;
    _hash = (_hash * 31) + type;
    _frames[count] = JfrStackFrame(mid, bci, type, method->method_holder());
    count++;
  }

  _nr_of_frames = count;
  return true;
}

