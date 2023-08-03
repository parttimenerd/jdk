
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

#ifndef SHARE_JFR_RECORDER_STACKTRACE_STACKWALKER_HPP
#define SHARE_JFR_RECORDER_STACKTRACE_STACKWALKER_HPP

#include "code/compiledMethod.hpp"
#include "oops/method.hpp"
#include "runtime/frame.hpp"
#include "runtime/registerMap.hpp"
#include "runtime/thread.hpp"
#include "runtime/vframe.inline.hpp"

// a helper stream
class compiledFrameStream : public vframeStreamCommon {
  bool cf_next_into_inlined;
  bool _invalid;
 public:
  compiledFrameStream(): vframeStreamCommon(RegisterMap(nullptr,
    RegisterMap::UpdateMap::skip,
    RegisterMap::ProcessFrames::skip, RegisterMap::WalkContinuation::skip)),
    cf_next_into_inlined(false), _invalid(true) {};
  // constructor that starts with sender of frame fr (top_frame)
  compiledFrameStream(JavaThread *jt, frame fr, bool stop_at_java_call_stub);
  void cf_next();
  bool cf_next_did_go_into_inlined() const { return cf_next_into_inlined; }
  bool inlined() const { return _sender_decode_offset != 0; }
  bool invalid() const { return _invalid; }
};

// errors, subset of forte errors
enum StackWalkerError {
  STACKWALKER_NO_JAVA_FRAME        =  0,  // too many c frames to skip and no java frame found
  STACKWALKER_INDECIPHERABLE_FRAME = -1,
  STACKWALKER_INDECIPHERABLE_NATIVE_FRAME = -2,
  STACKWALKER_INDECIPHERABLE_INTERPRETED_FRAME = -3,
  STACKWALKER_GC_ACTIVE            = -3,
  STACKWALKER_NOT_WALKABLE         = -6
};

enum StackWalkerReturn {
  STACKWALKER_END = 1,
  STACKWALKER_INTERPRETED_FRAME = 2,
  STACKWALKER_COMPILED_FRAME = 3,
  STACKWALKER_NATIVE_FRAME = 4,
  STACKWALKER_C_FRAME = 5, // might be runtime, stub or real C frame
  // at first bytecode backed Java frame, related to end_at_first_java_frame
  STACKWALKER_FIRST_JAVA_FRAME = 6,
  STACKWALKER_START = 7
};

// miscelaneous arguments for StackWalker,
// mainly used for walking the stack at safepoints
class StackWalkerMiscArguments {
  // method of the top Java frame if the frames method is not available
  // currently only supported for interpreted Java frames
  Method* _top_method = nullptr;
  // bcp of the first Java frame if this frame is interpreted and valid
  address _top_interp_bcp = nullptr;
public:
  StackWalkerMiscArguments() {};
  StackWalkerMiscArguments(Method* top_method, address top_interp_bcp):
    _top_method(top_method), _top_interp_bcp(top_interp_bcp) {};
  Method* top_method() const { return _top_method; }
  address top_interp_bcp() const { return _top_interp_bcp; }
};

// walk the stack of a thread from any given frame
// includes all c frames and lot's of checks
// borrowed from forte.hpp
class  StackWalker {

  // Java thread to walk
  // can be null for non java threads (only c frames then)
  JavaThread* _thread;

  // skip C frames between the Java frames
  bool _skip_c_frames;

  // end at the first/top most java frame, but don't process it, just obtain pc, fp and sp
  // end on this frame
  // java frame = bytecode backed frame
  bool _end_at_first_java_frame;

  StackWalkerMiscArguments _misc;

  // maximum number of C frames to skip, use this if there a problems with too large C stacks
  // in JNI libraries. ASGST limits it too.
  int _max_c_frames_skip;

  // allow to use _thread->last_java_frame()
  bool _allow_thread_last_frame_use;

  // current frame (surrounding frame if inlined)
  frame _frame;

  // is os::get_sender_for_C_frame currently supported?
  // invariant: true if _thread is null
  bool supports_os_get_frame;

  // StackWalkerError + StackWalkerReturn
  int _state;

  bool _inlined;

  Method *_method;

  int _bci;

  int _compilation_level;

  RegisterMap _map;

  // frame stream to walk the inner frames of compiled frames
  compiledFrameStream _st;

  // just walking the stack till the thread initiating frame
  bool in_c_on_top;

  // had first java frame (bytecode backed or native)
  bool had_first_java_frame;

  // is at first byte code backed Java frame
  bool is_at_first_java_frame = false;

  frame next_c_frame(frame fr);

  int get_bci_from_interp_frame(Method* method);

  void init();

  void process(bool potentially_first_java_frame = false);

  void advance();

  // reset _method, _bci and inlined
  void reset();

  // set the state and reset everything besides interpreted and compiled frame
  void set_state(int state);

  // check that current frame is processable
  bool checkFrame();

  void advance_normal();

  void advance_fully_c();

  void process_normal(bool potentially_first_java_frame = false);

  // additional check used in process_normal
  bool is_frame_indecipherable();

  void process_in_compiled();

public:

  StackWalker(JavaThread* thread, frame top_frame, bool skip_c_frames = true,
    bool end_at_first_java_frame = false, bool allow_thread_last_frame_use = true,
    StackWalkerMiscArguments misc = {}, int max_c_frames_skip = -1);

  // requires a non null thread
  StackWalker(JavaThread* thread, bool skip_c_frames = true, int max_c_frames_skip = -1);

  // returns an error code < 0 on error and StackWalkerReturn code otherwise.
  // 0 == ended,
  int next();

  // skip all c frames, return true if Java frame found
  bool skip_c_frames();

  // call advance at most skip times in a row
  void skip_frames(int skip);

  // StackWalkerError + StackWalkerReturn
  int state() const { return _state; }

  bool at_end() const { return _state == STACKWALKER_END; }

  bool at_error() const { return _state <= 0; }

  bool at_end_or_error() const { return at_end() || at_error(); }

  int error() const { return at_error() ? _state : 1;}

  // not at and and not at error
  bool has_frame() const { return !at_end_or_error() || is_at_first_java_frame; }

  // only if end_at_first_java_frame is true, stops afterwards
  // only base_frame is valid
  bool is_at_first_bc_java_frame() const { return _state == STACKWALKER_FIRST_JAVA_FRAME; }

  bool is_interpreted_frame() const { return _state == STACKWALKER_INTERPRETED_FRAME; }

  bool is_compiled_frame() const { return _state == STACKWALKER_COMPILED_FRAME; }

  bool is_native_frame() const { return _state == STACKWALKER_NATIVE_FRAME; }

  bool is_stub_frame() const { return _state == STACKWALKER_C_FRAME && _frame.is_stub_frame(); }

  bool is_c_frame() const { return _state == STACKWALKER_C_FRAME; }

  bool is_java_frame() const { return is_interpreted_frame() || is_compiled_frame() || is_native_frame(); }

  bool is_bytecode_based_frame() const { return is_interpreted_frame() || is_compiled_frame(); }

  // inlined, returns true only for inlined compiled frames, otherwise false
  bool is_inlined() const { return _inlined; }

  // current frame (surrounding frame if inlined) or NULL if at error or at end
  const frame* base_frame() const { return has_frame() ? &_frame : nullptr; }

  // current method or NULL if at error or at end
  Method* method() const { return _method; }

  // bci or -1 if not at a Java frame
  int bci() const { return _bci; }

  // -1 if not at a Java frame
  int compilation_level() const;

  // is the current frame the first/youngest Java frame
  // if at_end and at first Java frame, only base_frame is valid
  bool at_first_java_frame() const { return is_at_first_java_frame; }

  // return frame count
  int walk_till_end_or_error();

  // true: found Java frame
  bool walk_till_first_java_frame();

};

#endif // SHARE_JFR_RECORDER_STACKTRACE_STACKWALKER_HPP
