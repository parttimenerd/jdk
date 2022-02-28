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

#include "runtime/frame.hpp"
#include "runtime/thread.hpp"

// errors, subset of forte errors
enum StackWalkerError {
  STACKWALKER_INDECIPHERABLE_FRAME = -1,
  STACKWALKER_GC_ACTIVE            = -2,
  STACKWALKER_NOT_WALKABLE         = -6
};

// walk the stack of a current thread
class StackWalker {

  // Java thread to walk
  JavaThread* thread;

  // is os::get_sender_for_C_frame currently supported?
  const bool supports_os_get_frame;

  frame next_c_frame(frame fr);

public:

  StackWalker(JavaThread* thread);

  // walk the stack starting from the passed frame
  // -6: problems somewhere, -2 if gc seems to be active, otherwise the count handled frames
  // equivalent to the forte error codes
  int walk(frame top_frame);

  // find the top most (decipherable) java
  // returns the java or fame() (with pc = 0) if a frame is not found or the maximum depth is reached
  frame find_top_java_frame(frame top_frame, int max_depth = -1, int* err_code = NULL);

  // abort now? called after the handlers
  virtual bool abort() const { return false; }

  // called for every frame
  virtual void handle_frame(const frame& frame) {}

  // called for every interpreted or compiled (non inlined) java frame
  virtual void handle_java_frame(const frame& frame) {}

  // called for every interpreted frame
  virtual void handle_interpreted_frame(const frame& frame, Method* method, int bci) {}

  // called for every (JIT) compiled frame
  // if the frame is a compiled frame and a PCDesc is available
  // if inlined: base_frame is the envelopping frame
  virtual void handle_compiled_frame(const frame& base_frame, Method* method, int bci, bool inlined) {}

  // called for all frames that are neither compiled nor interpreted
  virtual void handle_misc_frame(const frame& frame) {}

};

#endif // SHARE_JFR_RECORDER_STACKTRACE_STACKWALKER_HPP
