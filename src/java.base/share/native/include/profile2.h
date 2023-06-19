/*
 * Copyright (c) 1997, 2022, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
 */

#ifndef JVM_PROFILE2_H
#define JVM_PROFILE2_H

#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>

#include "jni.h"

enum ASGST_Capabilities {
  ASGST_REGISTER_QUEUE = 1,
  ASGST_MARK_FRAME = 2
};

int ASGST_Capabilities();
struct _ASGST_Iterator;
typedef struct _ASGST_Iterator ASGST_Iterator;

enum ASGST_FrameTypeId {
  ASGST_FRAME_JAVA         = 1, // JIT compiled and interpreted
  ASGST_FRAME_JAVA_INLINED = 2, // inlined JIT compiled
  ASGST_FRAME_NATIVE       = 3, // native wrapper to call C methods from Java
  ASGST_FRAME_CPP          = 4  // C/C++/... frames
};

typedef struct {
  uint8_t type;            // frame type
  int comp_level;      // compilation level, 0 is interpreted, -1 is undefined, > 1 is JIT compiled
  int bci;            // -1 if the bci is not available (like in native frames)
  jmethodID method_id;    // method id or null if not available
  void *pc;          // current program counter inside this frame
  void *sp;          // current stack pointer inside this frame
  void *fp;          // current frame pointer inside this frame
} ASGST_Frame;

enum ASGST_Options {
  ASGST_INCLUDE_NON_JAVA_FRAMES = 1,
  ASGST_END_ON_FIRST_JAVA_FRAME = 2,
};

enum ASGST_TRACE_KIND {
  ASGST_JAVA_TRACE = 1,
  ASGST_NON_JAVA_TRACE = 2
};

enum ASGST_Error {
  ASGST_NO_FRAME         =  0, // come to and end
  ASGST_NO_THREAD        = -1, // thread is not here
  ASGST_THREAD_EXIT      = -2, // dying thread
  ASGST_UNSAFE_STATE     = -3, // thread is in unsafe state
  ASGST_NO_CLASS_LOAD    = -4, // class not loaded
  ASGST_NO_TOP_JAVA_FRAME = -5,
};

// Why not ASGST_CreateIterator? Because we then would have to
// - allocate memory for the iterator at the caller, exposing the size of the iterator
// - free the iterator at the caller, making the API more cumbersome to use
extern "C" JNIEXPORT
int ASGST_RunWithIterator(void* ucontext, int32_t options, void (*fun)(ASGST_Iterator*, void*), void* argument);

// returns 1 if successful, else error code
extern "C" JNIEXPORT
int ASGST_NextFrame(ASGST_Iterator* iter, ASGST_Frame* frame);

// returns error code or 1 if no error
extern "C" JNIEXPORT
int ASGST_State(ASGST_Iterator* iter);

extern "C" JNIEXPORT
int ASGST_ThreadState();

extern "C" JNIEXPORT
int ASGST_ThreadState();

#endif // JVM_PROFILE2_H
