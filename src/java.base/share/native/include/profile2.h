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
#include "jni_md.h"

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
  ASGST_FRAME_JAVA_NATIVE  = 3, // native wrapper to call C methods from Java
  ASGST_FRAME_NON_JAVA     = 4  // C/C++/... frames
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
  // end at the first/top most java frame, but don't process it, just obtain pc, fp and sp
  // but not method or bci, used if you want to deal with the Java frames at the safe point
  ASGST_END_ON_FIRST_JAVA_FRAME = 2,
};

enum ASGST_TRACE_KIND {
  ASGST_JAVA_TRACE = 1,
  ASGST_NON_JAVA_TRACE = 2
};

enum ASGST_Error {
  ASGST_NO_FRAME           =  0, // come to and end
  ASGST_NO_THREAD          = -1, // thread is not here
  ASGST_THREAD_EXIT        = -2, // dying thread
  ASGST_UNSAFE_STATE       = -3, // thread is in unsafe state
  ASGST_NO_CLASS_LOAD      = -4, // class not loaded
  ASGST_NO_TOP_JAVA_FRAME  = -5,
  ASGST_ENQUEUE_NO_QUEUE   = -6,
  ASGST_ENQUEUE_FULL_QUEUE = -7,
};

// Why not ASGST_CreateIterator? Because we then would have to
// - allocate memory for the iterator at the caller, exposing the size of the iterator
// - free the iterator at the caller, making the API more cumbersome to use
extern "C" {
JNIEXPORT
int ASGST_RunWithIterator(void* ucontext, int32_t options, void (*fun)(ASGST_Iterator*, void*), void* argument);

int ASGST_RunWithIteratorFromFrame(void* sp, void* fp, void* pc, int options, void (*fun)(ASGST_Iterator*, void*), void* argument);

// returns 1 if successful, else error code
JNIEXPORT
int ASGST_NextFrame(ASGST_Iterator* iter, ASGST_Frame* frame);

// returns error code or 1 if no error
JNIEXPORT
int ASGST_State(ASGST_Iterator* iter);

JNIEXPORT
int ASGST_ThreadState();

struct _ASGST_Queue;
typedef struct _ASGST_Queue ASGST_Queue;
typedef void (*ASGST_Handler)(ASGST_Iterator*, void*, void*);

// Register a queue to the current thread (or the one passed via env)
// @param fun handler called at safe point with iterators, the argument for RegisterQueue and the argument passed via Enqueue
// not signal safe
JNIEXPORT
ASGST_Queue* ASGST_RegisterQueue(JNIEnv* env, int size, int options, ASGST_Handler fun, void* argument);

// Enqueue the processing of the current stack and return the kind (or error if <= 0)
// you have to deal with the top C and native frames yourself (but there is an option for this)
// @param argument argument passed through to the ASGST_Handler for the queue as the third argument
// signal safe, but has to be called with a queue that belongs to the current thread
JNIEXPORT
int ASGST_Enqueue(ASGST_Queue* queue, void* ucontext, void* argument);

}
#endif // JVM_PROFILE2_H
