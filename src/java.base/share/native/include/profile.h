/*
 * Copyright (c) 2024, SAP SE or its affiliates. All rights reserved.
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

#ifndef JVM_PROFILE_H
#define JVM_PROFILE_H

#include "jni.h"
#include "jni_md.h"
#include <stdint.h>

enum ASGST_FrameTypeId {
  ASGST_FRAME_JAVA         = 1, // JIT compiled and interpreted
  ASGST_FRAME_JAVA_INLINED = 2, // inlined JIT compiled
  ASGST_FRAME_JAVA_NATIVE  = 3, // native wrapper to call C methods from Java
};

typedef struct {
  void *pc;             // current program counter inside this frame
  void *sp;             // current stack pointer inside this frame, might be null
  void *fp;             // current frame pointer inside this frame, might be null
} ASGST_Frame;

typedef struct {
  uint8_t type;         // frame type
  int comp_level;       // compilation level, 0 is interpreted, -1 is undefined, > 1 is JIT compiled
  int bci;              // -3 if the bci is not available (like in native frames)
  jmethodID method;     // method or nullptr if not available
  ASGST_Frame frame;
} ASGST_FrameInfo;

enum ASGST_Options {
  // Reconstitute frames, improves the number of frames walked,
  // only available at safepoints
  ASGST_RECONSTITUTE = 0x001,
  // Walk Loom threads, only available at safepoints
  ASGST_WALK_LOOM_THREADS = 0x011,
};

enum ASGST_Error {
  ASGST_no_Java_frame         =  0,
  ASGST_no_class_load         = -1,
  ASGST_GC_active             = -2,
  ASGST_unknown_not_Java      = -3,
  ASGST_not_walkable_not_Java = -4,
  ASGST_unknown_Java          = -5,
  ASGST_not_walkable_Java     = -6,
  ASGST_unknown_state         = -7,
  ASGST_thread_exit           = -8,
  ASGST_deopt                 = -9,
  ASGST_safepoint             = -10,
  ASGST_unsupported_option    = -11,
};

extern "C" {

// Callback called for a frame during stack walking.
// Returns 1 to continue walking the stack, all other returns stop the walk.
typedef int (*ASGST_WalkStackCallback)(ASGST_FrameInfo *frame, void *arg);

typedef int (*ASGST_CFrameCallback)(ASGST_Frame *frame, void *arg);

// Walks the stack from the passed frame and calls the Java callback for each frame,
// stopping if the callback returns a value other than 1.
//
// @param javaCallback callback called for Java frames
// @param nonJavaCallback callback called for every direct non-Java parent frame of an emitted Java frame,
//        might be null
// @param arg passed to the callbacks
// @param options options to control the walk
//
// Returns 1 on success or an error code on failure.
JNIEXPORT
int ASGST_WalkStackFromFrame(ASGST_Frame frame,
  ASGST_WalkStackCallback javaCallback,
  ASGST_CFrameCallback nonJavaCallback,
  void *arg, int options);

// Is this pc in the code heap?
JNIEXPORT
int ASGST_IsJavaFrame(ASGST_Frame pc);

// Get the frame for the passed ucontext, or pc=nullptr if not available.
// use the focus on Java version for storing away at safepoints
// supports nullptr for ucontext, takes the current last Java frame
JNIEXPORT
ASGST_Frame ASGST_GetFrame(void* ucontext, bool focusOnJava);

// Callback called at thread-local safepoints, see ASGST_SetSafepointCallback.
typedef void (*ASGST_SafepointCallback)(void *arg);

// Sets the thread-local safepoint callback and argument.
// Unset by passing nullptr for callback and/or arg.
//
// signal and safepoint safe
JNIEXPORT
void ASGST_SetSafepointCallback(ASGST_SafepointCallback callback, void *arg);

// Obtains the thread-local safepoint callback and argument if set and
// stores them in the provided pointers, otherwise sets them to nullptr.
//
// signal and safepoint safe
JNIEXPORT
void ASGST_GetSafepointCallback(ASGST_SafepointCallback *callback, void **arg);

// Trigger a thread-local safepoint
// that will call the callback set with ASGST_SetSafepointCallback.
//
// signal and safepoint safe
JNIEXPORT
void ASGST_TriggerSafePoint();

// Compute the top frame at a safepoint for the passed captured frame.
// This is useful for capturing the top Java frame in a signal handler
// and then walking it at a safepoint.
// Returns a frame with pc=nullptr on error
JNIEXPORT
ASGST_Frame ASGST_ComputeTopFrameAtSafepoint(ASGST_Frame captured);

};

#endif // JVM_PROFILE_H