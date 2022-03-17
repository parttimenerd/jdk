
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

#ifndef SHARE_PRIMS_ASGCT2
#define SHARE_PRIMS_ASGCT2

#include "gc/shared/collectedHeap.inline.hpp"
#include "memory/universe.hpp"
#include "oops/oop.inline.hpp"
#include "prims/jvmtiExport.hpp"
#include "compiler/compilerDefinitions.hpp"
#include <sys/types.h>


namespace asgct2 {
// error codes, equivalent to the forte error code for AsyncGetCallTrace
enum class Error : jint {
  NO_JAVA_FRAME         =  0,
  NO_CLASS_LOAD         = -1,
  GC_ACTIVE             = -2,
  UNKNOWN_NOT_JAVA      = -3,
  NOT_WALKABLE_NOT_JAVA = -4,
  UNKNOWN_JAVA          = -5,
  // NOT_WALKABLE_JAVA     = -6, // not used
  UNKNOWN_STATE         = -7,
  THREAD_EXIT           = -8,
  DEOPT                 = -9,
  // SAFEPOINT             = -10 // not used
};

enum class FrameTypeId : uint8_t {
    FRAME_INTERPRETED = 0,
    FRAME_JIT         = 1,
    FRAME_INLINE      = 2,
    FRAME_NATIVE      = 3,
    FRAME_CPP         = 4   // stubs are C frames with CompLevel_all
};

typedef struct {
  void *machine_pc;           // program counter, for C and native frames (frames of native methods)
  uint8_t type;               // frame type (single byte)
  uint8_t comp_level;         // highest compilation level of a method related to a Java frame
  // information from original CallFrame
  jint bci;                   // bci for Java frames
  jmethodID method_id;        // method ID for Java frames
} CallFrame;



// Enumeration to distinguish tiers of compilation, only >= 0 are used
/*enum CompLevel {
  CompLevel_any               = -1,        // Used for querying the state
  CompLevel_all               = -1,        // Used for changing the state  // unused
  CompLevel_none              = 0,         // Interpreter
  CompLevel_simple            = 1,         // C1
  CompLevel_limited_profile   = 2,         // C1, invocation & backedge counters
  CompLevel_full_profile      = 3,         // C1, invocation & backedge counters + mdo
  CompLevel_full_optimization = 4          // C2 or JVMCI
};*/

typedef struct {
    JNIEnv *env_id;                   // Env where trace was recorded
    jint num_frames;                  // number of frames in this trace
    CallFrame *frames;                // frames
    void *frame_info;
} CallTrace;

}


// AsyncGetCallTrace2() entry point.
//
// New version of of AsyncGetCallTrace()
//
// void (*AsyncGetCallTrace2)(CallTrace *trace, jint depth, void* ucontext)
//
// Called by the profiler to obtain the current method call stack trace for
// a given thread. The thread is identified by the env_id field in the
// CallTrace structure. The profiler agent should allocate a CallTrace
// structure with enough memory for the requested stack depth. The VM fills in
// the frames buffer and the num_frames field.
//
// Arguments:
//
//   trace    - trace data structure to be filled by the VM.
//   depth    - depth of the call stack trace.
//   ucontext - ucontext_t of the LWP
//
// CallTrace:
//   typedef struct {
//     JNIEnv *env_id;                   // Env where trace was recorded
//     jint num_frames;                  // number of frames in this trace
//     CallFrame *frames;                // frames
//     void* frame_info;                 // more information on frames
//   } CallTrace;
//
// Fields:
//   env_id     - ID of thread which executed this trace.
//   num_frames - number of frames in the trace.
//                (< 0 indicates the frame is not walkable).
//   frames     - the ASGCT_CallFrames that make up this trace. Callee followed by callers.
//
// ASGCT_CallFrame:
// typedef struct {
//   void *machine_pc;           // program counter, for C and native frames (frames of native methods)
//   uint8_t type;               // frame type (single byte)
//   uint8_t comp_level;         // highest compilation level of a method related to a Java frame
//   // information from original CallFrame
//   jint bci;                   // bci for Java frames
//   jmethodID method_id;        // method ID for Java frames
// } CallFrame;
extern "C" {
JNIEXPORT
void AsyncGetCallTrace2(asgct2::CallTrace *trace, jint depth, void* ucontext);
}

#endif // SHARE_PRIMS_ASGCT2