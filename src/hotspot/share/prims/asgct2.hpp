
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
enum Error {
  NO_JAVA_FRAME         =  0,
  NO_CLASS_LOAD         = -1,
  GC_ACTIVE             = -2,
  UNKNOWN_NOT_JAVA      = -3,
  NOT_WALKABLE_NOT_JAVA = -4,
  UNKNOWN_JAVA          = -5,
  UNKNOWN_STATE         = -7,
  THREAD_EXIT           = -8,
  DEOPT                 = -9
};

enum FrameTypeId {
  FRAME_JAVA         = 1, // JIT compiled and interpreted
  FRAME_JAVA_INLINED = 2, // inlined JIT compiled
  FRAME_NATIVE       = 3, // native wrapper to call C methods from Java
  FRAME_STUB         = 4, // VM generated stubs
  FRAME_CPP          = 5  // C/C++/... frames
};

typedef struct {
  uint8_t type;            // frame type
  uint8_t comp_level;      // compilation level, 0 is interpreted
  uint16_t bci;            // 0 < bci < 65536
  jmethodID method_id;
} JavaFrame;               // used for FRAME_JAVA and FRAME_JAVA_INLINED

typedef struct {
  FrameTypeId type;  // frame type
  void *pc;          // current program counter inside this frame
} NonJavaFrame;

typedef union {
  FrameTypeId type;     // to distinguish between JavaFrame and NonJavaFrame
  JavaFrame java_frame;
  NonJavaFrame non_java_frame;
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
  JNIEnv *env_id;                 // environment to record trace for or null for current thread environment
  jint num_frames;                // number of frames in this trace
  CallFrame *frames;              // frames
  void* frame_info;               // more information on frames
} CallTrace;

enum Options {
  INCLUDE_C_FRAMES = 1
};

}


// AsyncGetCallTrace2() entry point.
//
// New version of of AsyncGetCallTrace()
// See JEP draft https://bugs.openjdk.java.net/browse/JDK-8284289 for more information
extern "C" {
JNIEXPORT
void AsyncGetCallTrace2(asgct2::CallTrace *trace, jint depth, void* ucontext, int32_t options);
}

#endif // SHARE_PRIMS_ASGCT2