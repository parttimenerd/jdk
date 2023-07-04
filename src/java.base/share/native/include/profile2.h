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

#include <cstddef>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>

#include "jni.h"
#include "jni_md.h"

struct _ASGST_Method;
typedef struct _ASGST_Method* ASGST_Method;

struct _ASGST_Class;
typedef struct _ASGST_Class* ASGST_Class;

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
  ASGST_Method method;    // method or nullptr if not available
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

JNIEXPORT
bool ASGST_DeregisterQueue(JNIEnv* env, ASGST_Queue* queue);

// Enqueue the processing of the current stack and return the kind (or error if <= 0)
// you have to deal with the top C and native frames yourself (but there is an option for this)
// @param argument argument passed through to the ASGST_Handler for the queue as the third argument
// signal safe, but has to be called with a queue that belongs to the current thread
JNIEXPORT
int ASGST_Enqueue(ASGST_Queue* queue, void* ucontext, void* argument);


// Returns the jmethodID for a given ASGST_Method, null if the method has
// no corresponding jmethodID.
JNIEXPORT
jmethodID ASGST_MethodToJMethodID(ASGST_Method method);

typedef struct {
  ASGST_Class klass;
  char* method_name;
  jint method_name_length;
  char* signature;
  jint signature_length;
  char* generic_signature;
  jint generic_signature_length;
  jint modifiers;
} ASGST_MethodInfo;

#define ASGST_METHOD_INFO(variable_name, method_name_length, signature_length, generic_signature_length) \
  char variable_name##__method_name[method_name_length];\
  char variable_name##__signature[signature_length];\
  char variable_name##__generic_signature[generic_signature_length];\
  ASGST_MethodInfo variable_name;\
  variable_name.method_name = (char*)variable_name##__method_name;\
  variable_name.method_name_length = method_name_length;\
  variable_name.signature = (char*)variable_name##__signature;\
  variable_name.signature_length = signature_length;\
  variable_name.generic_signature = (char*)variable_name##__generic_signature;\
  variable_name.generic_signature_length = generic_signature_length;

typedef struct {
  char* class_name;
  jint class_name_length;
  char* generic_class_name;
  jint generic_class_name_length;
  jint modifiers;
} ASGST_ClassInfo;

#define ASGST_CLASS_INFO(variable_name, class_name_length, generic_class_name_length) \
  char variable_name##__class_name[class_name_length];\
  char variable_name##__generic_class_name[generic_class_name_length];\
  ASGST_ClassInfo variable_name;\
  variable_name.class_name = (char*)variable_name##__class_name;\
  variable_name.class_name_length = class_name_length;\
  variable_name.generic_class_name = (char*)variable_name##__generic_class_name;\
  variable_name.generic_class_name_length = generic_class_name_length;

// Obtain the method information for a given ASGST_Method and store it in the pre-allocated info struct.
// It stores the actual length in the _len fields and at a null terminated string in the string fields.
// Safe to call from signal handlers.
// A field is set to null if the information is not available.
JNIEXPORT
void ASGST_GetMethodInfo(ASGST_Method method, ASGST_MethodInfo* info);

JNIEXPORT
void ASGST_GetClassInfo(ASGST_Class klass, ASGST_ClassInfo* info);

JNIEXPORT
ASGST_Class ASGST_GetClass(ASGST_Method method);

JNIEXPORT
jclass ASGST_ClassToJClass(ASGST_Class klass);

typedef void (*ASGST_ClassUnloadHandler)(ASGST_Class klass, ASGST_Method *methods, size_t count);

// not signal and safe point safe
JNIEXPORT
void ASGST_RegisterClassUnloadHandler(ASGST_ClassUnloadHandler handler);
}
#endif // JVM_PROFILE2_H
