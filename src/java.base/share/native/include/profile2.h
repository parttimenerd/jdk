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
#include "profile2.h"

struct _ASGST_Method;
typedef struct _ASGST_Method* ASGST_Method;

struct _ASGST_Class;
typedef struct _ASGST_Class* ASGST_Class;

typedef struct _ASGST_FrameMark ASGST_FrameMark;

// implementations don't have to implement all methods,
// only the iterator related and those that match their capabilities
enum ASGST_Capabilities {
  ASGST_REGISTER_QUEUE = 1, // everything safepoint queue related
  ASGST_MARK_FRAME     = 2  // frame marking related
};

struct _ASGST_Iterator;
typedef struct _ASGST_Iterator ASGST_Iterator;

enum ASGST_FrameTypeId {
  ASGST_FRAME_JAVA         = 1, // JIT compiled and interpreted
  ASGST_FRAME_JAVA_INLINED = 2, // inlined JIT compiled
  ASGST_FRAME_JAVA_NATIVE  = 3, // native wrapper to call C methods from Java
  ASGST_FRAME_NON_JAVA     = 4  // C/C++/... frames
};

typedef struct {
  uint8_t type;         // frame type
  int comp_level;       // compilation level, 0 is interpreted, -1 is undefined, > 1 is JIT compiled
  int bci;              // -1 if the bci is not available (like in native frames)
  ASGST_Method method;  // method or nullptr if not available
  void *pc;             // current program counter inside this frame
  void *sp;             // current stack pointer inside this frame, might be null
  void *fp;             // current frame pointer inside this frame, might be null
} ASGST_Frame;

enum ASGST_Options {
  ASGST_INCLUDE_NON_JAVA_FRAMES = 1,
  // end at the first/top most java frame, but don't process it, just obtain pc, fp and sp
  // but not method or bci, used if you want to deal with the Java frames at the safe point
  ASGST_END_ON_FIRST_JAVA_FRAME = 2,
};

// There are different kinds of traces depending
// on the purpose of the currently running code
// in the walked thread, all above 1 are implementation specific
enum ASGST_TRACE_KIND {
  ASGST_JAVA_TRACE     = 1,
  ASGST_NON_JAVA_TRACE = 2
};

enum ASGST_Error {
  ASGST_NO_FRAME            =  0, // come to and end
  ASGST_NO_THREAD           = -1, // thread is not here
  ASGST_THREAD_EXIT         = -2, // dying thread
  ASGST_UNSAFE_STATE        = -3, // thread is in unsafe state
  ASGST_NO_TOP_JAVA_FRAME   = -4, // no top java frame
  ASGST_ENQUEUE_NO_QUEUE    = -5, // no queue registered
  ASGST_ENQUEUE_FULL_QUEUE  = -6, // safepoint queue is full
  ASGST_ENQUEUE_OTHER_ERROR = -7, // other error, like currently at safepoint
  // everything lower than -16 is implementation specific
};

// Note: Safepoint == Thread Local Handshake

extern "C" {

// returns the supported capabilities
JNIEXPORT
int ASGST_Capabilities();

typedef void (*ASGST_IteratorHandler)(ASGST_Iterator* iterator, void* arg);

// Create an iterator and pass it to fun alongside the passed argument.
// @return error or kind
//
// Signal safe, has to be called on thread that belongs to the frame.
JNIEXPORT
int ASGST_RunWithIterator(void* ucontext, int32_t options, ASGST_IteratorHandler fun, void* argument);

// Similar to RunWithIterator, but starting from a frame (sp, fp, pc) instead of a ucontext.
//
// Signal safe, has to be called on thread that belongs to the frame.
JNIEXPORT
int ASGST_RunWithIteratorFromFrame(void* sp, void* fp, void* pc, int options, ASGST_IteratorHandler fun, void* argument);

// Rewind an interator to the top most frame
//
// Signal safe, has to be called on thread that belongs to the frame.
JNIEXPORT
void ASGST_RewindIterator(ASGST_Iterator* iterator);

// Obtains the next frame from the iterator
// @returns 1 if successful, else error code (< 0) / end (0)
// @see ASGST_State
//
// Typically used in a loop like:
//
// ASGST_Frame frame;
// while (ASGST_NextFrame(iterator, &frame) == 1) {
//   // do something with the frame
// }
//
// When using ASGST_END_ON_FIRST_JAVA_FRAME, then the first byte-code backed
// frame (not native Java) causes ASGST_NextFrame to return 0 and to
// only populate the pc, fp and sp fields of the frame, the frame
// has type ASGST_FRAME_JAVA. On error the return value is the error
// code and the frame type is 0, as well as all pointer fields.
//
// Signal safe, has to be called on thread that belongs to the frame.
JNIEXPORT
int ASGST_NextFrame(ASGST_Iterator* iterator, ASGST_Frame* frame);

// State of the iterator, corresponding to the next frame return code
// @returns error code or 1 if no error
// if iterator is null or at end, return ASGST_NO_FRAME,
// returns a value < -16 if the implementation encountered a specific error
//
// Signal safe, has to be called on thread that belongs to the frame.
JNIEXPORT
int ASGST_State(ASGST_Iterator* iterator);

// Returns state of the current thread, which is a subset
// of the JVMTI thread state.
// no JVMTI_THREAD_STATE_INTERRUPTED, limited JVMTI_THREAD_STATE_SUSPENDED.
//
// Signal safe, has to be called on thread that belongs to the frame.
JNIEXPORT
int ASGST_ThreadState();

// Returns the jmethodID for a given ASGST_Method, null if the method has
// no corresponding jmethodID.
JNIEXPORT
jmethodID ASGST_MethodToJMethodID(ASGST_Method method);

// Returns the ASGST_Method for a given jmethodID.
//
// Not signal safe, might crash for unloaded methods.
JNIEXPORT
ASGST_Method ASGST_JMethodIDToMethod(jmethodID methodID);

// Method info
// You have to preallocate the strings yourself and store the lengths
// in the appropriate fields, the lengths are set to the respective
// string lengths by the VM, be aware that strings are null-terminated
typedef struct {
  ASGST_Class klass;
  char* method_name;
  jint method_name_length;
  char* signature;
  jint signature_length;
  char* generic_signature;
  jint generic_signature_length;
  jint modifiers;
  jint idnum; // class local id, doesn't change with redefinitions
  jlong class_idnum; // class id, doesn't change with redefinitions
} ASGST_MethodInfo;

// Class info, like the method info
typedef struct {
  char* class_name;
  jint class_name_length;
  char* generic_class_name;
  jint generic_class_name_length;
  jint modifiers;
  jlong idnum; // id that doesn't change with redefinitions
} ASGST_ClassInfo;

// Obtain the method information for a given ASGST_Method and store it in the pre-allocated info struct.
// It stores the actual length in the _len fields and at a null-terminated string in the string fields.
// Safe to call from signal handlers.
// A field is set to null if the information is not available.
//
// Signal safe, but methods can be unloaded concurrently, use unload handler
JNIEXPORT
void ASGST_GetMethodInfo(ASGST_Method method, ASGST_MethodInfo* info);

// Returns the class local id of the method, or 0
//
// signal safe
JNIEXPORT
jint ASGST_GetMethodIdNum(ASGST_Method method);

typedef struct {
  jint start_bci;
  jint line_number;
} ASGST_MethodLineNumberEntry;

// Populates the method line number table, mapping BCI to line number.
// Returns the number of written elements
//
// Signal safe
JNIEXPORT
int ASGST_GetMethodLineNumberTable(ASGST_Method method, ASGST_MethodLineNumberEntry* entries, int length);

// Returns the line number for a given BCI, or -1 if not available
//
// Signal safe
JNIEXPORT
jint ASGST_GetMethodLineNumber(ASGST_Method method, jint bci);

// Similar to GetMethodInfo
//
// Signal safe
JNIEXPORT
void ASGST_GetClassInfo(ASGST_Class klass, ASGST_ClassInfo* info);

// Returns the id of the class that doesn't change with redefinitions
//
// signal safe
JNIEXPORT
jlong ASGST_GetClassIdNum(ASGST_Class klass);

// Returns the class that contains a given method
//
// Signal safe
JNIEXPORT
ASGST_Class ASGST_GetClass(ASGST_Method method);

// Returns the ASGST_Class for a given jclass
//
// Signal safe
JNIEXPORT
ASGST_Class ASGST_JClassToClass(jclass klass);

// Returns the JVMTI class id for a given class,
// used to obtain more information on classes via JVMTI
JNIEXPORT
jclass ASGST_ClassToJClass(ASGST_Class klass);

// handler called with the unloaded class and the methods that were unloaded (pointer + count)
// and whether the class is redefined or unloaded,
// and the argument passed when registering the handler
typedef void (*ASGST_ClassUnloadHandler)(ASGST_Class klass, ASGST_Method *methods, int count, bool redefined, void* arg);

// Register a handler to be called when class is unloaded
//
// not signal and safe point safe
JNIEXPORT
void ASGST_RegisterClassUnloadHandler(ASGST_ClassUnloadHandler handler, void* arg);

// Deregister a handler to be called when a class is unloaded
// @returns true if handler was present
//
// not signal and safe point safe
JNIEXPORT
bool ASGST_DeregisterClassUnloadHandler(ASGST_ClassUnloadHandler handler, void* arg);

// The following functions are only callable if ASGST_REGISTER_QUEUE is a capability

struct _ASGST_Queue;
typedef struct _ASGST_Queue ASGST_Queue;
// handler called at asafe point with iterator, queue argument, enqueue argument
typedef void (*ASGST_Handler)(ASGST_Iterator*, void*, void*);

// Register a queue to the current thread (or the one passed via env)
// @param fun handler called at a safe point with iterators,
// the argument for RegisterQueue and the argument passed via Enqueue
//
// The handler can only call safe point safe methods, which excludes all
// JVMTI methods, but the handler is not called inside a signal handler,
// so allocating or obtaining locks is possible
//
// Not signal safe, requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
ASGST_Queue* ASGST_RegisterQueue(JNIEnv* env, int size, int options, ASGST_Handler fun, void* argument);

// Remove queue, return true if successful
//
// Not signal safe, requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
bool ASGST_DeregisterQueue(JNIEnv* env, ASGST_Queue* queue);

// handler that is called at a safe point with enqueued samples before and after processing
// called with the queue, a frame iterator, and the OnQueue argument
// frame iterator is null if offerIterator at handler registration was false
typedef void (*ASGST_OnQueueSafepointHandler)(ASGST_Queue*, ASGST_Iterator*, void*);

// Set the handler that is called at a safe point before the elements in the (non-empty) queue
// are processed.
//
// @param before handler or null to remove the handler
//
// Not signal safe, requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
void ASGST_SetOnQueueProcessingStart(ASGST_Queue* queue, int options, bool offerIterator, ASGST_OnQueueSafepointHandler before, void* arg);

// Set the handler that is called at a safe point after the elements in the (non-empty) queue
// are processed.
//
// @param after handler or null to remove the handler
//
// Not signal safe, requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
void ASGST_SetOnQueueProcessingEnd(ASGST_Queue* queue, int options, bool offerIterator, ASGST_OnQueueSafepointHandler end, void* arg);

// Enqueue the processing of the current stack at the end of the queue and return the kind (or error if <= 0)
// you have to deal with the top C and native frames yourself (but there is an option for this)
//
// @param argument argument passed through to the ASGST_Handler for the queue as the third argument
// @return kind or error, returns ASGST_ENQUEUE_FULL_QUEUE if queue is full
// or ASGST_ENQUEUE_NO_QUEUE if queue is null
//
// Signal safe, but has to be called with a queue that belongs to the current thread, or the thread
// has to be stopped during the duration of this call
// Requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
int ASGST_Enqueue(ASGST_Queue* queue, void* ucontext, void* argument);

typedef struct {
  void* pc;  // program counter of the top most Java frame
  void* fp;
  void* sp;
  void* arg; // argument passed through to the handler
} ASGST_QueueElement;

// Obtains the element that would be stored in the queue if Enqueue was called
// Returns != 1 if an error occurred or kind is different
//
// arg is set to null
//
// Signal safe
// Requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
int ASGST_GetEnqueuableElement(void* ucontext, ASGST_QueueElement* element);

// Returns the nth element in the queue (from the front),
// 0 gives you the first/oldest element.
// -1 gives you the youngest element, ..., -size the oldest.
//
// Modification of the returned element are allowed, as long as the
// queue's size has not been modified between the call to ASGST_GetQueueElement
// and the modification (e.g. by calling ASGST_ResizeQueue).
//
// Modifiying anything besides the arg field is highly discouraged.
//
// @returns null if n is out of bounds
//
// Signal safe
JNIEXPORT
ASGST_QueueElement* ASGST_GetQueueElement(ASGST_Queue* queue, int n);

typedef struct {
  jint size; // size of the queue
  jint capacity; // capacity of the queue
  jint attempts; // attempts to enqueue since last safepoint end
} ASGST_QueueSizeInfo;

// Returns the number of elements in the queue, its capacity,
// and the number of attempts since finishing the previous safepoint
//
// Signal safe, but only proper values in queues thread
JNIEXPORT
ASGST_QueueSizeInfo ASGST_GetQueueSizeInfo(ASGST_Queue* queue);

// Trigger the resizing of the queue at end of the next safepoint
// (or the current if currently processing one)
//
// Signal safe, but has to be called with a queue that belongs to the current thread
// Requires ASGST_REGISTER_QUEUE capability
JNIEXPORT
void ASGST_ResizeQueue(ASGST_Queue* queue, int size);

// the following requires ASGST_MARK_FRAME capabilities
// and most methods are not signal safe

struct _ASGST_FrameMark;
typedef struct _ASGST_FrameMark ASGST_FrameMark;

// handler when a frame mark is being hit, gets passed the mark, an iterator,
// and the mark argument and
// returns the new mark
typedef void (*ASGST_FrameMarkHandler)(ASGST_FrameMark*, ASGST_Iterator*, void*);


// Register a frame mark
//
// Be aware that it will never be triggered, as the mark points to no frame
// (the stack pointer of the mark is null), use ASGST_MoveFrameMark to move
// the mark. This is typically done in the safe point handler.
//
// @param env thread or null for the current thread
// @param handler called whenever the stack pointer of the unwound frame
//                is larger (older) than the frame mark
// @param options options for the frame iterator
//
// Requires ASGST_MARK_FRAME capability, not signal safe
JNIEXPORT
ASGST_FrameMark* ASGST_RegisterFrameMark(JNIEnv* env, ASGST_FrameMarkHandler handler, int options, void* arg);

// Move the frame mark to a new stack pointer
// Requires ASGST_MARK_FRAME capability, not signal safe
JNIEXPORT
void ASGST_MoveFrameMark(ASGST_FrameMark* mark, void* new_sp);

// Returns the stack pointer of the mark
// Requires ASGST_MARK_FRAME capability, signal safe
JNIEXPORT
void* ASGST_GetFrameMarkStackPointer(ASGST_FrameMark* mark);

// Remove the frame mark
// @param mark mark to remove, has to belong to the specified thread
// Requires ASGST_MARK_FRAME capability, not signal safe
JNIEXPORT
void ASGST_RemoveFrameMark(ASGST_FrameMark* mark);
}
#endif // JVM_PROFILE2_H
