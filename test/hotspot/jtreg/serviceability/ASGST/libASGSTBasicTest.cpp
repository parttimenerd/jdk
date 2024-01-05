/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Copyright (c) 2019, Google and/or its affiliates. All rights reserved.
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
 */

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "util.hpp"
#include "profile.h"



extern "C" {

static
jint Agent_Initialize(JavaVM *jvm, char *options, void *reserved) {
  return initAgent(jvm, options, reserved);
}

JNIEXPORT
jint JNICALL Agent_OnLoad(JavaVM *jvm, char *options, void *reserved) {
  return Agent_Initialize(jvm, options, reserved);
}

JNIEXPORT
jint JNICALL Agent_OnAttach(JavaVM *jvm, char *options, void *reserved) {
  return Agent_Initialize(jvm, options, reserved);
}

JNIEXPORT
jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
  return JNI_VERSION_1_8;
}


int collectFrame(ASGST_FrameInfo *info, std::vector<ASGST_FrameInfo> *frames) {
  frames->push_back(*info);
  return 1;
}


JNIEXPORT jboolean JNICALL
Java_BasicTest_BasicTest_checkBasicStackWalk(JNIEnv* env, jclass cls) {

  std::vector<ASGST_FrameInfo> frames;

  ASGST_Frame top_frame = ASGST_GetFrame(nullptr, true);
  if (top_frame.pc == nullptr) {
    fprintf(stderr, "ASGST_GetFrame returned null\n");
    return false;
  }
  ASGST_WalkStackFromFrame(top_frame, (ASGST_WalkStackCallback)collectFrame, nullptr, &frames, 0);

  if (frames.size() <= 0) {
    fprintf(stderr, "The num_frames must be positive: %ld\n", frames.size());
    return false;
  }

  // check that the the top frame is a native frame
  if (frames[0].type != ASGST_FRAME_JAVA_NATIVE) {
    fprintf(stderr, "The top frame must be a native frame: %d\n", frames[0].type);
    return false;
  }

  if (frames[0].method == nullptr) {
    fprintf(stderr, "The top frame must have a method: %p\n", frames[0].method);
    return false;
  }

  JvmtiDeallocator<char*> name;


  jvmtiError err = jvmti->GetMethodName(frames[0].method, name.get_addr(), NULL, NULL);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "Error in GetMethodName: %d\n", err);
    return false;
  }

  if (name.get() == NULL) {
    fprintf(stderr, "Name is NULL\n");
    return false;
  }

  if (strcmp(name.get(), "checkBasicStackWalk") != 0) {
    fprintf(stderr, "Name is not checkBasicStackWalk: %s\n", name.get());
    return false;
  }

  // AsyncGetCallTrace and GetStackTrace should return comparable frames
  // so we obtain the frames using GetStackTrace and compare them.

  jthread thread;
  jvmti->GetCurrentThread(&thread);
  jvmtiFrameInfo gstFrames[1024];
  jint gstCount = 0;

  jvmti->GetStackTrace(thread, 0, 1024, gstFrames, &gstCount);

  if (gstCount != (int)frames.size()) {
    fprintf(stderr, "GetStackTrace and ASGST return different number of frames: %d vs %ld)", gstCount, frames.size());
    return false;
  }

  for (int i = 0; i < gstCount; ++i) {
    if (frames[i].type == ASGST_FRAME_JAVA_NATIVE) {
      if (gstFrames[i].location != -1) {
        fprintf(stderr, "%d: ASGST found native frame but GST did not\n", i);
        return false;
      }
    } else {
      if (gstFrames[i].method != frames[i].method) {
        fprintf(stderr, "%d: method_id mismatch: %p vs %p\n", i, gstFrames[i].method, frames[i].method);
        return false;
      }
    }
  }
  return true;
}

struct Param {
  std::vector<ASGST_FrameInfo> frames;
  size_t max_frames;
};

int collectNFrames(ASGST_FrameInfo *info, void *arg) {
  Param *param = (Param*)arg;
  param->frames.push_back(*info);
  if (param->frames.size() >= param->max_frames) {
    return 0;
  }
  return 1;
}

JNIEXPORT jboolean JNICALL
Java_BasicTest_BasicTest_check2Frames(JNIEnv* env, jclass cls) {

  Param param;
  param.max_frames = 2;

  ASGST_Frame top_frame = ASGST_GetFrame(nullptr, true);
  if (top_frame.pc == nullptr) {
    fprintf(stderr, "ASGST_GetFrame returned null\n");
    return false;
  }
  ASGST_WalkStackFromFrame(top_frame, (ASGST_WalkStackCallback)collectNFrames, nullptr, (void*)&param, 0);

  if (param.frames.size() != 2) {
    fprintf(stderr, "The num_frames must be 2: %ld\n", param.frames.size());
    return false;
  }
  return true;
}

}
