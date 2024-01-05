#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jvmti.h"

static jvmtiEnv* jvmti;

template <class T>
class JvmtiDeallocator {
 public:
  JvmtiDeallocator() {
    elem_ = NULL;
  }

  ~JvmtiDeallocator() {
    jvmti->Deallocate(reinterpret_cast<unsigned char*>(elem_));
  }

  T* get_addr() {
    return &elem_;
  }

  T get() {
    return elem_;
  }

 private:
  T elem_;
};

static void GetJMethodIDs(jclass klass) {
  jint method_count = 0;
  JvmtiDeallocator<jmethodID*> methods;
  jvmtiError err = jvmti->GetClassMethods(klass, &method_count, methods.get_addr());

  // If ever the GetClassMethods fails, just ignore it, it was worth a try.
  if (err != JVMTI_ERROR_NONE) {
    //fprintf(stderr, "GetJMethodIDs: Error in GetClassMethods: %d\n", err);
  }
}

// AsyncGetCallTrace needs class loading events to be turned on!
static void JNICALL OnClassLoad(jvmtiEnv *jvmti, JNIEnv *jni_env,
                                jthread thread, jclass klass) {
}

static void JNICALL OnClassPrepare(jvmtiEnv *jvmti, JNIEnv *jni_env,
                                   jthread thread, jclass klass) {
  // We need to do this to "prime the pump" and get jmethodIDs primed.
  GetJMethodIDs(klass);
}

static void JNICALL OnVMInit(jvmtiEnv *jvmti, JNIEnv *jni_env, jthread thread) {
  jint class_count = 0;

  // Get any previously loaded classes that won't have gone through the
  // OnClassPrepare callback to prime the jmethods for AsyncGetCallTrace.
  JvmtiDeallocator<jclass*> classes;
  jvmtiError err = jvmti->GetLoadedClasses(&class_count, classes.get_addr());
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "OnVMInit: Error in GetLoadedClasses: %d\n", err);
    return;
  }

  // Prime any class already loaded and try to get the jmethodIDs set up.
  jclass *classList = classes.get();
  for (int i = 0; i < class_count; ++i) {
    GetJMethodIDs(classList[i]);
  }
}

void expectSuccess(jvmtiError err, const char* msg) {
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "%s: Error: %d\n", msg, err);
    exit(1);
  }
}

jint initAgent(JavaVM *jvm, char *options, void *reserved) {
  jint res = jvm->GetEnv((void **) &jvmti, JVMTI_VERSION);
  if (res != JNI_OK || jvmti == NULL) {
    fprintf(stderr, "Error: wrong result of a valid call to GetEnv!\n");
    return JNI_ERR;
  }

  jvmtiError err;
  jvmtiCapabilities caps;
  memset(&caps, 0, sizeof(caps));
  caps.can_get_line_numbers = 1;
  caps.can_get_source_file_name = 1;

  err = jvmti->AddCapabilities(&caps);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "AgentInitialize: Error in AddCapabilities: %d\n", err);
    return JNI_ERR;
  }

  jvmtiEventCallbacks callbacks;
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.ClassLoad = &OnClassLoad;
  callbacks.VMInit = &OnVMInit;
  callbacks.ClassPrepare = &OnClassPrepare;

  err = jvmti->SetEventCallbacks(&callbacks, sizeof(jvmtiEventCallbacks));
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "AgentInitialize: Error in SetEventCallbacks: %d\n", err);
    return JNI_ERR;
  }

  err = jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_CLASS_LOAD, NULL);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr, "AgentInitialize: Error in SetEventNotificationMode for CLASS_LOAD: %d\n", err);
    return JNI_ERR;
  }

  err = jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_CLASS_PREPARE, NULL);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(stderr,
            "AgentInitialize: Error in SetEventNotificationMode for CLASS_PREPARE: %d\n",
            err);
    return JNI_ERR;
  }

  err = jvmti->SetEventNotificationMode(JVMTI_ENABLE, JVMTI_EVENT_VM_INIT, NULL);
  if (err != JVMTI_ERROR_NONE) {
    fprintf(
        stderr, "AgentInitialize: Error in SetEventNotificationMode for VM_INIT: %d\n",
        err);
    return JNI_ERR;
  }

  return JNI_OK;
}