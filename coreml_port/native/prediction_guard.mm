/* Experimental process-local GIL release around synchronous Core ML prediction.
 * No installed wheel or Core ML framework file is modified. This is qualified
 * only against the recorded CPython/Core ML Tools/macOS combination.
 */
#include <Python.h>
#include <CoreML/CoreML.h>
#include <dispatch/dispatch.h>
#include <objc/runtime.h>
#include <dlfcn.h>
#include <atomic>
#include <cstring>

using PredictPlain = id (*)(id, SEL, id, id *);
using PredictState = id (*)(id, SEL, id, id, id *);
static PredictPlain original_plain = nullptr;
static PredictState original_state = nullptr;
static std::atomic<unsigned long> plain_calls(0), state_calls(0), gil_releases(0), synthetic_workers(0);

class ScopedGILRelease {
    PyThreadState *saved = nullptr;
public:
    ScopedGILRelease() {
        if (Py_IsInitialized() && PyGILState_Check()) {
            saved = PyEval_SaveThread();
            gil_releases.fetch_add(1);
        }
    }
    void restore() {
        if (saved) {
            PyThreadState *thread = saved;
            saved = nullptr;
            PyEval_RestoreThread(thread);
        }
    }
    ~ScopedGILRelease() { restore(); }
};

static id guarded_plain(id self, SEL selector, id input, id *error) {
    plain_calls.fetch_add(1);
    ScopedGILRelease release;
    @try { return original_plain(self, selector, input, error); }
    @finally { release.restore(); }
}

static id guarded_state(id self, SEL selector, id input, id state, id *error) {
    state_calls.fetch_add(1);
    ScopedGILRelease release;
    @try { return original_state(self, selector, input, state, error); }
    @finally { release.restore(); }
}

static bool valid_method(Method method, unsigned arguments) {
    if (!method || method_getNumberOfArguments(method) != arguments) return false;
    char type[16];
    method_getReturnType(method, type, sizeof(type));
    if (std::strcmp(type, "@") != 0) return false;
    Dl_info image;
    if (!dladdr((void *)method_getImplementation(method), &image)) return false;
    return image.dli_fname && std::strstr(image.dli_fname, "CoreML") != nullptr;
}

extern "C" int violetto_install_prediction_guard() {
    if (original_plain && original_state) return 0;
    Class cls = objc_getClass("MLModel");
    if (!cls) return 1;
    Method plain = class_getInstanceMethod(cls, sel_registerName("predictionFromFeatures:error:"));
    Method state = class_getInstanceMethod(cls, sel_registerName("predictionFromFeatures:usingState:error:"));
    if (!valid_method(plain, 4)) return 2;
    if (!valid_method(state, 5)) return 3;
    original_plain = (PredictPlain)method_getImplementation(plain);
    original_state = (PredictState)method_getImplementation(state);
    method_setImplementation(plain, (IMP)guarded_plain);
    method_setImplementation(state, (IMP)guarded_state);
    return 0;
}

extern "C" unsigned long violetto_prediction_plain_calls() { return plain_calls.load(); }
extern "C" unsigned long violetto_prediction_state_calls() { return state_calls.load(); }
extern "C" unsigned long violetto_prediction_gil_releases() { return gil_releases.load(); }
extern "C" unsigned long violetto_prediction_synthetic_workers() { return synthetic_workers.load(); }

struct SyntheticWait { dispatch_semaphore_t done; };
static void synthetic_worker(void *context) {
    SyntheticWait *wait = (SyntheticWait *)context;
    PyGILState_STATE state = PyGILState_Ensure();
    synthetic_workers.fetch_add(1);
    PyGILState_Release(state);
    dispatch_semaphore_signal(wait->done);
}

/* Called through ctypes.PyDLL, which retains the GIL. Mode 0 intentionally
 * deadlocks until its disposable child process is killed by the test parent.
 * Mode 1 exercises the same release scope as the Objective-C wrappers. */
extern "C" int violetto_synthetic_prediction(int release_gil) {
    SyntheticWait wait = {dispatch_semaphore_create(0)};
    if (release_gil) {
        ScopedGILRelease release;
        dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), &wait, synthetic_worker);
        dispatch_semaphore_wait(wait.done, DISPATCH_TIME_FOREVER);
    } else {
        dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), &wait, synthetic_worker);
        dispatch_semaphore_wait(wait.done, DISPATCH_TIME_FOREVER);
    }
    return 0;
}
