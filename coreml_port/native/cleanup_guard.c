/* Project-local compatibility guard for the pinned Core ML Tools 9.0 bridge.
 * Its PybindCompatibleArray compiler-generated .cxx_destruct decrefs a
 * py::array without acquiring the GIL. Core ML can invoke it on a worker.
 * This wrapper changes only that bridge destructor, never model arithmetic.
 */
#include <Python.h>
#include <objc/runtime.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
static IMP original;
static _Atomic unsigned long calls, background_calls, shutdown_skips;
static void guarded_destruct(id object, SEL selector) {
    atomic_fetch_add(&calls,1);
    /* During interpreter teardown, leave the Python reference to process exit.
       Calling the allocator after finalization is unsafe. */
    if (!Py_IsInitialized() || _Py_IsFinalizing()) {
        atomic_fetch_add(&shutdown_skips,1);
        return;
    }
    if (!PyGILState_Check()) atomic_fetch_add(&background_calls,1);
    PyGILState_STATE state=PyGILState_Ensure();
    ((void (*)(id, SEL))original)(object,selector);
    PyGILState_Release(state);
}
int violetto_install_cleanup_guard(void) {
    if (original) return 0;
    Class cls=objc_getClass("PybindCompatibleArray");
    if (!cls) return 1;
    Method method=class_getInstanceMethod(cls,sel_registerName(".cxx_destruct"));
    if (!method || method_getNumberOfArguments(method)!=2) return 2;
    IMP found=method_getImplementation(method);
    Dl_info image;
    if (!dladdr((void*)found,&image) || !strstr(image.dli_fname,"libcoremlpython.so")) return 3;
    char type[16];method_getReturnType(method,type,sizeof(type));
    if (strcmp(type,"v")) return 4;
    original=found;
    method_setImplementation(method,(IMP)guarded_destruct);
    return 0;
}
unsigned long violetto_guard_calls(void) { return atomic_load(&calls); }
unsigned long violetto_guard_background_calls(void) { return atomic_load(&background_calls); }
unsigned long violetto_guard_shutdown_skips(void) { return atomic_load(&shutdown_skips); }
