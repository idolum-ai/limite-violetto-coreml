// Deterministically exercise the actual upstream array wrapper on a worker.
// Test-only: never loaded by the application runtime.
#include <Python.h>
#include <pybind11/numpy.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#include <unistd.h>
namespace py=pybind11;
@interface ViolettoArrayConstructor : NSObject
- (id)initWithArray:(py::array)array;
@end
extern "C" void violetto_release_array_on_worker(PyObject *input) {
    py::array array=py::reinterpret_borrow<py::array>(input);
    Class cls=NSClassFromString(@"PybindCompatibleArray");
    id wrapper=[(ViolettoArrayConstructor*)[cls alloc] initWithArray:array];
    // Capture an Objective-C owner, then release it on the dispatch worker.
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY,0), ^{
        usleep(100000);
        (void)[wrapper description];
    });
}
