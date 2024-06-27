
#ifndef MLINSIGHT_PYTORCHCALLBACK_H
#define MLINSIGHT_PYTORCHCALLBACK_H

#include <Python.h>
#include "common/Logging.h"

namespace mlinsight {

    PyObject *forward_pre_hook(PyObject * self, PyObject * args) __attribute((weak));

    PyObject *forward_hook(PyObject * self, PyObject * args) __attribute((weak));

    PyObject *full_backward_pre_hook(PyObject * self, PyObject * args) __attribute((weak));

    PyObject *full_backward_hook(PyObject * self, PyObject * args) __attribute((weak));

    PyObject *module_registration_hook(PyObject * self, PyObject * args) __attribute((weak));

    PyObject *parameter_registration_hook(PyObject * self, PyObject * args) __attribute((weak));

}
#endif
