#ifndef MLINSIGHT_PYHOOK_H
#define MLINSIGHT_PYHOOK_H
#include <Python.h>
#include <frameobject.h>
#include <ceval.h>
#include "trace/type/PyCodeExtra.h"
namespace mlinsight {
    bool installPythonInterceptor();

    PyCodeExtra *getPyCodeExtra(PyFrameObject *f);
}

#endif