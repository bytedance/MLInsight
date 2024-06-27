
#ifndef MLINSIGHT_PYTORCHPROXY_H
#define MLINSIGHT_PYTORCHPROXY_H

#include <Python.h>
#include "common/Logging.h"
#include "trace/proxy/PyTorchCallBacks.h"
#include "trace/tool/MLInsightPyAPI.h"

/**
 * This file holds the definition of the Python modules in  MLInsight
 */
namespace mlinsight {


    static PyMethodDef mlinsightModuleFunctions[]{
            {"forward_pre_hook",            (PyCFunction) forward_pre_hook,            METH_VARARGS, "forward_pre_hook"},
            {"forward_hook",                (PyCFunction) forward_hook,                METH_VARARGS, "forward_hook"},
            {"full_backward_pre_hook",      (PyCFunction) full_backward_pre_hook,      METH_VARARGS, "full_backward_pre_hook"},
            {"full_backward_hook",          (PyCFunction) full_backward_hook,          METH_VARARGS, "full_backward_hook"},
            {"module_registration_hook",    (PyCFunction) module_registration_hook,    METH_VARARGS, "module_registration_hook"},
            {"parameter_registration_hook", (PyCFunction) parameter_registration_hook, METH_VARARGS, "parameter_registration_hook"},
            {NULL, NULL, 0, NULL} /* sentinel */
    };


    static struct PyModuleDef mlInsightModuleDef = {
            PyModuleDef_HEAD_INIT,
            "mlinsight",
            NULL,
            -1,
            mlinsightModuleFunctions
    };

    static PyMethodDef mlinsightPyApiModuleFunctions[]{
            {"INFO_LOG",            (PyCFunction) infoLog,                      METH_VARARGS, "Print info log to MLInsight output file"},
            {"monkeypatch",         (PyCFunction) monkeyPatchInstaller,         METH_VARARGS, "Print info log to MLInsight output file"},
            {"monkeypatchFunction", (PyCFunction) monkeyPatchFunctionInstaller, METH_VARARGS, "Print info log to MLInsight output file"},
            {NULL, NULL, 0, NULL} /* sentinel */
    };

    //This is seperate from mlinsight, because mlinsight is not callable by the Python programs, and is initialized after Python interpreter is initialized.
    static struct PyModuleDef mlInsightPyApiDef = {
            PyModuleDef_HEAD_INIT,
            "mlinsightpyapi",
            NULL,
            -1,
            mlinsightPyApiModuleFunctions
    };

}
#endif
