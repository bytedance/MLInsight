#ifndef MLINSIGHT_TENSOR_H
#define MLINSIGHT_TENSOR_H

#include "common/CallStack.h"
#include "common/Tool.h"
#include "trace/type/PyCodeExtra.h"

namespace mlinsight {
/**
 * A frameworkGeneral, framework independent tensor object.
 *  Do not use this class directly. Use subclasses instead.
 *
 * To reduce runtime overhead, we do not use virtual methods. Subclasses of Tensor appear as mixins.
 * @tparam MIXIN_TYPES A set of mixin classes that ends with "Tensor", which can expand attributes in this base class.
 */
    template<typename... MIXIN_TYPES>
    class TensorObj : public MIXIN_TYPES ... {
    public:
        TensorObj(ssize_t size, void *ptr) : MIXIN_TYPES(size, ptr)... {

        }
    };

    template<typename CALLSTACK_TYPE>
    class TensorCallstackMixin {
    public:
        ssize_t size = 0;      // current object size;
        void *ptr = nullptr; // The real driverMemRecord address
        // The following is only available for an alive object
        // Please keep the second parameter to use macro PYTHON_CALL_STACK_LEVEL. Because this callstack type is used elsewhere,
        CALLSTACK_TYPE* callstack=nullptr;

    protected: //Do not allow direct use of this base class, please use subclasses instead. This is because we do not use virtual keyword for performance. So there should be no duplicated methods.

        /**
        * Constructor used when first allocating a block
        * @return
        */
        TensorCallstackMixin(ssize_t size, void *ptr) : size(size), ptr(ptr) {

        }

    };

/**
 * DriverTensor
 */
    class DriverTensorMixin : public TensorCallstackMixin<CCallStack> {
    public:
        DriverTensorMixin(ssize_t size, void *ptr);

        bool isAllocatedByFramework=false; //A flag indicating whether this driver object can be linked with a framework object. a.k.a is this alloc called by the framework allocator.

        void updateCallStack();
    };

/**
 * Freamework Tensor
 * @tparam MIXIN_TYPES
 */
    class FrameworkTensorMixin : public TensorCallstackMixin<PyCallStack> {
    public:

        FrameworkTensorMixin(ssize_t size, void *ptr);

        void updateCallstack();

    };


}
#endif //MLINSIGHT_TENSOR_H
