#include "common/TensorObj.h"
#include "trace/hook/PyHook.h"
#include "trace/type/PyCodeExtra.h"

namespace mlinsight {
    PyCallStack tmpPyCallStack;
     CCallStack tmpCCallStack;

    void FrameworkTensorMixin::updateCallstack() {
        tmpPyCallStack.snapshot();

        //Get the calstack id
        auto insertionIter = pyCallStackRegistery->find(&tmpPyCallStack); //The comparator in pyCallStackRegistery will compare value rather than compare pointer. That's why we pass in &tmpPyCallStack;
        if(insertionIter==pyCallStackRegistery->end()){
            ssize_t newCallStackId=globalCallStackIdCounter.fetch_add(1);
            tmpPyCallStack.callstackID=newCallStackId;
            tmpPyCallStack.isNewCallStackId=true;

            PyCallStack* newCallStackObj = pyCallStackHeap->alloc();
            new (newCallStackObj) PyCallStack(tmpPyCallStack); //Create a new callstack object from template. We use a heap here to ensure the pointer to callstack object is always valid.

            auto newObjIter=pyCallStackRegistery->emplace_hint(insertionIter,newCallStackObj, newCallStackId);
            this->callstack=newCallStackObj; //pyCallStackRegistry should not remove any callstack and should be insertion only.
        }else{
            this->callstack=insertionIter->first;
            this->callstack->isNewCallStackId=false;
        }
        assert(this->callstack);
        //ERR_LOGS("Obj %p updated callstack this->callstack=%p", this, this->callstack);
    }

    FrameworkTensorMixin::FrameworkTensorMixin(ssize_t size, void *ptr) : TensorCallstackMixin(size, ptr) {

    }

    void DriverTensorMixin::updateCallStack() {
        tmpCCallStack.snapshot();

        //Get the calstack id
        auto insertionIter = cCallStackRegistery->find(&tmpCCallStack); //The comparator in cCallStackRegistery will compare value rather than compare pointer. That's why we pass in &tmpPyCallStack;
        if(insertionIter==cCallStackRegistery->end()){
            ssize_t newCallStackId=globalCallStackIdCounter.fetch_add(1);
            tmpCCallStack.callstackID=newCallStackId;
            tmpCCallStack.isNewCallStackId=true;

            CCallStack* newCallStackObj = cCallStackHeap->alloc();
            new (newCallStackObj) CCallStack(tmpCCallStack); //Create a new callstack object from template. We use a heap here to ensure the pointer to callstack object is always valid.

            auto newObjIter=cCallStackRegistery->emplace_hint(insertionIter,newCallStackObj, newCallStackId);
            this->callstack=newCallStackObj; //pyCallStackRegistry should not remove any callstack and should be insertion only.
        }else{
            this->callstack=insertionIter->first;
            this->callstack->isNewCallStackId=false;
        }
        assert(this->callstack);
    }

    DriverTensorMixin::DriverTensorMixin(ssize_t size, void *ptr) : TensorCallstackMixin(size, ptr) {

    }
}