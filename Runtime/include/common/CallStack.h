#ifndef MLINSIGHT_CALLSTACK_H
#define MLINSIGHT_CALLSTACK_H

#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include "common/Logging.h"
#include "trace/type/PyCodeExtra.h"

using namespace std; 

/**
* author: Steven Tang <stevne.tang@bytedance.com>
*/

namespace mlinsight {
template<typename ARRAY_ELEMENT_POINTER_TYPE, int MAX_LEVEL>
class CallStack {
public:
    //A pointer to an void* array storing callstack
    ARRAY_ELEMENT_POINTER_TYPE* array;
    int levels=0;
public:

    CallStack(){
        levels = MAX_LEVEL; 
        array= (ARRAY_ELEMENT_POINTER_TYPE *)malloc(MAX_LEVEL * sizeof(ARRAY_ELEMENT_POINTER_TYPE));
        if(!array){
            fatalError("Cannot allocate memory for callstack");
        }
    }

    /**
    * Copy and swap idiom: copy constructor.
    */
    CallStack(const CallStack& rho):levels(rho.levels){
        //DBG_LOG("Performance warning: Deep copy of callstack maybe extremely slow, try to move rather than copy!")
        array= (ARRAY_ELEMENT_POINTER_TYPE *)malloc(MAX_LEVEL * sizeof(ARRAY_ELEMENT_POINTER_TYPE));
        if(!array){
            fatalError("Cannot allocate memory for callstack");
        }
        for(int i=0;i<this->levels;++i){
            new (array+i) ARRAY_ELEMENT_POINTER_TYPE(rho.array[i]);
            //this->array[i]=rho.array[i];
        }
    }

    /**
    * Copy and swap idiom: move constructor.
    */
    CallStack(CallStack&& rho):levels(rho.levels){
        this->array=rho.array;
        //Prevent rho from freeing this memory.
        rho.array=nullptr;
    }

    /**
    * Copy and swap idiom: copy assignment.
    */
    CallStack& operator=(const CallStack& rho){
        CallStack tempObject(rho);
        swap(*this,tempObject);
        return *this;
    }

   /**
    * Copy and swap idiom: move assignment.
    */
    CallStack& operator=(CallStack&& rho){
        swap(*this,rho);
        return *this;
    }

   /**
    * Copy and swap idiom: move assignment.
    */
    void swap(CallStack& lho,CallStack& rho){
        std::swap(lho.array,rho.array);
        std::swap(lho.levels,rho.levels);
    }

    ~CallStack(){
        if(array){
            free(array);
        }
    }

    inline const ssize_t getMaxDepth() const{
        return MAX_LEVEL;
    }

    bool operator==(const CallStack &rho) const {
        bool isEqual = this->levels == rho.levels;
        if(isEqual){
            for(int i=0;i<this->levels;++i){
                if(!(this->array[i]==rho.array[i])){
                    isEqual=false;
                    break;
                }
            }
        }
        return isEqual;
    }
};

};
#endif