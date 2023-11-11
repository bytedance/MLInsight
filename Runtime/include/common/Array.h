/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_ARRAY_H
#define MLINSIGHT_ARRAY_H

#include <cassert>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <sys/mman.h>
#include "trace/tool/Math.h"
#include "common/Logging.h"
namespace mlinsight {

    /**
     * Auto expansion array. Won't call
     * Won't call any external function for read-only operation
     * @tparam T Value type
     * @tparm Args Default constructor arguments
     */
    template<typename T>
    class Array {
    protected:
        ssize_t internalArrSize = 0;
        ssize_t size = 0;
    public:
        //todo: change this to protected
        T *internalArray = nullptr;
    public:

        Array() = default;

        explicit Array(const ssize_t &initialSize):internalArrSize(initialSize) {
            if(initialSize>0){
                internalArray = (T *) malloc(internalArrSize * sizeof(T));

                if(!internalArray){
                    fatalError("Cannot allocateArray memory for internalArray");
                }
                assert(internalArray != nullptr);
                memset(internalArray, 0, internalArrSize * sizeof(T));
                //INFO_LOGS("Internal array %d bytes",internalArrSize * sizeof(VALUE_TYPE));
            }
        }


        /**
         * Copy and swap idiom: Copy constructor.
         */
        Array(const Array &rho) : internalArrSize(rho.internalArrSize), size(rho.size) {
            //User needs to fix code, &rho should not be this.
            assert(&rho!=this);
            internalArray = (T *) malloc(rho.internalArrSize * sizeof(T));
            for(int i=0;i<internalArrSize;++i){
                new (internalArray+i*sizeof(T)) T(rho.internalArray[i]);
            }        
        }

        /**
         * Copy and swap idiom: Move constructor.
         */
        Array(Array &&rho) noexcept :internalArray(rho.internalArray),size(rho.size),internalArrSize(rho.internalArrSize)  {
            //Prevent rho's destructor from deleting internalArray
            rho.internalArray=nullptr;
        }

        /**
         * Copy and swap idiom: Copy operator.
         */
         Array& operator=(const Array& rho){
             if(this!=&rho){
                 Array tempObject(rho);
                 swap(*this, tempObject);
             }
             return *this;
         }
        /**
         * Copy and swap idiom: Move operator.
         */
        Array& operator=(Array&& rho){
            swap(*this, rho);
            return *this;
        }

        /**
         * Copy and swap idiom: swap.
         */
        friend void swap(Array& lho,Array& rho) noexcept{
            using std::swap;//Fallback to std::swap is not find appropriate swap
            swap(lho.internalArrSize,rho.internalArrSize);
            swap(lho.size,rho.size);
            swap(lho.internalArray,rho.internalArray);
        }

        virtual ~Array() {
            if (internalArray){
                free(internalArray);
            }
        }


        bool isEmpty() const {
            return size == 0;
        }

        inline T &operator[](const ssize_t &index) {
            if(!(0 <= index && index < size)){
                INFO_LOGS("%zd,%zd",index,size);
            }
            assert(0 <= index && index < size);
            assert(internalArray != nullptr);
            return internalArray[index];
        }

        inline T &get(const ssize_t &index) {
            assert(0 <= index && index < size);
            assert(internalArray != nullptr);
            return internalArray[index];
        }

        inline void erase(const ssize_t &index) {
            assert(0 <= index && index < size);
            size -= 1;
            memmove(internalArray + index, internalArray + index + 1, (size - index) * sizeof(T));
        }


        /**
         * Insert at index and mlinsight::Array will return a reference to an object initialized with default arguments.
         * This function will perform no memory copy.
         * @param index The new element will be inserted after index.
         * @return Uninitialized memory
         */
        template<typename... Args>
        inline T& insert(ssize_t index, Args&&... args) {
            T* rawMemory= insertLazyConstruct(index);
            new (rawMemory) T(std::forward<Args>(args)...);
            //insertAtTemplate is for perfect forwarding. Core logic is in insetAtRaw.
            return *rawMemory;
        }

        /**
         * Insert at index, but mlinsight::Array will return uninitialized memory.
         * This function is useful if the user wants to control memory construction.
         * @param index The new element will be inserted after index.
         * @return Uninitialized memory
         */
        inline T* insertLazyConstruct(ssize_t index) {
            assert(0 <= index && index <= size);
            if (size+1 > internalArrSize) {
                expand((size+1) * 2);
            }
            if (index < size) {
                memmove(internalArray + index + 1, internalArray + index, (size - index) * sizeof(T));
            }
            size += 1;

            return internalArray + index;
        }


        /**
         * Allocate a bunch of objects and return an array.
         * @param amount: The number of objects in this array.
         * @param templateObj: An lvalue object used to initialize array elements.
         */
        T* allocateArray(ssize_t amount, T& templateObj) {
            T* ret= allocateArrayRaw(amount);
            for(int i=0;i<amount;++i){
                new(ret + i) T(templateObj);
            }
            return ret;
        }

        /**
         * Allocate a bunch of objects and return an array.
         * There will be no extra memory copy. Objects will be constructed with the default constructor.
         * @param index The new element will be inserted after index.
         * @return Uninitialized memory
         */
        template<typename... Args>
        T* allocateArray(ssize_t amount,Args... args) {
            ssize_t requiredSize = size + amount;
            if (requiredSize > internalArrSize)
                expand(requiredSize * 2);

            T* ret= internalArray + size;
            for(ssize_t i=0;i<amount;++i){
                new (ret+i) T(std::forward<Args>(args)...);
            }
            size += amount;
            return ret;
        }
        /**
         * Allocate a bunch of objects and return an array.
         * But mlinsight::Array will return uninitialized memory to ensure there is no memory copy.
         * @param index The new element will be inserted after index.
         * @return Uninitialized memory
         */
        T* allocateArrayRaw(ssize_t amount) {
            ssize_t requiredSize = size + amount;
            if (requiredSize > internalArrSize)
                expand(requiredSize * 2);

            T* ret= internalArray + size;
            size += amount;
            return ret;
        }

        /**
         * Insert value at the back of this array
         */
        template<typename... Args>
        inline T& pushBack(Args&&... args) {
            T* rawMemory = insertLazyConstruct(size);
            new (rawMemory) T(std::forward<Args>(args)...);
            return *rawMemory;
        }

        /**
         * Insert value at the back of this array. But returns unintialized memory.
         * See insertLazyConstruct.
         */
        inline T* pushBackRaw() {
            return insertLazyConstruct(size);
        }

        inline void popBack() {
            size -= 1;
        }

        inline ssize_t getSize() {
            return size;
        }

        inline ssize_t getTypeSizeInBytes() {
            return sizeof(T);
        }

        inline bool willExpand() {
            return size == internalArrSize;
        }

        inline void clear() {
            size = 0;
        }

        T *data() const {
            return internalArray;
        }

    protected:

        bool expand(ssize_t newSize) {
            //INFO_LOGS("Array expansion %zd",newSize);
            T *oldInternalArr = internalArray;

            internalArray = (T *) malloc(newSize * sizeof(T));
            if(!internalArray){
                fprintf(stderr,"Cannot allocateArray memory");
                exit(-1);
                return false;
            }

            if(oldInternalArr){
                for(ssize_t i=0;i<internalArrSize;++i){
                    new (internalArray+i) T(oldInternalArr[i]);
                }
            }
            free(oldInternalArr);
            internalArrSize = newSize;
            return true;
        }
    };

}
#endif
