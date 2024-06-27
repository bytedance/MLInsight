
#ifndef MLINSIGHT_MEMORYHEAP_H
#define MLINSIGHT_MEMORYHEAP_H
/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <sys/mman.h>
#include <cassert>
#include <cstdio>
#include <sys/mman.h>
#include <cstring>
#include <cassert>
#include <cerrno>
#include <utility>
#include "common/Logging.h"
#include "common/LazyValueType.h"

namespace mlinsight {


    /**
     * A no-op driverMemRecord heap that delegate all allocations and frees to system driverMemRecord allocator.
     * This class do not handle object construct/deconstruct
     */
    template<class T>
    class PassThroughMemoryHeap {
    public:
        T *alloc() {
            return reinterpret_cast<T *>( malloc(sizeof(T)));
        }

        void dealloc(T *&obj) {
            free(obj);
#ifndef NDEBUG
            obj = nullptr;
#endif
        }

        T *allocArray(ssize_t arraySize) {
            return reinterpret_cast<T *>(calloc(arraySize, sizeof(T)));
        }

        void deallocArray(T *&obj) {
            free(obj);
        }

    };

    template<class T>
    class ObjectPoolHeap;

    template<class T>
    struct FreeListNode {
        LazyConstructValue<T> object;
        FreeListNode *next = nullptr;
    };

    /**
     * A pre-allocated driverMemRecord chunk that stores a group of slots. Each slot can hold one object of type T. Used in ObjectPoolHeap.
     */
    template<class T>
    class Chunk {
    protected:
        //The starting address of driverMemRecord segment. This memory is page aligned because it is allocated by mmap
        FreeListNode<T> *internalArray = nullptr;
        //Pointer to the next chunk. (Only used when freeing all chunks in the destructor of ObjectPoolHeap)
        Chunk *nextChunk = nullptr;
        //The number of objects in this chunk
        ssize_t internalArrSize = 0;
        //Slots tracked in freelistHead. If a block is allocated, it will be either in use or in the freelistHead.
        //If no block is available one block will be allocated in this chunk.
        ssize_t allocatedSize = 0;
    public:
        Chunk(ssize_t objNum) : nextChunk(nullptr), internalArrSize(objNum) {
            internalArray = (FreeListNode<T> *) (mmap(NULL, internalArrSize * sizeof(FreeListNode<T>),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            if (internalArray == MAP_FAILED) {
                fatalErrorS("Failed to allocateArray driverMemRecord for MemoryHeap::Chunk at %p because: %s", internalArray,
                            strerror(errno));
                exit(-1);
            }
        }

        /**
         * Copy and swap idiom: Copy constructor.
         * Do not allow deep copy chunk because chunk stores unused object cache. Deep copy should be implemented
         * from upper pyCallStackLevel class.
         */
        Chunk(const Chunk &rho) = delete;

        /**
         * Copy and swap idiom: Move constructor.
         */
        Chunk(const Chunk &&rho) : internalArray(rho.internalArray), nextChunk(rho.nextChunk),
                                   internalArrSize(rho.internalArrSize),
                                   allocatedSize(rho.allocatedSize) {
            //Prevent rho's destructor from freeing rho.internalArray
            rho.internalArray = nullptr;
        };

        /**
         * Copy and swap idiom: Copy assignment.
         * Do not allow deep copy chunk, because chunk usually stores unused object cache. Deep copy should be implemented
         * from upper pyCallStackLevel.
         */
        Chunk &operator=(const Chunk &rho) = delete;

        /**
         * Copy and swap idiom: Move assignment.
         */
        Chunk &operator=(Chunk &&rho) {
            //Invoke move constructor to move rho into a temporary object.
            Chunk tempObject(rho);
            swap(*this, tempObject);
        };

        friend void swap(Chunk &lho, Chunk &rho) {
            using std::swap;
            swap(lho.internalArray, rho.internalArray);
            swap(lho.nextChunk, rho.nextChunk);
            swap(lho.internalArrSize, rho.internalArrSize);
            swap(lho.allocatedSize, rho.allocatedSize);
        }

        inline bool isFull() const {
            return allocatedSize == internalArrSize;
        }

        ~Chunk() {
            if (internalArray) {
                if (munmap(internalArray, internalArrSize * sizeof(FreeListNode<T>)) == -1) {
                    fatalErrorS("Failed to deallocate driverMemRecord for MemoryHeap::Chunk at %p because: %s", internalArray,
                                strerror(errno));
                    exit(-1);
                }

            }
        }

        friend class ObjectPoolHeap<T>;
    };

    /**
     * Cache object when freed.
     * @tparam T
     */
    template<class T>
    class ObjectPoolHeap {
    protected:
        size_t largestChunkSize;
        Chunk<T> *chunkHead;
        FreeListNode<T> *freelistHead;
        ssize_t allocatedSize = 0; //The number of objects in this object pool
    public:
        explicit ObjectPoolHeap(size_t initialChunkSize = 32) : largestChunkSize(initialChunkSize) {
            assert(initialChunkSize >= 1);
            chunkHead = new Chunk<T>(largestChunkSize);
            freelistHead = new FreeListNode<T>();
        }

        /**
         * Copy and swap idiom: Copy constructor.
         * Construct self as a temporary object based on rho.
         * Since object pool contains all reserved driverMemRecord, we do not need to copy actual driverMemRecord from the original pool.
         * We only copy the size of rho and construct that many driverMemRecord in one chunk.
         */
        ObjectPoolHeap(const ObjectPoolHeap &rho) : ObjectPoolHeap(rho.allocatedSize) {
            //rho shouldn't be * this, user needs to modify code.
            assert(this != &rho);
            //Initialize chunk head to be the entire size of rho but do not copy element.
        }

        /**
         * Copy and swap idiom: Move constructor.
         */
        ObjectPoolHeap(ObjectPoolHeap &&rho) : largestChunkSize(rho.largestChunkSize), chunkHead(rho.chunkHead),
                                               freelistHead(rho.freelistHead),
                                               allocatedSize(rho.allocatedSize) {
            //Prevent rho destructor from deleting this object again.
            rho.chunkHead = nullptr;
            //Prevent rho destructor from deleting this object again.
            rho.freelistHead = nullptr;
        }

        /**
         * Copy and swap idiom: Copy assignment.
         */
        ObjectPoolHeap &operator=(const ObjectPoolHeap &rho) {
            //Invoke copy constructor to create a temporary object, then swap
            if (this != &rho) {
                ObjectPoolHeap tempObj(rho);
                swap(*this, tempObj);
            }
            return *this;
        }

        /**
         * Copy and swap idiom: Move assignment.
         */
        ObjectPoolHeap &operator=(ObjectPoolHeap &&rho) {
            //Invoke copy constructor to create a temporary object, then swap.
            swap(*this, rho);
            return *this;
        }

        friend void swap(ObjectPoolHeap &lho, ObjectPoolHeap &rho) noexcept {
            using std::swap;
            swap(lho.largestChunkSize, rho.largestChunkSize);
            swap(lho.chunkHead, rho.chunkHead);
            swap(lho.allocatedSize, rho.allocatedSize);
            swap(lho.freelistHead, rho.freelistHead);
        }

    public:

        ~ObjectPoolHeap() {
            //Release all chunks
            while (chunkHead) {
                Chunk<T> *cur = chunkHead;
                chunkHead = chunkHead->nextChunk;
                delete cur;
            }
            if (freelistHead) {
                delete freelistHead;
            }

        }

        void dealloc(T *&obj) {
            //Add this object back to the head of freelistHead
            FreeListNode<T> *recycledNode = (FreeListNode<T> *) obj;
            recycledNode->next = freelistHead->next;
            freelistHead->next = recycledNode;
#ifndef NDEBUG
            obj = nullptr;
#endif
            this->allocatedSize -= 1;
        }

        T *alloc() {
            FreeListNode<T> *returnNode = nullptr;
            if (!freelistHead->next) {
                //If there are no dealloc nodes in freelistHead, use unallocated entry from current chunk
                if (chunkHead->isFull()) {
                    //No dealloc entries in the current slot either.
                    largestChunkSize *= 2;
                    //Allocate a chunk that is larger
                    expand(largestChunkSize);
                }
                //Use the next dealloc slot in this chunk
                returnNode = chunkHead->internalArray + chunkHead->allocatedSize;
                chunkHead->allocatedSize += 1;
            } else {
                //If there are free nodes, use one dealloc node.
                returnNode = freelistHead->next;
                //Pop firstFreeListNode from freelistHead
                freelistHead->next = returnNode->next;
            }
            this->allocatedSize += 1;
            //The first part of returned object is the object we want. The rest is a next pointer but this pointer is hidden to the user.
            return reinterpret_cast<T *>(returnNode);
        }

    protected:
        inline void expand(ssize_t targetSize) {
            //Need a larger chunk
            //The dealloc list is used up, allocateArray more chunks
            auto *newChunk = new Chunk<T>(targetSize);
            if (!newChunk) {
                fatalError("Cannot allocateArray driverMemRecord in MemoryHeap.");
            }
            newChunk->nextChunk = chunkHead;
            chunkHead = newChunk;
        }

    };
}

#endif
