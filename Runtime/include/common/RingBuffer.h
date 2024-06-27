/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_RINGBUFFER_H
#define MLINSIGHT_RINGBUFFER_H

#include <sys/mman.h>
#include "common/LazyValueType.h"

namespace mlinsight {
    template<typename T>
    class RingBufferIterator;

    template<typename VALUE_TYPE>
    class RingBuffer {
    private:
        ssize_t internalArrSize;
        VALUE_TYPE *buffer = nullptr;
        ssize_t head = 0;
        ssize_t tail = 0;

        friend class RingBufferIterator<VALUE_TYPE>;

    public:

        RingBuffer(ssize_t internalArrSize) : internalArrSize(internalArrSize + 1) {
            //There should be one more element that stores tail node
            buffer = reinterpret_cast<VALUE_TYPE *>(mmap(NULL, this->internalArrSize * sizeof(VALUE_TYPE),
                                                         PROT_READ | PROT_WRITE,
                                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        }

        /**
         * Copy and swap idiom: Copy constructor.
         */
        RingBuffer(const RingBuffer &rho) : internalArrSize(rho.internalArrSize) {
            buffer = reinterpret_cast<VALUE_TYPE *>(malloc(rho.internalArrSize * sizeof(VALUE_TYPE)));
            for (int i = 0; i < internalArrSize; ++i) {
                new(buffer + i) VALUE_TYPE(rho.buffer[i]);
            }
        }

        /**
         * Copy and swap idiom: Move constructor.
         */
        RingBuffer(RingBuffer &&rho) noexcept: buffer(rho.buffer),
                                               internalArrSize(rho.internalArrSize) {
            //Prevent rho's destructor from deleting internalArray
            rho.buffer = nullptr;
        }

        /**
         * Copy and swap idiom: Copy operator.
         */
        RingBuffer &operator=(const RingBuffer &rho) {
            if (this != &rho) {
                RingBuffer tempObject(rho);
                swap(*this, tempObject);
            }
            return *this;
        }

        /**
         * Copy and swap idiom: Move operator.
         */
        RingBuffer &operator=(RingBuffer &&rho) {
            swap(*this, rho);
            return *this;
        }

        /**
         * Copy and swap idiom: swap.
         */
        friend void swap(RingBuffer &lho, RingBuffer &rho) noexcept {
            using std::swap;//Fallback to std::swap is not find appropriate swap
            swap(lho.internalArrSize, rho.internalArrSize);
            swap(lho.buffer, rho.buffer);
            swap(lho.head, rho.head);
            swap(lho.tail, rho.tail);
            swap(lho.size, rho.size);
        }

        ~RingBuffer() {
            if (buffer) {
                munmap(buffer, internalArrSize * sizeof(VALUE_TYPE));
            }
        }

        template<typename ...Args>
        inline VALUE_TYPE &enqueue(Args &&... args) {
            VALUE_TYPE *rawMemory = enqueueLazyConstruct();
            new(rawMemory) VALUE_TYPE(std::forward<Args>(args)...);
            return *rawMemory;
        }

        // Add an item to this circular buffer.
        inline VALUE_TYPE *enqueueLazyConstruct() {
            assert(!isFull());
            VALUE_TYPE *ret = reinterpret_cast<VALUE_TYPE *>(&buffer[tail]);
            tail = (tail + 1) % internalArrSize;
            return ret;
        }

        template<typename ...Args>
        inline VALUE_TYPE &forceEnqueue(Args &&... args) {
            VALUE_TYPE *rawMemory = forceEnqueueLazyConstruct();
            if (rawMemory) {
                new(rawMemory) VALUE_TYPE(std::forward<Args>(args)...);
            }
            return rawMemory;
        }

        /*
        * Add an item to this buffer. If buffer is full, dequeue first and then enqueue.
        */
        inline VALUE_TYPE &forceEnqueueLazyConstruct() {
            if (isFull()) {
                head = (head + 1) % internalArrSize;
            }
            return *enqueueLazyConstruct();
        }

        inline VALUE_TYPE &dequeue() {
            assert(!isEmpty());
            VALUE_TYPE *ret = reinterpret_cast<VALUE_TYPE *>(&buffer[head]);
            head = (head + 1) % internalArrSize;
            return *ret;
        }

        inline VALUE_TYPE *front() {
            if (isEmpty()) {
                return nullptr;
            }
            return buffer[head];
        }

        inline bool isEmpty() { return head == tail; }

        inline bool isFull() { return (tail + 1) % internalArrSize == head; }

        inline size_t size() {
            if (tail >= head) {
                return tail - head;
            }
            return internalArrSize - head - tail;
        }
    };

    template<typename VALUE_TYPE>
    class RingBufferIterator {
    public:

        RingBufferIterator(RingBuffer<VALUE_TYPE> &ringBuffer) : ringBuffer(ringBuffer), curHead(ringBuffer.head) {

        }

        void operator++() {
            assert(hasNext());
            curHead = (curHead + 1) % (ringBuffer->internalArrSize);
        }

        void operator++(int) {
            operator++();
        }

        void operator--() {
            assert(hasNext());
            curHead = (curHead + 1) % (ringBuffer->internalArrSize);
        }

        bool operator==(const RingBufferIterator &rho) const override {
            return ringBuffer == rho.ringBuffer && curHead == rho.curHead;
        }

        inline bool operator!=(const RingBufferIterator &rho) const override {
            return !operator==(rho);
        }

        virtual VALUE_TYPE &operator*() {
            return ringBuffer->buffer[curHead];
        }

    protected:
        ssize_t curHead = 0;

        RingBuffer<VALUE_TYPE> *ringBuffer = nullptr;

        bool hasNext(ssize_t curPosi) {
            return curPosi != ringBuffer->tail;
        }

        bool hasPrev(ssize_t curPosi) {
            return curPosi != ringBuffer->head;
        }
    };
}
#endif