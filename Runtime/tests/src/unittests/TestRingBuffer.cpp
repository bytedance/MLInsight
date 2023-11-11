#include <gtest/gtest.h>
#include "common/RingBuffer.h"
using namespace mlinsight;

TEST(RingBuffer, isEmpty) {
    RingBuffer<int> myRingbuffer(10);
    ASSERT_TRUE(myRingbuffer.isEmpty());
}

TEST(RingBuffer, isFull) {
    RingBuffer<int> myRingbuffer(1);
    ASSERT_FALSE(myRingbuffer.isFull());
    myRingbuffer.enqueue(1);
    ASSERT_TRUE(myRingbuffer.isFull());
}

TEST(RingBuffer, enqueueAndDequeue) {
    RingBuffer<int> myRingbuffer(4);
    ASSERT_EQ(myRingbuffer.enqueue(1),1);
    ASSERT_EQ(myRingbuffer.size(),1);
    ASSERT_EQ(myRingbuffer.enqueue(2),2);
    ASSERT_EQ(myRingbuffer.size(),2);
    ASSERT_EQ(myRingbuffer.enqueue(3),3);
    ASSERT_EQ(myRingbuffer.size(),3);
    ASSERT_EQ(myRingbuffer.enqueue(4),4);
    ASSERT_EQ(myRingbuffer.size(),4);
    ASSERT_TRUE(myRingbuffer.isFull());
    ASSERT_DEATH(myRingbuffer.enqueue(5),".*!isFull\\(\\).*");

    ASSERT_EQ(myRingbuffer.dequeue(),1);
    ASSERT_EQ(myRingbuffer.size(),3);
    ASSERT_EQ(myRingbuffer.dequeue(),2);
    ASSERT_EQ(myRingbuffer.size(),2);
    ASSERT_EQ(myRingbuffer.dequeue(),3);
    ASSERT_EQ(myRingbuffer.size(),1);
    ASSERT_EQ(myRingbuffer.dequeue(),4);
    ASSERT_EQ(myRingbuffer.size(),0);

    ASSERT_DEATH(myRingbuffer.dequeue(),".*!isEmpty\\(\\).*");
}

TEST(RingBuffer, copyConstruct) {
    RingBuffer<int>* myRingbuffer=new RingBuffer<int>(4);
    ASSERT_EQ(myRingbuffer->enqueue(1),1);
    ASSERT_EQ(myRingbuffer->enqueue(2),2);
    ASSERT_EQ(myRingbuffer->enqueue(3),3);
    ASSERT_EQ(myRingbuffer->enqueue(4),4);

    RingBuffer<int> myRingbufferCopy(*myRingbuffer);

}