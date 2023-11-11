#include <gtest/gtest.h>
#include "common/MemoryHeap.h"
using namespace mlinsight;

class CustomClass{
public:
    int a=0;
    int b=0;
    CustomClass(int a,int b):a(a),b(b){

    }
};
TEST(ObjectPoolHeap, mallocFree){
    ObjectPoolHeap<CustomClass> objPool(1);
    ASSERT_EQ(objPool.allocatedSize,0);

    CustomClass* object1=objPool.alloc();
    ASSERT_EQ(objPool.allocatedSize,1);
    new (object1) CustomClass(1,1);

    CustomClass* object2=objPool.alloc();
    ASSERT_EQ(objPool.allocatedSize,2);
    new (object2) CustomClass(2,2);


    object1->~CustomClass();
    objPool.dealloc(object1);
    ASSERT_EQ(objPool.allocatedSize,1);
    object2->~CustomClass();
    objPool.dealloc(object2);
    ASSERT_EQ(objPool.allocatedSize,0);
    ASSERT_EQ(object2,nullptr);
}

TEST(ObjectPoolHeap, copyConstructor){
    ObjectPoolHeap<CustomClass>* objPool=new ObjectPoolHeap<CustomClass>(1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,0);
    CustomClass* object1=objPool->alloc();
    new (object1) CustomClass(1,1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,1);

    ObjectPoolHeap<CustomClass> objPoolCpy(*objPool);
    //Manually free objPool
    delete objPool;
    ASSERT_EQ(objPoolCpy.largestChunkSize,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,0);
    CustomClass* object2=objPoolCpy.alloc();
    new (object2) CustomClass(20,20);
    ASSERT_EQ(objPoolCpy.largestChunkSize,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,1);
    ASSERT_DEATH(object1->a=2,".*");
    object2->a=20;
    ASSERT_EQ(object2->a,20);
}

TEST(ObjectPoolHeap, copyAssignment){
    ObjectPoolHeap<CustomClass>* objPool=new ObjectPoolHeap<CustomClass>(1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,0);
    CustomClass* object1=objPool->alloc();
    new (object1) CustomClass(1,1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,1);

    ObjectPoolHeap<CustomClass> objPoolCpy;
    objPoolCpy=*objPool;
    //Manually free objPool
    delete objPool;
    ASSERT_EQ(objPoolCpy.largestChunkSize,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,0);
    CustomClass* object2=objPoolCpy.alloc();
    new (object2) CustomClass(1,1);
    ASSERT_EQ(objPoolCpy.largestChunkSize,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,1);
    ASSERT_DEATH(object1->a=2,".*");
    object2->a=20;
    ASSERT_EQ(object2->a,20);
}

TEST(ObjectPoolHeap, moveConstruct){
    ObjectPoolHeap<CustomClass>* objPool=new ObjectPoolHeap<CustomClass>(1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,0);
    CustomClass* object1=objPool->alloc();
    new (object1) CustomClass(1,1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,1);

    ObjectPoolHeap<CustomClass> objPoolCpy(std::move(*objPool));
    //Manually free objPool
    delete objPool;
    ASSERT_EQ(objPoolCpy.largestChunkSize,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,1);
    CustomClass* object2=objPoolCpy.alloc();
    new (object2) CustomClass(1,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,2);
    object1->a=21;
    object2->a=22;
    ASSERT_EQ(object1->a,21);
    ASSERT_EQ(object2->a,22);
}

TEST(ObjectPoolHeap, moveAssignment){
    ObjectPoolHeap<CustomClass>* objPool=new ObjectPoolHeap<CustomClass>(1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,0);
    CustomClass* object1=objPool->alloc();
    new (object1) CustomClass(1,1);
    ASSERT_EQ(objPool->largestChunkSize,1);
    ASSERT_EQ(objPool->allocatedSize,1);

    ObjectPoolHeap<CustomClass> objPoolCpy;
    objPoolCpy=std::move(*objPool);
    //Manually free objPool
    delete objPool;
    ASSERT_EQ(objPoolCpy.largestChunkSize,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,1);
    CustomClass* object2=objPoolCpy.alloc();
    new (object2) CustomClass(1,1);
    ASSERT_EQ(objPoolCpy.allocatedSize,2);
    object1->a=21;
    object2->a=22;
    ASSERT_EQ(object1->a,21);
    ASSERT_EQ(object2->a,22);
}