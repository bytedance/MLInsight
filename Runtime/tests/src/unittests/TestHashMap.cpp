/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <gtest/gtest.h>
#include "common/HashMap.h"

using namespace mlinsight;

TEST(Hashmap, InsertQuery) {
    //Basic store and query

    HashMap<int, int> hashMap(20);

    for (int i = 0; i < 20; ++i) {
        hashMap.insert(i, i);
    }
    //Each bucket should have only one value
    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(hashMap.buckets[i].getSize(), 1);
    }

    //Put existing element, but refuse to replace
    hashMap.insert(0, 0);

    for (int i = 0; i < 5; ++i) {
        hashMap.insert(i, i);
    }

    //First 5 buckets should one value, since the key already exist
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(hashMap.buckets[i].getSize(), 1);
    }

    for (int i = 20; i < 25; ++i) {
        hashMap.insert(i, i);
    }

    //First 5 buckets should two values now
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(hashMap.buckets[i].getSize(), 2);
    }
    for (int i = 6; i < 20; ++i) {
        ASSERT_EQ(hashMap.buckets[i].getSize(), 1);
    }

    for (int i = 0; i < 25; ++i) {
        ASSERT_EQ(*hashMap.find(i), i);
    }

    //Get non-existing
    ASSERT_EQ(hashMap.find(25), nullptr);

    HashMap<int, int> hashMap1(1);

    for (int i = 0; i < 20; ++i) {
        hashMap1.insert(i, i);
    }
    ASSERT_EQ(hashMap1.buckets[0].getSize(), 20);

    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hashMap1.find(i), i);
    }


}


TEST(HashMap, Erase) {
    //Basic stroe and query

    HashMap<int, int> hashMap(20);

    for (int i = 0; i < 25; ++i) {
        hashMap.insert(i, i);
    }


    //Erase a value from one-element buckets
    hashMap.erase(15);
    ASSERT_EQ(hashMap.find(15), nullptr);

    //Erase a value from two-element buckets
    hashMap.erase(2);
    ASSERT_EQ(hashMap.find(2), nullptr);

}

TEST(HashMap, Iteration) {
    HashMap<int, int> hashMap20(20);
    HashMap<int, int> hashMap1(1);
    auto hm20Beg = hashMap20.begin();
    auto hm20End = hashMap20.end();
    //Iterator self-assignment test;
    ASSERT_TRUE(hm20Beg == hm20Beg);
    ASSERT_TRUE(hm20End == hm20End);
    ASSERT_TRUE(hm20Beg == hm20End);
    ASSERT_DEATH(*hm20Beg, ".*curBucketEntry != hashMap->buckets\\[curBucketId\\].getTail\\(\\).*");
    ASSERT_DEATH(*hm20End, ".*curBucketEntry != hashMap->buckets\\[curBucketId\\].getTail\\(\\).*");
    ASSERT_EQ(hashMap1.getSize(),0);
    ASSERT_EQ(hashMap1.getSize(),0);

    auto hm1Beg = hashMap1.begin();
    auto hm1End = hashMap1.end();
    ASSERT_TRUE(hm1Beg == hm1Beg);
    ASSERT_TRUE(hm1End == hm1End);
    ASSERT_TRUE(hm1Beg == hm1End);
    ASSERT_DEATH(*hm1Beg, ".*curBucketEntry != hashMap->buckets\\[curBucketId\\].getTail\\(\\).*");
    ASSERT_DEATH(*hm1End, ".*curBucketEntry != hashMap->buckets\\[curBucketId\\].getTail\\(\\).*");

    hashMap20.insert(12, 12);
    hashMap1.insert(12, 12);
    ASSERT_EQ(hashMap1.getSize(),1);
    ASSERT_EQ(hashMap1.getSize(),1);

    hm20Beg = hashMap20.begin();
    hm20End = hashMap20.end();
    ASSERT_TRUE(hm20Beg == hm20Beg);
    ASSERT_TRUE(hm20End == hm20End);
    ASSERT_TRUE(hm20Beg != hm20End);
    ASSERT_EQ(*hm20Beg, 12);
    ++hm20Beg;
    ASSERT_TRUE(hm20Beg == hm20End);


    hm1Beg = hashMap1.begin();
    hm1End = hashMap1.end();
    ASSERT_TRUE(hm1Beg == hm1Beg);
    ASSERT_TRUE(hm1End == hm1End);
    ASSERT_TRUE(hm1Beg != hm1End);
    ASSERT_EQ(*hm1Beg, 12);
    hm1Beg++;
    ASSERT_TRUE(hm1Beg == hm1End);


    hashMap1.erase(12);
    hashMap20.erase(12);
    ASSERT_EQ(hashMap1.getSize(),0);
    ASSERT_EQ(hashMap1.getSize(),0);

    for (int i = 12; i < 20; i += 2) {
        hashMap1.insert(i, i);
        hashMap20.insert(i, i);
    }
    ASSERT_EQ(hashMap1.getSize(),4);
    ASSERT_EQ(hashMap1.getSize(),4);

    char rlt[] = {18, 16, 14, 12};
    int index = 0;
    auto hashMap1BeginIter=hashMap1.begin();
    const auto& hashMap1EndIter=hashMap1.begin();
    for (; hashMap1BeginIter != hashMap1.end(); ++hashMap1BeginIter) {
        ASSERT_EQ(*hashMap1BeginIter, rlt[index++]);
    }
    index = 0;
    for (auto &elem:hashMap1) {
        ASSERT_EQ(elem, rlt[index++]);
    }

    char rlt1[] = {12, 14, 16, 18};

    index = 0;
    for (auto elem = hashMap20.begin(); elem != hashMap20.end(); ++elem) {
        ASSERT_EQ(*elem, rlt1[index++]);
    }

    index = 0;
    for (auto& elem:hashMap20) {
        ASSERT_EQ(elem, rlt1[index++]);
    }

    //Iterator copy
    auto iter20A = hashMap20.begin();
    ASSERT_EQ(*iter20A, 12);
    ++iter20A;
    ASSERT_EQ(*iter20A, 14);
    auto iter20B = iter20A;
    ++iter20A;
    ASSERT_EQ(*iter20A, 16);
    ++iter20A;
    ASSERT_EQ(*iter20A, 18);
    ++iter20B;
    ASSERT_EQ(*iter20B, 16);
    ++iter20B;

}


TEST(Hashmap, copyconstruct) {
    HashMap<int, int>* hm1=new HashMap<int, int>(2);
    ASSERT_EQ(hm1->insert(1, 1),1);
    ASSERT_EQ(hm1->insert(2, 2),2);
    ASSERT_EQ(hm1->insert(3, 3),3);
    ASSERT_EQ(hm1->insert(4, 4),4);

    HashMap<int, int>* hm2(hm1);
    ASSERT_EQ(*hm2->find(1),1);
    ASSERT_EQ(*hm2->find(2),2);
    ASSERT_EQ(*hm2->find(3),3);
    ASSERT_EQ(*hm2->find(4),4);

    HashMap<int, int>* hmA=new HashMap<int, int>(15);
    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(hmA->insert(i, i),i);
    }
    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hmA->find(i), i);
    }
//    INFO_LOG("======================hmA print");
//    for(int i=0;i<hmA->bucketNum;++i){
//        auto& bucket= hmA->buckets[i];
//        INFO_LOGS("Bucket %d with size %zd",i,bucket.getSize());
//        auto* curEntry=bucket.getHead();
//        for(int j=0;j<bucket.getSize();++j){
//            curEntry=curEntry->getNext();
//            assert(curEntry);
//            INFO_LOGS("     BucketEntry %d = %d",j,curEntry->getValue().getValue());
//        }
//    }

    HashMap<int, int> hmC(*hmA);
    //Manually free hmA
    delete hmA;

    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hmC.find(i), i);
    }
//    INFO_LOG("======================hmC print");

//    for(int i=0;i<hmC.bucketNum;++i){
//        auto& bucket= hmC.buckets[i];
//        INFO_LOGS("Bucket %d with size %zd",i,bucket.getSize());
//        auto* curEntry=bucket.getHead();
//        for(int j=0;j<bucket.getSize();++j){
//            curEntry=curEntry->getNext();
//            assert(curEntry);
//            INFO_LOGS("     BucketEntry %d = %d",j,curEntry->getValue().getValue());
//        }
//    }
}

TEST(Hashmap, copyassignment) {
    HashMap<int, int>* hmA=new HashMap<int, int>(15);
    for (int i = 0; i < 20; ++i) {
        hmA->insert(i, i);
    }
    HashMap<int, int> hmC;
    hmC=*hmA;
    //Manually free hmA
    delete hmA;

    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hmC.find(i), i);
    }
}

TEST(Hashmap, moveconstructor) {
    HashMap<int, int>* hmA=new HashMap<int, int>(15);
    for (int i = 0; i < 20; ++i) {
        hmA->insert(i, i);
    }
    HashMap<int, int> hmC(std::move(*hmA));
    //Manually free hmA
    delete hmA;

    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hmC.find(i), i);
    }
}

TEST(Hashmap, moveassignment) {
    HashMap<int, int>* hmA=new HashMap<int, int>(15);
    for (int i = 0; i < 20; ++i) {
        hmA->insert(i, i);
    }
    HashMap<int, int> hmC;
    hmC=std::move(*hmA);
    //Manually free hmA
    delete hmA;

    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hmC.find(i), i);
    }
}

class CustomClass{
public:
    int a=0;
    int b=0;
    CustomClass(int a,int b):a(a),b(b){

    }
    bool operator==(const CustomClass &rho) const {
        return a==rho.a && b==rho.b;
    }
};

TEST(Hashmap, customClassOperations) {
    HashMap<int, CustomClass> hmA;
    for (int i = 0; i < 20; ++i) {
        hmA.insert(i, i ,i);
    }
    for (int i = 0; i < 20; ++i) {
        ASSERT_EQ(*hmA.find(i), CustomClass(i,i));
    }
}

TEST(Hashmap, clear) {
    HashMap<int, CustomClass> hmA;
    hmA.insert(1,1,1);
    hmA.insert(2,2,2);
    hmA.insert(3,3,3);
    ASSERT_EQ(hmA.find(1)->a,1);
    ASSERT_EQ(hmA.find(2)->a,2);
    ASSERT_EQ(hmA.find(3)->a,3);
    ASSERT_EQ(hmA.getSize(),3);
    hmA.clear();
    ASSERT_EQ(hmA.getSize(),0);
    ASSERT_EQ(hmA.find(1),nullptr);

    //Clear multiple times
    hmA.clear();
    hmA.clear();
}

TEST(Hashmap, insertRepalcement) {
    HashMap<int, CustomClass> hmA;
    bool retFound=false;


    //Should replace, element does not exist
    hmA.findAndInsert(0,true,retFound, 0 ,0);
    ASSERT_EQ(retFound,false);
    hmA.clear();

    //Should replace, element exists
    CustomClass parm1(4,4);
    CustomClass& ret1=hmA.findAndInsert(0,false,retFound, parm1); //Not exists, return new value.
    ASSERT_EQ(retFound,false);
    CustomClass& ret2=hmA.findAndInsert(0,true,retFound, 4 ,4); //Return replace value.
    ASSERT_EQ(retFound,true);
    ASSERT_EQ(&ret1,&ret2);
    ASSERT_EQ(ret2.a,4);
    ASSERT_EQ(ret2.b,4);
    hmA.clear();

    //Should not replace, element does not exist
    CustomClass& ret3=hmA.findAndInsert(0,false,retFound, 5 ,5);
    ASSERT_EQ(retFound,false);
    ASSERT_EQ(ret3.a,5);
    ASSERT_EQ(ret3.b,5);
    hmA.clear();

    //Should not replace, element exists
    CustomClass& ret4=hmA.findAndInsert(0,false,retFound, 6 ,6);
    CustomClass& ret5=hmA.findAndInsert(0,false,retFound, 7 ,7);
    ASSERT_EQ(ret4,ret5);
    ASSERT_EQ(ret5.a,6);
    ASSERT_EQ(ret5.b,6);
    hmA.clear();
}


TEST(Hashmap, customHeap) {
    HashMap<int, CustomClass,ObjectPoolHeap> hmA;
    bool retFound=false;


    //Should replace, element does not exist
    hmA.findAndInsert(0,true,retFound, 0 ,0);
    ASSERT_EQ(retFound,false);
    hmA.clear();

    //Should replace, element exists
    CustomClass parm1(4,4);
    CustomClass& ret1=hmA.findAndInsert(0,false,retFound, parm1); //Not exists, return new value.
    ASSERT_EQ(retFound,false);
    CustomClass& ret2=hmA.findAndInsert(0,true,retFound, 4 ,4); //Return replace value.
    ASSERT_EQ(retFound,true);
    ASSERT_EQ(&ret1,&ret2);
    ASSERT_EQ(ret2.a,4);
    ASSERT_EQ(ret2.b,4);
    hmA.clear();

    //Should not replace, element does not exist
    CustomClass& ret3=hmA.findAndInsert(0,false,retFound, 5 ,5);
    ASSERT_EQ(retFound,false);
    ASSERT_EQ(ret3.a,5);
    ASSERT_EQ(ret3.b,5);
    hmA.clear();

    //Should not replace, element exists
    CustomClass& ret4=hmA.findAndInsert(0,false,retFound, 6 ,6);
    CustomClass& ret5=hmA.findAndInsert(0,false,retFound, 7 ,7);
    ASSERT_EQ(ret4,ret5);
    ASSERT_EQ(ret5.a,6);
    ASSERT_EQ(ret5.b,6);
    hmA.clear();
}
