/*
This file is part of Scaler.
Scaler is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Scaler is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <gtest/gtest.h>
#include "common/LinkedList.h"

using namespace mlinsight;


TEST(LinkedList, isEmpty) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;
    
    LinkedList_t myList;
    ASSERT_TRUE(myList.isEmpty());
    ASSERT_TRUE(myList.head->next == myList.tail);
    ASSERT_TRUE(myList.head->next->prev == myList.head);
    ASSERT_TRUE(myList.tail->prev == myList.head);
    ASSERT_TRUE(myList.tail->prev->next == myList.tail);
    ASSERT_EQ(myList.getSize(), 0);

    myList.insertAfter(myList.getHead(), 0);
    ASSERT_EQ(myList.getSize(), 1);
    ASSERT_FALSE(myList.isEmpty());
    ASSERT_TRUE(myList.head->next->prev == myList.head);
    ASSERT_TRUE(myList.head->next->next == myList.tail);
}

TEST(LinkedList, insertAfter) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;
    
    LinkedList_t myList;
    //Insert after nullptr
    ASSERT_DEATH(myList.insertAfter(nullptr, 0), ".*node != nullptr.*");

    //Insert after tail
    ASSERT_DEATH(myList.insertAfter(myList.getTail(), 0), ".*node != tail.*");

    //Insert at top
    for (int i = 1; i <= 5; ++i){
        ASSERT_EQ(myList.insertAfter(myList.getHead(), i),i);
    }
    ASSERT_EQ(myList.getSize(), 5);

    ListEntry_t *curEntry = myList.getHead()->getNext();
    for (int i = 5; i >= 1; --i) {
        ASSERT_EQ(curEntry->getValue(), i);
        curEntry = curEntry->getNext();
    }

    //Insert in the middle
    myList.insertAfter(curEntry->getPrev()->getPrev()->getPrev()->getPrev()->getPrev()->getPrev(), 0);
    ASSERT_EQ(myList.getSize(),6);

    curEntry = myList.getHead();
    for (int i = 6; i <= 0; ++i) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(i, curEntry->getValue());
    }
}



TEST(LinkedList, insertBack) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;
    LinkedList<int> myList;

    myList=LinkedList<int>();
    //Pushback
    for (int i = 1; i <= 5; ++i)
        myList.insertBack(i);

    ListEntry_t *curEntry = myList.getHead()->getNext();
    ASSERT_EQ(myList.getSize(),5);
    for (int i = 1; i <= 5; ++i) {
        ASSERT_EQ(curEntry->getValue(), i);
        curEntry = curEntry->getNext();
    }
    ASSERT_EQ(curEntry, myList.getTail());
}

TEST(LinkedList, erase) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;
    LinkedList<int> myList;

    //Erase nullptr
    ASSERT_DEATH(myList.erase(nullptr), ".*node != nullptr.*");
    //Erase head
    ASSERT_DEATH(myList.erase(myList.getHead()), ".*node != head.*");
    //Erase tail
    ASSERT_DEATH(myList.erase(myList.getTail()), ".*node != tail.*");

    //Erase one element
    myList.insertBack(0);
    myList.erase(myList.getHead()->getNext());
    ASSERT_TRUE(myList.isEmpty());

    //Erase multiple lements
    for (int i = 1; i <= 5; ++i)
        myList.insertBack(i);
    ListEntry_t *curEntry = myList.getHead()->getNext();
    for (int i = 1; i <= 5; ++i) {
        auto nextPtr = curEntry->getNext();
        if (i % 2 == 0)
            myList.erase(curEntry);
        curEntry = nextPtr;
    }
    ASSERT_EQ(myList.getSize(), 3);
    curEntry = myList.getHead();
    for (int i = 1; i <= 5; i += 2) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(i, curEntry->getValue());
    }

}

TEST(LinkedList, copyconstructor) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;
    LinkedList_t myList;
    for (int i = 1; i <= 5; ++i)
        myList.insertBack(i);

    //Test copy constructor
    LinkedList<int> myList1(myList);
    ASSERT_TRUE(myList1.getSize()==myList.getSize());

    //Modify copied list
    myList1.erase(myList1.getHead()->getNext());
    ASSERT_TRUE(myList1.getSize()==myList.getSize()-1);
    ListEntry_t *curEntry = myList1.getHead();
    for (int i = 2; i <= 5; ++i) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(curEntry->getValue(), i);
        //Modify value to test whether this modification changes the original array.
        curEntry->getValue()= curEntry->getValue() * 10;
    }

    //Double check to see if myList remains the same.
    curEntry = myList.getHead();
    for (int i = 1; i <= 5; ++i) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(curEntry->getValue(), i);
    }
    ASSERT_TRUE(curEntry->getNext()==myList.getTail());
}

TEST(LinkedList, copyassignment) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;

    LinkedList_t* myList=new LinkedList_t();
    for (int i = 1; i <= 5; ++i)
        myList->insertBack(i);
    //Test copy constructor
    LinkedList_t myList1;
    myList1=*myList;
    ASSERT_TRUE(myList1.getSize()==myList->getSize());

    //Manually trigger deconstruction of moved object.
    delete myList;
    myList=nullptr;

    //Modify copied list
    myList1.erase(myList1.getHead()->getNext());
    ListEntry_t *curEntry = myList1.getHead();
    for (int i = 2; i <= 5; ++i) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(curEntry->getValue(), i);
        //Modify value to test whether this modification changes the original array.
        curEntry->getValue()= curEntry->getValue() * 10;
    }

    //copy self
    LinkedList_t& myListSelfCopy=myList1;
    myListSelfCopy = myList1;
    ASSERT_EQ(myList1.getHead(), myListSelfCopy.getHead());
    ASSERT_EQ(myList1.getTail(), myListSelfCopy.getTail());
    curEntry = myListSelfCopy.getHead();
    for (int i = 2; i <= 5; ++i) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(curEntry->getValue(), i * 10);
    }
}

TEST(LinkedList, moveconstructor) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;

    LinkedList_t* myList=new LinkedList_t();
    for (int i = 1; i <= 5; ++i)
        myList->insertBack(i);

    //Test copy constructor
    LinkedList_t myList1(std::move(*myList));
    //Manually trigger deconstruction of moved object, since object have been moved, the deconstructor will not free memory.
    delete myList;
    myList=nullptr;

    //Modify moved list
    myList1.erase(myList1.getHead()->getNext());
    ListEntry_t *curEntry = myList1.getHead();
    for (int i = 2; i <= 5; ++i) {
        curEntry = curEntry->getNext();
        ASSERT_EQ(curEntry->getValue(), i);
        //Modify value to test whether this modification changes the original array.
        curEntry->getValue()= curEntry->getValue() * 10;
    }
}
TEST(LinkedList, clear) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;
    LinkedList_t myList;
    for (int i = 1; i <= 5; ++i)
        myList.insertBack(i);
    myList.clear();
    ASSERT_EQ(myList.getSize(),0);
    ASSERT_TRUE(myList.isEmpty());
    //Clear again, should have no problem.
    myList.clear();
    ASSERT_EQ(myList.getSize(),0);
    ASSERT_TRUE(myList.isEmpty());
}

TEST(LinkedList, Iteration) {
    using LinkedList_t=LinkedList<int,PassThroughMemoryHeap>;
    using ListEntry_t=ListEntry<int,PassThroughMemoryHeap>;
    using ListIterator_t=ListIterator<int,PassThroughMemoryHeap>;

    LinkedList_t myList;
    ASSERT_TRUE(myList.isEmpty());
    auto beg = myList.begin();
    auto end = myList.end();
    ASSERT_TRUE(beg == beg);
    ASSERT_TRUE(end == end);
    ASSERT_TRUE(beg == end);

    //Populate some data
    for (int i = 1; i <= 7; ++i)
        myList.insertBack(i);

    ASSERT_TRUE(myList.begin() != myList.end());

    int index = 1;
    for (auto elem = myList.begin(); elem != myList.end(); ++elem) {
        ASSERT_EQ(*elem, index++);
    }
    index = 1;

    ListEntry_t *curEntry = myList.getHead()->getNext();
    for (int i = 1; i <= 7; ++i) {
        auto nextPtr = curEntry->getNext();
        if (curEntry->getValue() % 2 == 0) {
            myList.erase(curEntry);
        }
        curEntry = nextPtr;
    }
    ASSERT_EQ(myList.getSize(), 4);
    auto begin = myList.rbegin();
    ASSERT_EQ(*begin, 7);
    EXPECT_FALSE(begin == myList.rend());
    --begin;
    ASSERT_EQ(*begin, 5);
    EXPECT_FALSE(begin == myList.rend());
    begin--;
    ASSERT_EQ(*begin, 3);
    EXPECT_FALSE(begin == myList.rend());
    begin--;
    ASSERT_EQ(*begin, 1);
    --begin;
    EXPECT_TRUE(begin == myList.rend());

    myList.clear();
    for (int i = 1; i <= 7; ++i)
        myList.insertBack(i);
    //Check whether iterator still works after copy
    LinkedList_t* myListCpy = new LinkedList_t();
    *myListCpy=myList;
    for (auto elem = myListCpy->begin(); elem != myListCpy->end(); ++elem) {
        ASSERT_EQ(*elem, index++);
    }
    //Check whether iterator still works after move
    LinkedList_t myListMove=std::move(*myListCpy);
    delete myListCpy;
    for (auto elem = myListMove.begin(); elem != myListMove.end(); ++elem) {
        ASSERT_EQ(*elem, index++);
    }

}

//class CustomClass{
//public:
//    int a;
//    int b;
//    CustomClass(int a,int b):a(a),b(b){
//    }
//};
//
//TEST(LinkedList, customClassOperations) {
//    using LinkedList_t=LinkedList<CustomClass,PassThroughMemoryHeap>;
//    using ListEntry_t=ListEntry<CustomClass,PassThroughMemoryHeap>;
//    using ListIterator_t=ListIterator<CustomClass,PassThroughMemoryHeap>;
//
//    LinkedList<CustomClass> myList;
//    myList=LinkedList<CustomClass>();
//    //Pushback using four different ways
//    myList.insertBack(1,1);
//    myList.insertBack(CustomClass(2,2));
//    CustomClass temp3(3,3);
//    myList.insertBack(temp3);
//    CustomClass temp4(4,4);
//    myList.insertBack(std::move(temp4));
//
//
//    ListEntry_t *curEntry = myList.getHead()->getNext();
//    ASSERT_EQ(myList.getSize(),4);
//    for (int i = 1; i <= 4; ++i) {
//        ASSERT_EQ(curEntry->getValue().a, i);
//        ASSERT_EQ(curEntry->getValue().b, i);
//        curEntry = curEntry->getNext();
//    }
//
//
//    ASSERT_EQ(curEntry, myList.getTail());
//
//}
//
//TEST(LinkedList, customHeap) {
//    using LinkedList_t=LinkedList<int,ObjectPoolHeap>;
//    using ListEntry_t=ListEntry<int,ObjectPoolHeap>;
//    using ListIterator_t=ListIterator<int,ObjectPoolHeap>;
//
//    LinkedList_t myList;
//    //Insert after nullptr
//    ASSERT_DEATH(myList.insertAfter(nullptr, 0), ".*node != nullptr.*");
//
//    //Insert after tail
//    ASSERT_DEATH(myList.insertAfter(myList.getTail(), 0), ".*node != tail.*");
//
//    //Insert at top
//    for (int i = 1; i <= 5; ++i)
//        myList.insertAfter(myList.getHead(), i);
//    ASSERT_EQ(myList.getSize(), 5);
//    myList.erase(myList.getHead()->next);
//
//    ListEntry_t *curEntry = myList.getHead()->getNext();
//    for (int i = 5; i >= 1; --i) {
//        ASSERT_EQ(curEntry->getValue(), i);
//        curEntry = curEntry->getNext();
//    }
//
//    //Insert in the middle
//    myList.insertAfter(curEntry->getPrev()->getPrev()->getPrev()->getPrev()->getPrev()->getPrev(), 0);
//    ASSERT_EQ(myList.getSize(),6);
//
//    curEntry = myList.getHead();
//    for (int i = 6; i <= 0; ++i) {
//        curEntry = curEntry->getNext();
//        ASSERT_EQ(i, curEntry->getValue());
//    }
//}