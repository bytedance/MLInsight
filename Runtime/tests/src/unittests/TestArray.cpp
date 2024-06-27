/*
This file is part of Scaler.
Scaler is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Scaler is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <gtest/gtest.h>
#include "common/Array.h"

using namespace mlinsight;

TEST(Array, isEmpty) {
    using Array_t=Array<int>;

    Array_t myList;
    ASSERT_TRUE(myList.isEmpty());
    ASSERT_EQ(myList.getSize(), 0);

    myList.insert(myList.getSize())=1;
    ASSERT_EQ(myList.getSize(), 1);
    ASSERT_FALSE(myList.isEmpty());
}

TEST(Array, insert) {
    using Array_t=Array<int>;

    Array_t myArray;

    //Insert at top
    for (int i = 0; i <= 4; ++i){
        myArray.pushBack(i);
    }
    ASSERT_EQ(myArray.getSize(), 5);

    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myArray[i], i);
    }

    //Insert in the middle
    myArray.insert(3, 3);
    ASSERT_EQ(myArray.getSize(), 6);

    for (int i = 0; i <= 3; ++i) {
        ASSERT_EQ(myArray.get(i), i);
    }
    ASSERT_EQ(myArray.get(4), 3);
    ASSERT_EQ(myArray.get(5), 4);
}

TEST(Array, insertBack) {
    Array<int> myList;
    //Pushback
    for (int i = 0; i <= 4; ++i){
        myList.pushBack(i);
    }

    for (int i = 0; i <= 4; ++i){
        ASSERT_EQ(myList[i],i);
    }
}

TEST(Array, erase) {
    Array<int> myArray;

    //Erase non-existing element
    ASSERT_DEATH(myArray.erase(0), ".*0 <= index \\&\\& index < size.*");

    //Erase one element
    myArray.pushBack(0);
    myArray.erase(0);
    ASSERT_TRUE(myArray.isEmpty());

    //Erase multiple lements
    for (int i = 0; i <= 4; ++i){
        myArray.pushBack(i);
    }
    for (int i = 0; i <= 2; ++i) {
        myArray.erase(0);
    }
    ASSERT_EQ(myArray.getSize(), 2);
    ASSERT_EQ(3, myArray[0]);
    ASSERT_EQ(4, myArray[1]);
}

TEST(Array, copyconstructor) {
    Array<int> myArray;
    for (int i = 0; i <= 4; ++i) {
        myArray.pushBack(i);
    }
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myArray[i], i);
    }

    //Test copy constructor
    Array<int> myList1(myArray);
    ASSERT_TRUE(myList1.getSize() == myArray.getSize());
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myList1[i], i);
    }

    //Modify copied list
    myList1.erase(0);
    ASSERT_TRUE(myList1.getSize() == myArray.getSize() - 1);

    for (int i = 0; i <= 3; ++i) {
        myList1[i]=i*10;
    }

    //Double check to see if myArray remains the same.
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myArray[i], i);
    }
}

TEST(Array, copyassignment) {
    Array<int> myArray;
    for (int i = 0; i <= 4; ++i) {
        myArray.pushBack(i);
    }
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myArray[i], i);
    }

    //Test copy constructor
    Array<int> myArrayCpy;
    myArrayCpy = myArray;
    ASSERT_TRUE(myArrayCpy.getSize() == myArray.getSize());
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myArrayCpy[i], i);
    }

    //Modify copied list
    myArrayCpy.erase(0);
    ASSERT_TRUE(myArrayCpy.getSize() == myArray.getSize() - 1);

    for (int i = 0; i <= 3; ++i) {
        myArrayCpy[i]= i * 10;
    }

    //Double check to see if myArray remains the same.
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myArray[i], i);
    }

    //Test self-copy
    myArray=myArray;
}

TEST(Array, moveconstructor) {
    Array<int>* myArray=new Array<int>();
    for (int i = 0; i <= 4; ++i)
        myArray->pushBack(i);

    //Test copy constructor
    Array<int> myArrayCpy(std::move(*myArray));
    delete myArray;
    for (int i = 0; i <= 4; ++i)
        ASSERT_EQ(myArrayCpy[i],i);
}

TEST(Array, moveassignment) {
    Array<int>* myArray=new Array<int>();
    for (int i = 0; i <= 4; ++i)
        myArray->pushBack(i);

    //Test copy constructor
    Array<int> myArrayCpy;
    myArrayCpy=std::move(*myArray);
    delete myArray;
    for (int i = 0; i <= 4; ++i)
        ASSERT_EQ(myArrayCpy[i],i);
}

TEST(Array, clear) {
    Array<int> myArray;
    for (int i = 0; i <= 4; ++i){
        myArray.pushBack(i);
    }
    myArray.clear();
}

class CustomClass{
public:
    int a;
    int b;
    CustomClass(int a,int b):a(a),b(b){
    }
};

TEST(Array, customClassOperations) {

    Array<CustomClass> myList;
    myList=Array<CustomClass>();
    //Pushback using four different ways
    myList.pushBack(1,1);
    myList.pushBack(CustomClass(2,2));
    CustomClass temp3(3,3);
    myList.pushBack(temp3);
    CustomClass temp4(4,4);
    myList.pushBack(std::move(temp4));
    CustomClass* rawMemory=myList.pushBackRaw();
    new (rawMemory) CustomClass(5,5);

    ASSERT_EQ(myList.getSize(),5);
    for (int i = 0; i <= 4; ++i) {
        ASSERT_EQ(myList[i].a, i+1);
        ASSERT_EQ(myList[i].b, i+1);
    }
}

