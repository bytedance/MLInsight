#include <gtest/gtest.h>
#include "common/LazyValueType.h"

using namespace mlinsight;
TEST(LazyValueType,constructDeconstruct){
    LazyConstructValue<int> a;
    ASSERT_FALSE(a.isValueConstructed());
    a.constructValue(1);
    ASSERT_TRUE(a.isValueConstructed());
}
TEST(LazyValueType,equalComapre){
    LazyConstructValue<int> a;
    a.constructValue(1);
    LazyConstructValue<int> b;
    b.constructValue(1);
    LazyConstructValue<int> c;
    LazyConstructValue<int> d;
    ASSERT_TRUE(a==b);
    ASSERT_FALSE(a!=b);
    ASSERT_TRUE(b!=c);
    ASSERT_TRUE(c==d);
}


TEST(LazyValueType,copyconstructor){
    LazyConstructValue<int>* a=new LazyConstructValue<int>();
    a->constructValue(1);
    LazyConstructValue<int> b(*a);
    delete a;
    ASSERT_EQ(b.getConstructedValue(),1);
}

TEST(LazyValueType,copyoperator){
    LazyConstructValue<int>* a=new LazyConstructValue<int>();
    a->constructValue(1);
    LazyConstructValue<int> b;
    b=*a;
    delete a;
    ASSERT_EQ(b.getConstructedValue(),1);
}

TEST(LazyValueType,moveconstructor){
    LazyConstructValue<int>* a=new LazyConstructValue<int>();
    a->constructValue(1);
    LazyConstructValue<int> b(std::move(*a));
    delete a;
    ASSERT_EQ(b.getConstructedValue(),1);
}

TEST(LazyValueType,moveassignment){
    LazyConstructValue<int>* a=new LazyConstructValue<int>();
    a->constructValue(1);
    LazyConstructValue<int> b;
    b=std::move(*a);
    delete a;
    ASSERT_EQ(b.getConstructedValue(),1);
}