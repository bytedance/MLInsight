#include <gtest/gtest.h>
#include "analyse/FlameGraph.h"

using namespace mlinsight;

const int A=0;
const int B=1;
const int C=2;
const int D=3;
const int E=4;

//void initPyModuleInfo(){
//    //0
//    hookInstallerInstance->pyModuleInfoMap.pushBack("A", USER_DONT_CARE, -1);
//    //1
//    hookInstallerInstance->pyModuleInfoMap.pushBack("B", USER_DONT_CARE, -1);
//    //2
//    hookInstallerInstance->pyModuleInfoMap.pushBack("C", USER_DONT_CARE, -1);
//    //3
//    hookInstallerInstance->pyModuleInfoMap.pushBack("D", USER_DONT_CARE, -1);
//    //4
//    hookInstallerInstance->pyModuleInfoMap.pushBack("E", USER_DONT_CARE, -1);
//
//}


void initCallStackaAB(CallStackOld<PyCallStack, 20>& exampleCallStack){
    exampleCallStack.levels=2;
    exampleCallStack.array[0].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[0].cachedCodeExtra->globalPyModuleId=B;

    exampleCallStack.array[1].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[1].cachedCodeExtra->globalPyModuleId=A;
}
void initCallStackaABC(CallStackOld<PyCallStack, 20>& exampleCallStack){
    exampleCallStack.levels=3;
    exampleCallStack.array[0].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[0].cachedCodeExtra->globalPyModuleId=C;

    exampleCallStack.array[1].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[1].cachedCodeExtra->globalPyModuleId=B;


    exampleCallStack.array[2].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[2].cachedCodeExtra->globalPyModuleId=A;
}

void initCallStackaAEB(CallStackOld<PyCallStack, 20>& exampleCallStack){
    exampleCallStack.levels=3;
    exampleCallStack.array[0].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[0].cachedCodeExtra->globalPyModuleId=B;

    exampleCallStack.array[1].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[1].cachedCodeExtra->globalPyModuleId=E;

    exampleCallStack.array[2].cachedCodeExtra = new PyCodeExtra();
    exampleCallStack.array[2].cachedCodeExtra->globalPyModuleId=A;
}


TEST(FlameGraph, flameTreeInsertionAndRemoval) {

    FlameTree<FTMemCell> flameTree;

    CallStackOld<PyCallStack, 20> ABC;
    initCallStackaABC(ABC);
    CallStackOld<PyCallStack, 20> AEB;
    initCallStackaAEB(AEB);

    const FTMemCell *abcCell= flameTree.addData(ABC, 2);
    const FTMemCell *aebCell= flameTree.addData(AEB,4);

    ASSERT_EQ(flameTree.snapShotArray[0]->children.size(),1);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[B]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[B]->children[C]->allocationSize,2);

    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[E]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[E]->children[B]->allocationSize,4);

    flameTree.rmData(abcCell, ABC, 2);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[B]->children[C]->allocationSize,0);

    flameTree.rmData(aebCell, AEB, 4);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[E]->children[B]->allocationSize,0);
}


TEST(FlameGraph, flameTreeSnapshot) {

    FlameTree<FTMemCell> flameTree;

    CallStackOld<PyCallStack, 20> ABC;
    initCallStackaABC(ABC);
    CallStackOld<PyCallStack, 20> AEB;
    initCallStackaAEB(AEB);

    const FTMemCell *abcCell= flameTree.addData(ABC, 2);
    ASSERT_EQ(flameTree.snapshot(),0);
    const FTMemCell *aebCell= flameTree.addData(AEB,4);
    ASSERT_EQ(flameTree.snapshot(),1);
    flameTree.rmData(abcCell, ABC, 2);
    ASSERT_EQ(flameTree.snapshot(),2);
    flameTree.rmData(aebCell, AEB, 4);
    ASSERT_EQ(flameTree.snapshot(),3);

    ASSERT_EQ(flameTree.snapShotArray[0]->children.size(),1);
    //At this them there is only one allocation
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children.size(),1);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[B]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[B]->children[C]->allocationSize,2);

    //At this time there are two allocations
    ASSERT_EQ(flameTree.snapShotArray[1]->children[A]->children.size(),2);
    ASSERT_EQ(flameTree.snapShotArray[1]->children[A]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[1]->children[A]->children[E]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[1]->children[A]->children[E]->children[B]->allocationSize,4);

    //At this time, there are two allocations and one deallocation
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children.size(),2);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children[B]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children[B]->children[C]->allocationSize,0);

    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children[E]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children[E]->children[B]->allocationSize,4);

    //At this time, there are two allocations and two deallocations
    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->children[B]->children[C]->allocationSize,0);
    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->children[E]->children[B]->allocationSize,0);

    //Check copy on write is correct or not. Nodes should only be copied if it is on the callstack.
    //AEB is the last removed so it should has snapshotId 3.
    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->snapshotId,3);
    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->children[E]->snapshotId,3);
    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->children[E]->children[B]->snapshotId,3);

    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->children[B]->snapshotId,2);
    ASSERT_EQ(flameTree.snapShotArray[3]->children[A]->children[B]->children[C]->snapshotId,2);
}



TEST(FlameGraph, flameTreeSnapshotShortcut) {

    FlameTree<FTMemCell> flameTree;

    CallStackOld<PyCallStack, 20> ABC;
    initCallStackaABC(ABC);
    CallStackOld<PyCallStack, 20> AB;
    initCallStackaAEB(AB);

    const FTMemCell *oldAbcCell= flameTree.addData(ABC, 2);
    ASSERT_EQ(flameTree.snapshot(),0);
    const FTMemCell *newAbcCell= flameTree.addData(ABC, 2);
    ASSERT_EQ(flameTree.snapshot(),1);
    ASSERT_EQ(flameTree.snapShotArray[0]->children[A]->children[B]->children[C]->allocationSize,2);
    ASSERT_EQ(flameTree.snapShotArray[1]->children[A]->children[B]->children[C]->allocationSize,4);

    //Check whether oldAbcCell points to the newAbcCell.
    ASSERT_EQ(oldAbcCell->snapshotId,0);
    ASSERT_EQ(oldAbcCell->nextSnapshotNode,newAbcCell);

    //Invoke remove on oldAbcCell
    const FTMemCell* tmp1=oldAbcCell;
    flameTree.rmData(tmp1, ABC, 2);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children[B]->children[C]->allocationSize,2);
    const FTMemCell* tmp2=oldAbcCell;
    flameTree.rmData(tmp2, ABC, 2);
    ASSERT_EQ(flameTree.snapShotArray[2]->children[A]->children[B]->children[C]->allocationSize,0);

}

