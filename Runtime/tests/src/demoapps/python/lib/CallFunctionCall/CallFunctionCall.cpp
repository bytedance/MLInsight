/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <CallFunctionCall.h>
#include <FuncWithDiffParms.h>
#include <TenThousandFunc.h>

extern "C" {

void callFuncA() {
    printf("Inside callFuncA\n");
    funcA();
}

void callFunc1000() {
    printf("Inside callFunc1000\n");
    func1000();
}

void callSleepDelay(long long milliseconds) {
    printf("callSleepDelay\n");
    sleepDelay(milliseconds);
}

void callLoopDelay(long long times) {
    printf("callLoopDelay\n");
    loopDelay(times);
}


}