/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef CALL_FUNCTION_CALL_H
#define CALL_FUNCTION_CALL_H

#include "immintrin.h"
#include <string>

extern "C" {

void callFuncA();

void callFunc1000();

void callSleepDelay(long long milliseconds);

void callLoopDelay(long long times);



}
#endif
