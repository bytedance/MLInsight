/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <CallFunctionCall.h>
#include <signal.h>

using namespace std;

volatile int DEBUGGER_WAIT = 1;

void test_continue() {
    DEBUGGER_WAIT = 0;
}

int main() {
    callFuncA();
    return 0;
}