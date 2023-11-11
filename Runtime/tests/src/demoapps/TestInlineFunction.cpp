/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <cstdio>
#include <thread>
//=========================================================================
// Thread Functions
//=========================================================================

inline int funcInline(int &num) {
    std::this_thread::sleep_for (std::chrono::milliseconds(1));
    return num += 15;
}

void funcNormal(int num) {
    num+=1;
    funcInline(num);
    std::this_thread::sleep_for (std::chrono::milliseconds(1));
}

int main() {

    for (int i = 0; i < 999; ++i) {
        if (i % 10) {
            funcNormal(i);
        }
    }
    return 0;
}

