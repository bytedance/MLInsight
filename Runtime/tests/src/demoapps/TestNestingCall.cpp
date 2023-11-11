/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <iostream>
#include <thread>
#include <hook/install.h>
#include <FuncWithDiffParms.h>
#include "installTest.h"
#include <immintrin.h>

using namespace std;

inline unsigned long long getTimeByTSC() {
#if defined(__GNUC__)
#   if defined(__i386__)
    uint64_t x;
        __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
        return x;
#   elif defined(__x86_64__)
    uint32_t hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t) lo) | ((uint64_t) hi << 32);
#   else
#       error Unsupported architecture.
#   endif
#elif defined(_MSC_VER)
    __asm {
            return __rdtsc();
        }
#else
#   error Other compilers not supported...
#endif
}


#include <x86intrin.h>

//int main() {
//    install([](std::string fileName, std::string funcName) -> bool {
//        if (funcName == "funcTiming") {
//            return true;
//        } else {
//            return false;
//        }
//    });
//
//    FILE *pFile = fopen("timing.csv", "w");
//
//    for (int i = 0; i < 10000; ++i) {
//
//        auto startCycle = __rdtsc();
//
//        funcTiming();
//
//        auto endCycle = __rdtsc();
//
//        auto timePass=endCycle-startCycle;
//
//        printf("%d,", timePass);
//    }
//
//    return 0;
//}

//int main() {
//    install([](std::string fileName, std::string funcName) -> bool {
//        if (funcName == "funcTiming") {
//            return true;
//        } else {
//            return false;
//        }
//    });
//
//    FILE *pFile = fopen("timing.csv", "w");
//
//    for (int i = 0; i < 10000; ++i) {
//
//        auto startCycle = __rdtsc();
//
//        funcTiming();
//
//        auto endCycle = __rdtsc();
//
//        auto timePass=endCycle-startCycle;
//
//        printf("%d,\n", timePass);
//        fprintf(pFile,"%d,", timePass);
//        cout<<timePass<<endl;
//    }
//
//    return 0;
//}


int main() {

    FILE *pFile = fopen("timing.csv", "w");

    for (int i = 0; i < 10000; ++i) {

        auto startCycle = __rdtsc();

        funcTiming();

        auto endCycle = __rdtsc();

        auto timePass=endCycle-startCycle;

        printf("%d,\n", timePass);
        fprintf(pFile,"%d,", timePass);
        cout<<timePass<<endl;
    }

    return 0;
}
