/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef LISTTEST_H
#define LISTTEST_H

#include "immintrin.h"
#include <string>


extern "C" {
void *callMalloc(int i);

void funcA();

inline int64_t getunixtimestampms() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((int64_t) hi << 32) | lo;
}

int64_t funcATimed(int &cache);

void funcB(int a);

void funcC(int a, int b);

void funcD(int a, int b, int c);

int funcE(int a, int b, int c);

float funcF(float a, float b, float c);

void loopDelay(long long times);

void sleepDelay(long long seconds);

uint64_t funcTiming();

void resolveSystemFunc();

pthread_t myGetThreadID();
typedef struct {
    int a, b;
    double d;
} structparm;


void *getFuncAddr(std::string funcName);

void *findRdbg();

void *findDYNAMIC();

#ifdef __AVX__


void funcEverything(int e, int f,
                    structparm s, int g, int h,
                    long double ld, double m,
                    __m256 y,
                    __m512 z,
                    double n, int i, int j, int k);
__m256 funcRetm256();

__m512 funcRetm512();
#endif
class A {
public:
    static int asdf;
};


}
#endif
