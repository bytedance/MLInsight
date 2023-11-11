/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <iostream>
#include <atomic>
#include <cmath>

std::atomic<int> foobar(8);

inline int64_t getunixtimestampms() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((int64_t) hi << 32) | lo;
}

int main() {
    int *array = (int *) malloc(sizeof(int) * 4);
    uint64_t start = getunixtimestampms();
    for (int i = 0; i < pow(10,5); ++i) {
        __asm__ (
        "movq %%r11,(%0)"
        ://Output
        : "r" (array)//Input
        : "r11"//Clobbers
        );
    }
    uint64_t stop = getunixtimestampms();

    uint64_t duration = stop - start;
    printf("Mem write %lu\n", duration);



    start = getunixtimestampms();
    for (int i = 0; i < pow(10,5); ++i) {
        __asm__ (
        "movq (%0),%%r11"
        ://Output
        : "r" (array)//Input
        : "r11"//Clobbers
        );
    }
    stop = getunixtimestampms();

    duration = stop - start;
    printf("Mem read %lu\n", duration);


    start = getunixtimestampms();
    for (int i = 0; i < pow(10,5); ++i) {
        __asm__ (
        "addq $0x1,(%0)\n\t"
        ://Output
        : "r" (array)//Input
        : "r11"//Clobbers
        );
    }
    stop = getunixtimestampms();

    duration = stop - start;
    printf("Mem add %lu\n", duration);


    start = getunixtimestampms();
    for (int i = 0; i < pow(10,5); ++i) {
        __asm__ (
        "lock addq $0x1,(%0)\n\t"
        ://Output
        : "r" (array)//Input
        : "r11"//Clobbers
        );
    }
    stop = getunixtimestampms();

    duration = stop - start;
    printf("Mem atomic add %lu\n", duration);
    return 0;
}
