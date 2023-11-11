/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <cstdio>
#include <FuncWithDiffParms.h>
#include <string>
#include <link.h>
#include <thread>
#include <cassert>

extern "C" {

inline int64_t getunixtimestampms() {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((int64_t) hi << 32) | lo;
}

void *callMalloc(int i) {
    return malloc(i);
}

void funcA() {
    printf("Inside Function A\n");
}


int64_t funcATimed(int& cache) {
    int64_t time1 = getunixtimestampms();
    for (int i = 0; i < 1 << 10; ++i) {
        cache += i;
    }
    return getunixtimestampms() - time1;;
}

void loopDelay(long long times) {
    printf("loopdelay\n");
    long long sum = 0;
    for (long long i = 0; i < times; ++i) {
        ++sum;
    }
}

void sleepDelay(long long millseconds) {
    printf("Sleep delay\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(millseconds));
}

void funcB(int a) {
    printf("Inside Function B\n");
}

void funcC(int a, int b) {
    printf("Inside Function C\n");
}

void funcD(int a, int b, int c) {
    printf("Inside Function D\n");
}

int funcE(int a, int b, int c) {
    printf("Inside Function E\n");
    return 1;
}
uint64_t funcTiming() {
//    return __rdtsc();
    return 0;
}

void resolveSystemFunc() {
    int success = 0;
    if (system("")) {
        ++success;
    }
    if (system("")) {
        ++success;
    }
    if (system("")) {
        ++success;
    }
}

#ifdef __AVX__
#include <immintrin.h>
void funcEverything(int e, int f, structparm s, int g, int h, long double ld,
                    double m, __m256 y, __m512 z, double n, int i, int j, int k) {
    assert(e == 565);
    assert(f == 11256);
    assert(s.a == 25);
    assert(s.d == 325823.21121251);
    assert(g == 121894);
    assert(h == 69783);
    assert(i == 245);
    assert(j == 12357);
    assert(k == 88776);
    assert(ld = 8371652.2765257);
    assert(m == 2871.2746362);
    assert(n == 271232.3782);
    __m256 y1 = _mm256_set_ps(1278611.1225, 21852.576284, 21124566.78088, 921734562.23, 0.28914970, 12.021315,
                              214.52160,
                              162.0242);
    __m512 z1 = _mm512_set_ps(224152.215680, 89794.021145, 89065436.213, 883.340, 10251.0122, 121234.025251, 14567.0567,
                              16567.0567, 2234.607, 482.03, 653.02, 879.03, 46310.07, 12342.07376, 142.021412, 16.022);

    uint8_t ret = _mm512_cmp_ps_mask(z, z1, _CMP_EQ_OS);
    assert(_mm256_cmp_ps_mask(y, y1, _CMP_EQ_OS) == 0b11111111);
    assert(_mm512_cmp_ps_mask(z, z1, _CMP_EQ_OS) == 0b1111111111111111);
}

__m256 funcRetm256() {
    __m256 ret256 = _mm256_set_ps(123152.16784534123, 3543475612.567129823, 123567234567123.12, 4323415.765234345, 1234562453.798678456, 13213265.6746523,
                                  12334556.79867856,
                                  126124934.1276567);
    return ret256;
}

__m512 funcRetm512() {
    __m512 ret512 = _mm512_set_ps(7982241.2156522580, 12356412.521612634, 3463461232.2353223, 325525676556.211321212568,
                                  99778967.56893446, 12823612.855462334,
                                  123526513.054574575767,
                                  1653242567.457457, 789789789.60788789, 789567459265.345987403, 7693277.02768673,
                                  56883745662.234465734, 237257245.23676454561, 3273443534517.23236472343,
                                  23712623235.23655657556,
                                  1235623435.23235373453423);
    return ret512;
}
#endif
long double funcRetLongDouble() {
    long double retLongD = 2242.612879712;
    return retLongD;
}

void *getFuncAddr(std::string funcName) {
    if (funcName == "funcA") {
        return (void *) funcA;
    } else if (funcName == "funcB") {
        return (void *) funcB;
    } else if (funcName == "funcC") {
        return (void *) funcC;
    } else if (funcName == "funcD") {
        return (void *) funcD;
    } else if (funcName == "funcE") {
        return (void *) funcE;
    }
#ifdef __AVX__
    else if (funcName == "funcEverything") {
        return (void *) funcEverything;
    }
#endif
    return nullptr;
}

void *findRdbg() {
    r_debug *_myRdebug = nullptr;
    const ElfW(Dyn) *dyn = _DYNAMIC;
    for (dyn = _DYNAMIC; dyn->d_tag != DT_NULL; ++dyn)
        if (dyn->d_tag == DT_DEBUG)
            _myRdebug = (struct r_debug *) dyn->d_un.d_ptr;
    return _myRdebug;
}

void *findDYNAMIC() {
    return _DYNAMIC;
}

int EXTVAR_VAR1 = 1;
static int EXTVAR_VAR2 = 2597;
}

static int callback(struct dl_phdr_info *info, size_t size, void *data) {
    int j;

    printf("name=%s (%d segments)\n", info->dlpi_name,
           info->dlpi_phnum);

    for (j = 0; j < info->dlpi_phnum; j++)
        printf("\t\t header %2d: address=%10p\n", j,
               (void *) (info->dlpi_addr + info->dlpi_phdr[j].p_vaddr));
    return 0;
}


float funcF(float a, float b, float c) {
    return a + b + c;
}

# define THREAD_SELF \
  ({ pthread_t *__self;                                                      \
     asm ("mov %%fs:%c1,%0" : "=r" (__self)                                      \
          : "i" (0x10));                       \
     __self;})

pthread_t myGetThreadID() {
    return reinterpret_cast<pthread_t>(THREAD_SELF);
}

int A::asdf = 1;


