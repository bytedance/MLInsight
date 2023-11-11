/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <cstdio>
#include <FuncWithDiffParms.h>
#include <CallFunctionCall.h>
#include <TenThousandFunc.h>
#include <cassert>

using namespace std;

class Father {
public:
    float fValFather;
};

class Son : public Father {
public:
    float fValSon;
};

int main() {
    printf("Calling funcA\n");
    funcA();

    printf("Calling funcB\n");
    funcB(1);

    printf("Calling funcC\n");
    funcC(1, 2);

    printf("Calling funcD\n");
    funcD(1, 2, 3);

    printf("Calling funcE\n");
    funcE(1, 2, 3);

    printf("Calling func1472\n");
    func1472();

#ifdef __AVX__
    printf("Calling funcEverything\n");
    structparm strP;
    strP.a = 15;
    strP.b = 215;
    strP.d = 0.32551278;
    structparm s;
    s.a = 25;
    s.d = 325823.21121251;
    int e = 565, f = 11256, g = 121894, h = 69783, i = 245, j = 12357, k = 88776;
    long double ld = 8371652.2765257;
    double m = 2871.2746362, n = 271232.3782;
    __m256 y = _mm256_set_ps(1278611.1225, 21852.576284, 21124566.78088, 921734562.23, 0.28914970, 12.021315, 214.52160,
                             162.0242);
    __m512 z = _mm512_set_ps(224152.215680, 89794.021145, 89065436.213, 883.340, 10251.0122, 121234.025251, 14567.0567,
                             16567.0567, 2234.607, 482.03, 653.02, 879.03, 46310.07, 12342.07376, 142.021412, 16.022);
    funcEverything(e, f, s, g, h, ld, m, y, z, n, i, j, k);
#endif
    float srcFloat = 1.75;
    Son son;
    son.fValFather = 3.141;
    son.fValSon = 6.283;

    Father *fatherPtr = dynamic_cast<Father *>(&son);
    assert(fatherPtr->fValFather - 3.141 < 1e-5);
    assert(son.fValFather - 3.141 < 1e-5);
    assert(son.fValSon - 6.283 < 1e-5);

#ifdef __AVX__
    printf("Calling funcRetm256();\n");
    __m256 rlt256_real = _mm256_set_ps(123152.16784534123, 3543475612.567129823, 123567234567123.12, 4323415.765234345,
                                       1234562453.798678456, 13213265.6746523,
                                       12334556.79867856,
                                       126124934.1276567);
    __m256 rlt256 = funcRetm256();
    assert(_mm256_cmp_ps_mask(rlt256, rlt256_real, _CMP_EQ_OS) == 0b11111111);

    printf("Calling funcRetm512();\n");
    __m512 rlt512_real = _mm512_set_ps(7982241.2156522580, 12356412.521612634, 3463461232.2353223,
                                       325525676556.211321212568,
                                       99778967.56893446, 12823612.855462334,
                                       123526513.054574575767,
                                       1653242567.457457, 789789789.60788789, 789567459265.345987403, 7693277.02768673,
                                       56883745662.234465734, 237257245.23676454561, 3273443534517.23236472343,
                                       23712623235.23655657556,
                                       1235623435.23235373453423);
    __m512 rlt512 = funcRetm512();
    assert(_mm512_cmp_ps_mask(rlt512, rlt512_real, _CMP_EQ_OS) == 0b1111111111111111);
#endif

    printf("Calling callFuncA\n");
    callFuncA();

    printf("Calling callFunc1000\n");
    callFunc1000();

    return 0;
}