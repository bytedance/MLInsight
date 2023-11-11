/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <iostream>
#include <installTest.h>
#include <FuncWithDiffParms.h>
#include <cassert>
#include <hook/install.h>
#include "common/ProcInfoParser.h"

using namespace std;

int main() {
    mlinsight::PmParser_Linux pmP;
    std::string curFileName = pmP.curExecAbsolutePath;
    install([](std::string fileName, std::string funcName) -> bool {
        if (funcName.find("funcA") != std::string::npos) {
            return true;
        } else {
            return false;
        }
    });

    printf("Calling funcA\n");
    funcA();

//    structparm s;
//    int e, f, g, h, i, j, k;
//    long double ld;
//    double m, n;
//    __m256 y;
//    __m512 z;
//    funcEverything(e, f, s, g, h, ld, m, y, z, n, i, j, k);
//
//    float floatRlt=funcF(1.0, 2.0, 3.0);
//    printf("%f\n",floatRlt);
//
//    callFuncA();
    return 0;
}

