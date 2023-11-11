/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/

#ifndef MLINSIGHT_MATH_H
#define MLINSIGHT_MATH_H

#include <ctype.h>

namespace mlinsight {
    inline void calcMean(const int64_t &x, int64_t &counter, double &mean) {
        counter++;
        double delta = x - mean;
        mean += delta / counter;
        //var += (double) (n - 1) / n * delta * delta;
    }

    template<typename T>
    inline const T &max(const T &a, const T &b) {
        return a > b ? a : b;
    }

    template<typename T>
    inline const T &min(const T &a, const T &b) {
        return a > b ? b : a;
    }
}
#endif //MLINSIGHT_MATH_H
