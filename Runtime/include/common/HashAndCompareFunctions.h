#ifndef DOUBLETAKE_HASHFUNCS_H
#define DOUBLETAKE_HASHFUNCS_H

/*
 * @file   hashfuncs.h
 * @brief  Some functions related to hash table.
 * @author Steven Tang <steven.tang@bytedance.com>
 * @author Tongping Liu <http://www.cs.umass.edu/~tonyliu>
 */
#include <stdint.h>
#include <string.h>
#include <stdint.h>

namespace mlinsight {
    /**
     * Default hash function that passes getKey without processing.
     * This only works for hashmap that use numerical value as getKey.
     */
    template<typename KEY_TYPE>
    size_t hash(const KEY_TYPE &key) {
        return (size_t) (key);
    }

    /**
     * Hash any address
     * @tparam KEY_TYPE
     * @param addr
     * @return
     */
    template<typename KEY_TYPE>
    inline size_t hash(const KEY_TYPE* addr) {
        size_t key = (size_t) addr;
        key ^= (key << 15) ^ 0xcd7dcd7d;
        key ^= (key >> 10);
        key ^= (key << 3);
        key ^= (key >> 6);
        key ^= (key << 2) + (key << 14);
        key ^= (key >> 16);
        return key;
    }

    struct CallStackAddress {
        uint64_t callStackAddress;

        CallStackAddress(uint64_t address) : callStackAddress(address) {
        };
    };

    /*
     * Updated hash function; peak performance seems to occur at >= 8 bits; greater than
     * 10 bits seems to plataue. 48 bits (as a test, of course) is worse than no bits.
     */
    template<>
    inline size_t hash(const CallStackAddress &x) {
        size_t key = x.callStackAddress;
        return key >> 10;
    }

    //template<>
    //inline size_t hash(const std::string& x) {
    //    size_t __h = 0;
    //    const char* __s = x.c_str();
    //    ssize_t len=x.length();
    //    for(ssize_t i =0; i <= (int) len; i++, ++__s)
    //        __h = 5 * __h + *__s;
    //    return __h;
    //}

    template<>
    inline size_t hash(const std::string &key) {
        static std::hash<std::string> stdStringHash;
        //todo: size_t to ssize_t! In hashmap use size_t should be fine
        size_t currentHash = (size_t) (stdStringHash(key));
        return currentHash;
    }


    template<typename KEY_TYPE>
    char compareWithEqual(const KEY_TYPE &src, const KEY_TYPE &dst) {
        if (src == dst) {
            return 0;
        } else if (src > dst) {
            return 1;
        } else {
            return -1;
        }
    }

    /**
   * Default comparator for hashmap by using operator==
   * @tparam KEY_TYPE The type of hashmap getKey
   * @return 0 if equals, 1 if src>dst, -1 if src<dst
   */
    template<typename KEY_TYPE>
    bool compare(const KEY_TYPE &src, const KEY_TYPE &dst) {
        if (src == dst) {
            return true;
        } else {
            return false;
        }
    }

    template<>
    inline char compareWithEqual(const std::string &src, const std::string &dst) {
        auto srcLength = src.length();
        auto dstLength = dst.length();
        if (srcLength == dstLength && src == dst) {
            return 0;
        } else if (srcLength > dstLength && src > dst) {
            return 1;
        } else {
            return -1;
        }
    }
}
#endif
