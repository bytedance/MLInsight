/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_TOOL_H
#define MLINSIGHT_TOOL_H


#include <inttypes.h>
#include <string>
#include <immintrin.h>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <sstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>
#include <elf.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "Logging.h"
#include "common/CallStack.h"

namespace mlinsight {
    #define CPP_CALL_STACK_LEVEL 20
    #define PYTHON_CALL_STACK_LEVEL 20

    inline int64_t getunixtimestampms() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((int64_t) hi << 32) | lo;
    }

    /**
     * Get the allocatedSize of a file
     */
    long int getFileSize(FILE *file);

    /**
     * Find the split indexes of string srcStr separated by splitChar
     * [ret[2i],ret[2i+1]) marks the starting and ending indexes of segment i.
     * Notice the right bound is NOT inclusive, meaning the length of this string segment is ret[2i+1]-ret[2i]
     *
     * Repeated splitChar is treated as a single character.
     *
     * eg: Input ""
     *
     * @return An array of paired indexes.
     */
    std::vector<ssize_t> findStrSplit(std::string &srcStr, char splitChar);

    bool extractFileName(std::string absolutePath, std::string &pathName, std::string &fileName);

    bool getPWD(std::string &retPwdPath);

    /**
     * Open a new file and write to it
     * @param fd Returned file descriptor
     * @param retMemAddr
     * @return Success or not
     */
    template<typename T>
    bool fOpen4Write(const char *fileName, int &fd, size_t fileSizeInBytes, T *&retMemAddr){
        fd = open(fileName, O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0644);
        if (fd == -1) {
            ERR_LOGS("Cannot open %s because:%s", fileName, strerror(errno));
            return false;
        }

        retMemAddr = (T *) mmap(NULL, fileSizeInBytes,
                                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (retMemAddr == MAP_FAILED) {
            ERR_LOGS("Cannot mmap %s because:%s", fileName, strerror(errno));
            close(fd);
            return false;
        }

        if (ftruncate(fd, fileSizeInBytes) == -1) {
            ERR_LOGS("Cannot truncate %s because:%s", fileName, strerror(errno));
            close(fd);
            return false;
        }

        return true;
    }

    template<typename T>
    bool fClose(int &fd, size_t fileSizeInBytes, T *&retMemAddr) {
        if (munmap(retMemAddr, fileSizeInBytes) == -1) {
            ERR_LOG("Cannot close file");
            return false;
        }
        close(fd);
        return true;
    }

    #define GET_PAGE_BOUND(addr, page_size) (Elf64_Addr *) ((ssize_t) (addr) / page_size * page_size)

    void *memSearch(void *target, ssize_t targetSize, void *keyword, ssize_t keywordSize);

    bool adjustMemPerm(void *startPtr, void *endPtr, int prem);

    inline uint8_t *autoAddBaseAddr(uint8_t *targetAddr, uint8_t *baseAddr, uint8_t *startAddr, uint8_t *endAddr) {
        uint8_t *ret = baseAddr + (uint64_t) targetAddr;
        if (ret < startAddr && endAddr > ret) {
            //ret not in the desired range
            ret = targetAddr;
        } else if (startAddr <= targetAddr &&
                   targetAddr <= endAddr) {
            ret = targetAddr;
        }
        return ret;
    }

    bool strEndsWith(std::string const &fullString, std::string const &ending);

    bool strStartsWith(std::string const &fullString, std::string const &ending);

    bool strContains(std::string const &fullString, std::string const &ending);

    /**
     * Replace multiple space by one
     * @param butterOutput This buffer should have the same allocatedSize as oriString.allocatedSize
     * @return true->success false->fail
     */
    bool collapseStrSpace(const std::string &oriString,std::string& outString);

    void print_stacktrace(void);
    void print_stacktrace(std::ofstream & output);

    /**
     * Debug function. Print pystacktrace.
     */
    void print_pystacktrace(void);

    void printPythonStackTrace();
    /**
     * @brief Collecting the stack trace and saving it to the memory pointed by ptr
     * Each level will be saved to one entry, not larger than CPP_CALL_STACK_LEVEL
     * This function will automatically change CallStack::levels to min(CallStack::levels.)
     *
     * @return: the number of stack frames
     */
    void getCppStacktrace(CallStack<void*, CPP_CALL_STACK_LEVEL>& retCallStack);

    //Copied from pytorch to just show that the output is consistent with the error message. We can delete it.
    inline std::string format_size(uint64_t size) {
        std::ostringstream os;
        os.precision(2);
        os << std::fixed;
        if (size <= 1024) {
            os << size << " bytes";
        } else if (size <= 1048576) {
            os << (size / 1024.0);
            os << " KiB";
        } else if (size <= 1073741824ULL) {
            os << size / 1048576.0;
            os << " MiB";
        } else {
            os << size / 1073741824.0;
            os << " GiB";
        }
        return os.str();
    }

    inline std::string getProcessName(ssize_t pid) {
        std::string procName;
        std::ifstream ifs;
        std::stringstream ss;
        ss<<"/proc/"<<pid<<"/comm";
        ifs.open(ss.str());
        ifs >> procName;
        ifs.close();
        return std::move(procName);
    }

    inline bool isPythonAvailable(){
        return dlsym(RTLD_DEFAULT, "Py_IsInitialized") != nullptr;
    }

#ifdef CUDA_ENABLED

#endif


}


#endif //TOOL_H
