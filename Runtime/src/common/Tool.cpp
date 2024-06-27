/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/

#include <atomic>
#include <iostream>
#include <fstream>
#include <climits>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <csignal>
#include <Python.h>
#include <frameobject.h>
#include <ceval.h>

#include "common/Tool.h"
#include "trace/hook/HookInstaller.h"

namespace mlinsight {
    std::vector<ssize_t> findStrSplit(std::string &srcStr, char splitChar) {
        std::vector<ssize_t> splitPoints;
        //Augment the first and last character in a string with splitChar.
        //This make edge cases easier to handle
        std::stringstream ss;
        ss << splitChar << srcStr << splitChar;
        std::string augSrcStr = ss.str(); //Augmented srcStr

        for (int i = 1; i < augSrcStr.size(); ++i) {
            //Loop through every character in augSrcStr

            if (augSrcStr[i - 1] != augSrcStr[i]) {
                //If there's a character change, then it may be the staring/ending position of a text segment

                if (augSrcStr[i - 1] == splitChar) {
                    //If current symbol is different from last symbol, and the previous symbol is splitchar.
                    //Then previous location i should be marked as the begging of the begging of a text segment.
                    //Here we are manipulating augmented string, there's one more character in the beginning.
                    //We want to return the index of the original string. So we should push back i-1 rather than 1.

                    //Mark the starting of this text segment as i-1
                    splitPoints.emplace_back(i - 1);

                } else if (augSrcStr[i] == splitChar) {
                    //If current symbol is different from last symbol, and the current symbol is splitchar.
                    //Then current location i (right NOT inclusive) should be marked as the end of a text segment.
                    //Here we are manipulating augmented string, there's one more character in the beginning.
                    //We want to return the index of the original string. So we should push back i-1 rather than 1.

                    //Mark the ending of this text segment as i-1
                    splitPoints.emplace_back(i - 1);
                }

            }
        }
        return splitPoints;
    }

    long int getFileSize(FILE * file) {
        //The type of this return value is used by ftell. So it should be universal
        if (fseek(file, 0L, SEEK_END) != 0) {
            ERR_LOGS("fseek failed because: %s", strerror(errno));
            return -1;
        }
        long int fileSize = ftell(file);
        if (!fileSize) {
            ERR_LOGS("ftell failed because: %s", strerror(errno));
            return -1;
        }
        if (fseek(file, 0L, SEEK_SET) != 0) {
            ERR_LOGS("ftell failed because: %s", strerror(errno));
            return -1;
        }
        return fileSize;
    }

    bool extractFileName(std::string absolutePath, std::string &pathName, std::string &fileName) {
        auto posi = absolutePath.find_last_of('/');
        if (posi == std::string::npos) {
            //Not found, return full string
            //ERR_LOGS("Path incorrect: %s", absolutePath.c_str());
            return false;
        } else {
            pathName = absolutePath.substr(0, posi);
            fileName = absolutePath.substr(posi + 1, absolutePath.length() - posi);
            return true;
        }
    }

    bool getPWD(std::string &retPwdPath) {
        auto mlinsightWorkDirPtr = getenv("MLINSIGHT_WORKDIR");

        if (!mlinsightWorkDirPtr) {
            //Use cwd
            char temp[PATH_MAX];

            if (getcwd(temp, PATH_MAX) != 0) {
                retPwdPath = std::string(temp);
                return true;
            } else {
                ERR_LOGS("getcwd failed because %s", strerror(errno));
                return false;
            }
        } else {
            //Use env variable
            retPwdPath = mlinsightWorkDirPtr;
            return true;
        }
    }

    void *memSearch(void *target, ssize_t targetSize, void *keyword, ssize_t keywordSize) {
        //Convert it to uint8* so that we can perform arithmetic operation on those pointers
        uint8_t *kwd = static_cast<uint8_t *>(keyword);
        uint8_t *tgt = static_cast<uint8_t *>(target);

        int i = 0, j = 0; //i is the index in target and j is the index in keyword
        uint8_t *beg = nullptr; //Markes the begging of the match

        while (i < targetSize && j < keywordSize) {
            if (tgt[i] == kwd[j]) {
                if (beg == nullptr) {
                    //First match. It's a potential starting position.
                    beg = tgt + i;
                }
                ++j;
            } else {
                //If tgt[i] != kwd[j] it means this is not the correct keyword. Reset beg and j.
                beg = nullptr;
                j = 0;
            }
            ++i;
        }
        // If j==keywordSize it means the previous loop exit because of this. Then it means a match if found.
        return j == keywordSize ? beg : nullptr;
    }

    /**
     * This function only applies to page-aligned memory.
    */
    bool adjustMemPerm(void *startPtr, void *endPtr, int prem) {
        //Get page allocatedSize
        ssize_t pageSize = sysconf(_SC_PAGESIZE);
        //Get Page Bound
        void *startPtrBound = GET_PAGE_BOUND(startPtr, pageSize);
        void *endPtrBound = endPtr;
        if (startPtrBound == endPtrBound)
            endPtrBound = (uint8_t *) startPtrBound + pageSize;

        //todo:(uint8_t *) endPtrBound - (uint8_t  *) startPtrBound,
        ssize_t memoryLength =
                (ceil(((uint8_t *) endPtrBound - (uint8_t *) startPtrBound) / (double) pageSize)) * pageSize;
        //DBG_LOGS("Real addr from:%p to:%p", startPtrBound, endPtrBound);
        if (mprotect(startPtrBound, memoryLength, prem) != 0) {
            ERR_LOGS("Could not change the process driverMemRecord permission at %p-%p because: %s", startPtrBound, endPtrBound,
                     strerror(errno));
            signal(SIGINT, 0);
            return false;
        }
        return true;
    }


    bool strEndsWith(const std::string &fullString, const std::string &ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }


    bool strStartsWith(const std::string &fullString, const std::string &starting) {
        if (fullString.length() >= starting.length()) {
            return (0 == fullString.compare(0, starting.length(), starting));
        } else {
            return false;
        }
    }


    bool strContains(const std::string &fullString, const std::string &keyword) {
        return fullString.find(keyword) != std::string::npos;
    }

    bool collapseStrSpace(const std::string &oriString, std::string &outString) {
        //todo: expensive op. Allocate with heap.
        outString = oriString;
        std::stringstream ss;

        bool spaceInserted = true;
        for (int i = 0; i < outString.size(); ++i) {
            auto &curChar = outString[i];
            if (outString[i] != ' ') {
                spaceInserted = false;
                ss << curChar;
            } else if (!spaceInserted) {
                ss << " ";
                spaceInserted = true;
            }
        }
        outString = ss.str();

        return true;
    }


    void *memSearch(void *target, ssize_t targetSize, void *keyword, ssize_t keywordSize);

    bool adjustMemPerm(void *startPtr, void *endPtr, int prem);

    /* Obtain a backtrace and print it to stdout. */
    void print_stacktrace(void) {
#define CALL_STACK_NUM 4096
        void *array[CALL_STACK_NUM];
        char **strings;
        int size, i;

        size = backtrace(array, CALL_STACK_NUM);
#if 0
        for (i = 0; i < allocatedSize; i++) {
            INFO_LOGS("[%d], %p\n", i, array[i]);
        }
#endif
        strings = backtrace_symbols(array, size);
        if (strings != NULL) {

            OUTPUTS("Obtained %d stack frames.\n", size);
            for (i = 0; i < size; i++) {
                OUTPUTS ("%s\n", strings[i]);
            }
        }

        free(strings);
    }

    void print_stacktrace(std::ostream &output) {
        using namespace std;
        void *array[CPP_CALL_STACK_LEVEL];
        char **strings;
        int size, i;

        size = backtrace(array, CPP_CALL_STACK_LEVEL);

        strings = backtrace_symbols(array, size);
        if (strings != NULL) {

            output << "Obtained " << size << "stack frames." << endl;
            for (i = 0; i < size; i++)
                output << strings[i] << endl;
        }

        free(strings);
    }

    extern int PyCodeExtra_Index;

    void print_pystacktrace() {
        if (isPythonAvailable() && Py_IsInitialized()) {
            //        //Child paorcess
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            PyFrameObject *currentFrame = PyEval_GetFrame();
            int i = 0;
            OUTPUT("=========PYCALLSTACK.LEVELS.START\n");
            while (currentFrame != NULL) {

                const char *pythonSourceFileName = PyUnicode_AsUTF8(currentFrame->f_code->co_filename);
                const char *pythonFunctionName = PyUnicode_AsUTF8(currentFrame->f_code->co_name);

                PythonFrameExtra_t *retPyCodeExtra;
                int pyCodeExtraRlt = _PyCode_GetExtra((PyObject *) currentFrame->f_code, PyCodeExtra_Index,
                                                      (void **) &retPyCodeExtra);

                HookInstaller *instance = HookInstaller::getInstance();
                if (retPyCodeExtra) {
                    OUTPUTS("%d Function: %s Line: %s:%d Package: %s\n", i++, pythonFunctionName,
                            pythonSourceFileName, PyFrame_GetLineNumber(currentFrame),
                            instance->pyModuleInfoMap[retPyCodeExtra->globalPyModuleId].moduleName.c_str());
                } else {
                    OUTPUTS("%d Function: %s Line: %s:%d Package: NONE\n", i++, pythonFunctionName,
                            pythonSourceFileName, PyFrame_GetLineNumber(currentFrame));
                }
                //Go to next frame
                currentFrame = currentFrame->f_back;
            }
            OUTPUT("=========PYCALLSTACK.LEVELS.END\n");
            //Release GIL
            PyGILState_Release(gstate);
        } else {
            ERR_LOG("Process %d is not a python process so MLInsight cannot print python process");
        }
    }

    extern void *pythonInterpreter_text_begin;
    extern void *pythonInterpreter_text_end;
    bool pythonIsAvailable = false;
    bool DEBUGGER_CONTINUE=false;

    void print_hybridstacktrace() {
        if (isPythonAvailable() && Py_IsInitialized()) {
            //        //Child paorcess
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            PyFrameObject *currentFrame = PyEval_GetFrame();
            int i = 0;
            OUTPUT("=========HYBRIDSTACK.LEVELS.START\n");

            //Find how many levels of stacks are below the first Python interpreter stack.
            using namespace std;
            const int hybridCStackLevel = 60;
            void *array[hybridCStackLevel];
            int size = backtrace(array, hybridCStackLevel);
            int bottomCCallStackSize = 0;

            for (int i = 0; i < size; ++i) {
                if (pythonInterpreter_text_begin <= array[i] && array[i] <= pythonInterpreter_text_end) {
                    //Is python interpreter stack frame
                    bottomCCallStackSize = i;
                    break;
                }
            }
            if(bottomCCallStackSize==0){
                bottomCCallStackSize=size;
            }

            //Insert C stack into

            char **strings = backtrace_symbols(array, bottomCCallStackSize);
            for (int i = 0; i < bottomCCallStackSize; ++i) {
                OUTPUTS("%s\n", strings[i]);
            }
            free(strings);

//            bool findProxyFunc=false;
            //Add the python part
            while (currentFrame != NULL) {

                const char *pythonSourceFileName = PyUnicode_AsUTF8(currentFrame->f_code->co_filename);
                const char *pythonFunctionName = PyUnicode_AsUTF8(currentFrame->f_code->co_name);

                PythonFrameExtra_t *retPyCodeExtra;
                int pyCodeExtraRlt = _PyCode_GetExtra((PyObject *) currentFrame->f_code, PyCodeExtra_Index,
                                                      (void **) &retPyCodeExtra);
//                if(strncmp(pythonFunctionName,"proxyFunc",9)==0){
//                    findProxyFunc=true;
//                }

                HookInstaller *instance = HookInstaller::getInstance();
                if (retPyCodeExtra) {
                    OUTPUTS("%d Function: %s Line: %s:%d Package: %s\n", i++, pythonFunctionName,
                            pythonSourceFileName, PyFrame_GetLineNumber(currentFrame),
                            instance->pyModuleInfoMap[retPyCodeExtra->globalPyModuleId].moduleName.c_str());
                } else {
                    OUTPUTS("%d Function: %s Line: %s:%d Package: NONE\n", i++, pythonFunctionName,
                            pythonSourceFileName, PyFrame_GetLineNumber(currentFrame));
                }
                //Go to next frame
                currentFrame = currentFrame->f_back;
            }
//            if(!findProxyFunc){
//                OUTPUT("WARNING: Did not see proxyFunc from the stack\n");
//            }
            OUTPUT("=========HYBRIDSTACK.LEVELS.END\n");
            //Release GIL
            PyGILState_Release(gstate);
        } else {
            ERR_LOG("Process %d is not a python process so MLInsight cannot print python process");
        }
    }


}