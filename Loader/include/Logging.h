/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_LOGGING_H
#define MLINSIGHT_LOGGING_H

#include <cstdio>
#include <string>
#include <atomic>
#include <unistd.h>
#include <dlfcn.h>
#include <string>
#include <fstream>

#define SAVE_LOG_TO_FILE 1
#define PRINT_DBG_LOG 1
#define PRINT_ERR_LOG 1
#define PRINT_INFO_LOG 1

namespace mlinsight {

    inline std::string getProcessName(ssize_t pid) {
        std::string procName;
        std::ifstream ifs;
        char tmpStr[4096];
        snprintf(tmpStr, 4096, "/proc/%ld/comm",pid);
        ifs.open(tmpStr);
        ifs >> procName;
        ifs.close();
        return std::move(procName);
    }

    extern bool pythonIsAvailable;
    inline bool isPythonAvailable(){
        if(!pythonIsAvailable){
            pythonIsAvailable = (dlsym(RTLD_DEFAULT, "Py_IsInitialized") != nullptr);
        }
        return pythonIsAvailable;
    }

    inline int64_t getunixtimestampms() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((int64_t) hi << 32) | lo;
    }


    inline bool strEndsWith(const std::string &fullString, const std::string &ending) {
        if (fullString.length() >= ending.length()) {
            return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
        } else {
            return false;
        }
    }


    inline bool strStartsWith(const std::string &fullString, const std::string &starting) {
        if (fullString.length() >= starting.length()) {
            return (0 == fullString.compare(0, starting.length(), starting));
        } else {
            return false;
        }
    }


    inline bool strContains(const std::string &fullString, const std::string &keyword) {
        return fullString.find(keyword) != std::string::npos;
    }

//Note that all configurations should be listed in "common/Config.h"

//All mlinsight output should be saved into this path
    extern std::string logRootPath;
    extern bool isRankParentProcess;

#if SAVE_LOG_TO_FILE
    //The root log path for this process
    extern std::string logProcessRootPath;
    extern const char* localRank;
    //The PID associated with logProcessRootPath
    extern std::atomic<ssize_t> logPid;
    extern FILE *logFileStd;


    void checkLogFolder();

    /**
     * Set variable logRootPath and create this folder
     */
    void initLog();

    /**
     * Check whether it is safe to print out onto stdout and stderr
     * @return Safe or not
     */
    bool isSafeToUseStdout();

#endif


/**
 * Four types of logs are available:
 *      DBG_LOG, INFO_LOG, ERR_LOG: Debugging related log.
 *      OUTPUT are used to report data to the user.
 *      fatalError are used to report program crash.
 * XXX_LOG only supports a constant string, while XXX_LOGS supports printf APIs.
 * ALL log types other than OUTPUT will insert __FILE__ and __LINE__ information and append '\n' to the end of the log.
 *
 * ALL log macros will not output to screen if "isSafeToUseStdout" returns false. By default, this function returns false if a process is a sub-process.
 * Developers can customize "isSafeToUseStdout" function to control the behavior of subprocess log printing.
 *
 * WARNING: Developers should not use cout or printf function to output data directly to stdout/stderr because the output may interfere with program execution.
 * Using log macros guarantee safe output.
 *
 * If SAVE_LOG_TO_FILE is defined, then logs are also saved to disk. The root log folder is printed at the end of the program main process.
 * Uses can use PRINT_XXX_LOG macros to turn specific types of debug logs on or off. But OUTPUTS and fatalErrors cannot be turned off.
 */
extern bool DEBUGGER_CONTINUE;
#if PRINT_DBG_LOG

#if SAVE_LOG_TO_FILE
    //Print log to file (Recommended)
    // Print a single log string
#define DBG_LOG(str) { \
            if(::mlinsight::isSafeToUseStdout()){ \
                fprintf(stderr,"DBG: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",str); \
            } \
            checkLogFolder(); fprintf(logFileStd,"DBG: %s:%d  ",__FILE__,__LINE__); fprintf(logFileStd,"%s\n",str);}
    // Print log strings using printf template format
#define DBG_LOGS(fmt, ...) { \
            if(::mlinsight::isSafeToUseStdout()){ \
                fprintf(stderr,"DBG: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,fmt,__VA_ARGS__); fprintf(stderr,"\n"); \
            } \
            checkLogFolder(); fprintf(logFileStd,"DBG: %s:%d  ",__FILE__,__LINE__); fprintf(logFileStd,fmt,__VA_ARGS__); fprintf(logFileStd,"\n");}
    // Print a single error string

#else
    //Print log to screen (Not recommended)

    // Print a single log string
#define DBG_LOG(str) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"DBG: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",str);}}
    // Print log strings using printf template format
#define DBG_LOGS(fmt, ...) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"DBG: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,fmt,__VA_ARGS__); fprintf(stderr,"\n");}}
    // Print a single error string
#endif

#else

    // Print a single log string
#define DBG_LOG(str)
    // Print log strings using printf template format
#define DBG_LOGS(fmt, ...)
    // Print a single error string

#endif

#if PRINT_ERR_LOG

#ifdef SAVE_LOG_TO_FILE

#define ERR_LOG(str) { \
            if(::mlinsight::isSafeToUseStdout()){ \
                fprintf(stderr,"ERR: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",str); \
            } \
            ::mlinsight::checkLogFolder(); fprintf(::mlinsight::logFileStd,"ERR: %s:%d  ",__FILE__,__LINE__); fprintf(::mlinsight::logFileStd,"%s\n",str);}
    // Print log strings using printf template format
#define ERR_LOGS(fmt, ...) { \
            if(::mlinsight::isSafeToUseStdout()){ \
                fprintf(stderr,"ERR: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,fmt,__VA_ARGS__);fprintf(stderr,"\n"); \
            } \
            ::mlinsight::checkLogFolder(); fprintf(::mlinsight::logFileStd,"ERR: %s:%d  ",__FILE__,__LINE__); fprintf(::mlinsight::logFileStd,fmt,__VA_ARGS__);fprintf(::mlinsight::logFileStd,"\n");}

#else

#define ERR_LOG(str) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"ERR: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",str);}}
    // Print log strings using printf template format
#define ERR_LOGS(fmt, ...) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"ERR: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,fmt,__VA_ARGS__);fprintf(logFileStd,"\n");}}

#endif
#else

#define ERR_LOG(str)
    // Print log strings using printf template format
#define ERR_LOGS(fmt, ...)

#endif

#if PRINT_INFO_LOG

#ifdef SAVE_LOG_TO_FILE
    // Print a single log string
#define INFO_LOG(str) { \
            if(::mlinsight::isSafeToUseStdout()){ \
                fprintf(stderr,"INFO: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",str); \
            }                   \
            ::mlinsight::checkLogFolder(); fprintf(::mlinsight::logFileStd,"INFO: %s:%d  ",__FILE__,__LINE__); fprintf(::mlinsight::logFileStd,"%s\n",str);}
    // Print log strings using printf template format
#define INFO_LOGS(fmt, ...) { \
            if(::mlinsight::isSafeToUseStdout()){ \
                fprintf(stderr,"INFO: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,fmt,__VA_ARGS__); fprintf(stderr,"\n"); \
            }                         \
            ::mlinsight::checkLogFolder(); fprintf(::mlinsight::logFileStd,"INFO: %s:%d  ",__FILE__,__LINE__); fprintf(::mlinsight::logFileStd,fmt,__VA_ARGS__); fprintf(::mlinsight::logFileStd,"\n");}
    // Print a single error string
#else
    // Print a single log string
#define INFO_LOG(str) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"INFO: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",str);}}
    // Print log strings using printf template format
#define INFO_LOGS(fmt, ...) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"INFO: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,fmt,__VA_ARGS__); fprintf(stderr,"\n");}}
    // Print a single error string
#endif
#else

#define INFO_LOG(str)

#define INFO_LOGS(fmt, ...)

#endif

//OUTPUT logs are always enabled and does not contain special file number and line number.
//OUTPUT log also does not automatically append '\n' to the end of the log string.
//For main process, OUTPUT log will directly print to screen. For sub process, OUTPUT log will only save to file (If SAVE_LOG_TO_FILE is defined).

#if SAVE_LOG_TO_FILE
    // Print a single log string
#define OUTPUT(str) { if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"%s",str);}  \
                        ::mlinsight::checkLogFolder(); fprintf(::mlinsight::logFileStd,"%s",str);}
    // Print log strings using printf template format
#define OUTPUTS(fmt, ...) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,fmt,__VA_ARGS__);} \
                        ::mlinsight::checkLogFolder(); fprintf(::mlinsight::logFileStd,fmt,__VA_ARGS__);}
    // Print a single error string
#else
    // Print a single log string
#define OUTPUT(str) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,"%s",str);}}
    // Print log strings using printf template format
#define OUTPUTS(fmt, ...) {if(::mlinsight::isSafeToUseStdout()){fprintf(stderr,fmt,__VA_ARGS__);}}
    // Print a single error string
#endif

//fatalError represents a program crash. So we can safely print it out to the screen for the ease of debugging.
#define fatalError(errMsg) {fprintf(stderr,"Fatal ERR: %s:%d  ",__FILE__,__LINE__); fprintf(stderr,"%s\n",errMsg); fprintf(stderr,"Waiting for debugger at tid=%zd pid=%d\n",pthread_self(),getpid());  fflush(logFileStd); while(!DEBUGGER_CONTINUE){ usleep(2000); } exit(-1);}
#define fatalErrorS(errFmt, ...) {fprintf(stderr,"Fatal ERR: %s:%d  ",__FILE__,__LINE__);  fprintf(stderr,errFmt,__VA_ARGS__); fprintf(stderr,"\n"); fprintf(stderr,"Waiting for debugger at tid=%zd pid=%d\n",pthread_self(),getpid()); fflush(logFileStd); while(!DEBUGGER_CONTINUE){ usleep(2000); }   exit(-1);}

}
#endif //MLINSIGHT_FILETOOL_H