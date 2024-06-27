#include "Logging.h"
#include <string>
#include <sys/stat.h>
#include <cstring>
#include <sstream>
#include <cassert>
#include <csignal>
#include <fstream>
#include <atomic>

namespace mlinsight {

    std::string logRootPath;
    std::string logProcessRootPath;
    const char* localRank=nullptr;
    bool isRankParentProcess=false;
    FILE *logFileStd = nullptr;
    std::atomic<ssize_t> logPid = -1;
    pthread_mutex_t logInitializationLock = PTHREAD_MUTEX_INITIALIZER;
    bool pythonIsAvailable = false;
    bool DEBUGGER_CONTINUE = false;


    void checkLogFolder() {
        if (logRootPath.empty()) {
            fatalError("logRootPath is empty, please invoke initLog() before calling any log macros");
        }

        if (getpid() == logPid.load(std::memory_order_acquire)) {
            //The process haven't changed which means the recording folder exists
            return;
        }
        pthread_mutex_lock(&logInitializationLock);

        if (getpid() == logPid.load(std::memory_order_acquire)) {
            //The pid and folder creation has been done by another thread.
            pthread_mutex_unlock(&logInitializationLock);
            return;
        }


        // New PID, this must be a fork
        logPid.store(getpid(), std::memory_order_release);

        // Get process name
        std::string procName = getProcessName(logPid);

        char tmpStr[4096];
        snprintf(tmpStr, 4096, "%s/%s_%ld",logRootPath.c_str(), procName.c_str(), logPid.load());

        localRank=getenv("LOCAL_RANK");
        if(localRank!=nullptr && getenv("MLINSIGHTLOADER_INSTALLED_RANK")==nullptr){
            isRankParentProcess=true;
            setenv("MLINSIGHTLOADER_INSTALLED_RANK",localRank,1);
        }else{
            isRankParentProcess=false;
        }

        if(isRankParentProcess){
            assert(isPythonAvailable());
            snprintf(tmpStr, 4096, "%s/%s_%ld_Rank%s",logRootPath.c_str(), procName.c_str(), logPid.load(), localRank);
        }

        logProcessRootPath = tmpStr;


        struct stat s1;
        if (stat(logProcessRootPath.c_str(), &s1) == 0) {
            //When fork_proxy is invoked, the child process already invokes "checkLogFolder" and created this folder.
            //However, the main function of the child maybe called again and the variable "logPid" is reset to -1 after folder creation.
            //This will cause the child process to execute this branch and attempt to create the folder again which is unnecessary.
            //In this case all log files has been created in fork_proxy we just need to return.
        } else {
            if (mkdir(logProcessRootPath.c_str(), 0755) == -1) {
                fatalErrorS("Cannot mkdir %s because: %s", logProcessRootPath.c_str(),
                            strerror(errno));
            }
        }

        snprintf(tmpStr, 4096, "%s/log.txt", logProcessRootPath.c_str());
        logFileStd = fopen(tmpStr, "a+");
        if (!logFileStd) {
            //File not exists use w to create a new file. Although the document say "a" will create file but it seems not working
            logFileStd = fopen(tmpStr, "w");
        }
        if (logFileStd == NULL) {
            fatalErrorS("Cannot create file %s because: %s",tmpStr,
                        strerror(errno));
        }
        pthread_mutex_unlock(&logInitializationLock);
    }

    /**
     * Create root recording folder on startup
     * @param logRoot
     */
    void initLog() {
        assert(logRootPath.empty());
        char *pathFromEnv = getenv("MLINSIGHTLOADER_LOGROOT");
        
        if (pathFromEnv == NULL) {
            //Log root not set before. Set it now.
            ssize_t rootPid = getpid();
            char tmpStr[4096];
            snprintf(tmpStr, 4096, "/tmp/mlinsightloader_%s_%zd_%jd", getProcessName(rootPid).c_str(), rootPid, getunixtimestampms());
            logRootPath = tmpStr;

            if (mkdir(logRootPath.c_str(), 0755) == -1) {
                fatalErrorS("Cannot mkdir %s because: %s", logRootPath.c_str(),
                            strerror(errno));
            }

            // Important: should not replace this variable.
            setenv("MLINSIGHTLOADER_LOGROOT", logRootPath.c_str(), 0);


            snprintf(tmpStr, 4096, "%d", getpid());
            setenv("MLINSIGHTLOADER_LOGROOT_PID", tmpStr, 0);
        } else {
            logRootPath = pathFromEnv;
        }

        //Create log folder
        checkLogFolder();

        INFO_LOGS("MLInsight debug log is located at: %s PID=%d\n", logRootPath.c_str(), getpid());

    }

    bool isSafeToUseStdout() {
        const char *logRootPidStr = getenv("MLINSIGHTLOADER_LOGROOT_PID");
        if (logRootPidStr == nullptr) {
            //Some processes (eg: Bash may clear environment variables. In this case we re-initialize the logging)
            initLog();
        }
        logRootPidStr = getenv("MLINSIGHTLOADER_LOGROOT_PID");
        if (logRootPidStr == nullptr) {
            fatalError("MLINSIGHTLOADER_LOGROOT_PID is empty. initLog() should be invoked before any log function?");
            return false;
        }
        ssize_t logRootPid = -1;
        try {
            logRootPid = std::stoi(logRootPidStr);
        } catch (std::invalid_argument &e) {
            fatalError("Cannot correctly parse MLINSIGHT_LOGROOT_PID");
            return false;
        }

        return logRootPid == getpid();
    };
}