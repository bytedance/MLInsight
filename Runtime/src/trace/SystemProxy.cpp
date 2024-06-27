/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <dlfcn.h>
#include "trace/proxy/SystemProxy.h"
#include "common/Logging.h"
#include "common/Tool.h"

#include <cassert>
#include <stdio.h>
#include <sstream>
#include <unistd.h>


namespace mlinsight {


    __pid_t fork_proxy(void) {
        __pid_t parentPid = getpid();
        __pid_t forkRet = fork();
        if (forkRet == 0) {
#ifndef NDEBUG
//        DBG_LOG("Here is the python stack that invoked the fork operation.")
//        print_pystacktrace();
#endif
            int childPid = getpid();
            //DBG_LOGS("This process (pid=%d) was forked from parent process (pid=%d)\n", childPid, parentPid);
            fflush(logFileStd);

        } else if (forkRet != -1) {
            //Parent process
            //DBG_LOGS("This process (pid=%d) forked child process (pid=%d)\n", parentPid, forkRet);
        }
        return forkRet;
    }
}