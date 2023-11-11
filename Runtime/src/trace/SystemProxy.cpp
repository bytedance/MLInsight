/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
@author: Tongping Liu <tongping.liu@bytedance.com>
*/
#include <dlfcn.h>
#include "trace/proxy/SystemProxy.h"

#include <cassert>
#include <stdio.h>
#include <unistd.h>

namespace mlinsight{
__pid_t fork_proxy(void) {
    
    return fork();
}
}