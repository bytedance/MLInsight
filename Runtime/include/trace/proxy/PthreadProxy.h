/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_PTHREADPROXY_H
#define MLINSIGHT_PTHREADPROXY_H

#include<pthread.h>

namespace mlinsight{
int pthread_create_proxy(pthread_t *thread, const pthread_attr_t *attr, void *(*start)(void *), void *arg);
//void pthread_exit(void *__retval);
}


#endif //MLINSIGHT_PTHREADPROXY_H
