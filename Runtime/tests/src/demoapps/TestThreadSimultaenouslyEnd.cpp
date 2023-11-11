/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <pthread.h>
#include <libWorkload.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdio>

void * pthread1Run(void *){
    sleep10Seconds();
}

void* pthread2Run(void*){
    sleep10Seconds();
}

void* pthread3Run(void*){
    sleep10Seconds();
}


int main(){
    pthread_t pthread1;
    pthread_create(&pthread1,nullptr,pthread1Run,nullptr);

    pthread_t pthread2;
    pthread_create(&pthread2,nullptr,pthread2Run,nullptr);

    pthread_t pthread3;
    pthread_create(&pthread3,nullptr,pthread3Run,nullptr);

    pthread_join(pthread1,nullptr);
    pthread_join(pthread2,nullptr);
    pthread_join(pthread3,nullptr);
}