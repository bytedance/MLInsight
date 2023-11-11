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
    sleep50Seconds();
}

void* pthread2Run(void*){
    sleep30Seconds();
}

void* pthread3Run(void*){
    sleep10Seconds();
}


int main(){
    pthread_t pthread1;
    pthread_create(&pthread1,nullptr,pthread1Run,nullptr);
    struct timespec rem={0,0}, req = {
            10, 0
    };
    int ret=nanosleep(&req,&rem);
    if(ret!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    pthread_t pthread2;
    pthread_create(&pthread2,nullptr,pthread2Run,nullptr);
    struct timespec rem1={0,0}, req1 = {
            10, 0
    };
    int ret1=nanosleep(&req1,&rem1);
    if(ret1!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    pthread_t pthread3;
    pthread_create(&pthread3,nullptr,pthread3Run,nullptr);

    pthread_join(pthread1,nullptr);
    pthread_join(pthread2,nullptr);
    pthread_join(pthread3,nullptr);

}