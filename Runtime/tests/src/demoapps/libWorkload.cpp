/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <unistd.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <cstring>
#include <cstdlib>

void sleep30Seconds(){
    printf("Sleep30Seconds Start\n");
    struct timespec rem={0,0}, req = {
            30, 0
    };
    int ret=nanosleep(&req,&rem);
    if(ret!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    printf("Sleep30Seconds End\n");
}

void sleep50Seconds(){
    printf("Sleep50Second Start\n");
    struct timespec rem={0,0}, req = {
            50, 0
    };
    int ret=nanosleep(&req,&rem);
    if(ret!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    printf("Sleep50Seconds End\n");
}


void sleep5Seconds(){
    printf("Sleep5Seconds Start\n");
    struct timespec rem={0,0}, req = {
            5, 0
    };
    int ret=nanosleep(&req,&rem);
    if(ret!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    printf("Sleep5Seconds End\n");
}

void sleep10Seconds(){
    printf("Sleep10Seconds Start\n");
    struct timespec rem={0,0}, req = {
            10, 0
    };
    int ret=nanosleep(&req,&rem);
    if(ret!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    printf("Sleep10Seconds End\n");
}


void sleep15Seconds(){
    printf("Sleep15Seconds Start\n");
    struct timespec rem={0,0}, req = {
            15, 0
    };
    int ret=nanosleep(&req,&rem);
    if(ret!=0){
        printf("Error: %s",strerror(errno));
        exit(-1);
    }
    printf("Sleep15Seconds End\n");
}

