/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <cstdio>
//=========================================================================
// Thread Functions
//=========================================================================

void AA(){
    printf("AA\n");
}

void A(){
    printf("A\n");
    AA();
}

void *testThread1(void *data) {
    for(int i=0;i<100;++i){
        printf("%d\n",i);
        A();
    }
    return nullptr;
}

int main() {

    for(int i=0;i<100;++i){
        printf("%d\n",i);
    }

    pthread_t thread1;
    int iret1;

    /* Create independent threads each of which will execute function */
    iret1 = pthread_create(&thread1, NULL, testThread1, NULL);

    /* Wait till threads are complete before main continues. Unless we  */
    /* wait we run the risk of executing an exit which will terminate   */
    /* the process and all threads before the threads have completed.   */
    pthread_join(thread1, NULL);


    exit(0);
    return 0;
}

