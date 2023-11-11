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

// Declaration of thread condition variable
pthread_cond_t cond0 = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond2 = PTHREAD_COND_INITIALIZER;

// declaring mutex
pthread_mutex_t lock0 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

unsigned int turn = 0;

void *testThread1(void *data) {
    while (true) {
        printf("thread1 pthread_mutex_lock lID=%p\n",&lock0);
        pthread_mutex_lock(&lock0);
        while (turn % 3 != 0 && turn < 6) {
            printf("thread1 pthread_cond_wait condID=%p lID=%p\n",&cond0,&lock0);
            pthread_cond_wait(&cond0, &lock0);
        }

        if (turn < 6) {
            int randNum = rand() % 100 + 1;
            void *memAddr = malloc(1024);
            fprintf(stderr, "A Rand:%d %p\n", randNum, memAddr);
            ++turn;
            printf("thread1 pthread_mutex_lock lID=%p\n",&lock1);
            pthread_mutex_lock(&lock1);
            printf("thread1 pthread_cond_signal condID=%p\n",&cond1);
            pthread_cond_signal(&cond1);
            printf("thread1 pthread_mutex_unlock lID=%p\n",&lock1);
            pthread_mutex_unlock(&lock1);
            printf("thread1 pthread_mutex_unlock lID=%p\n",&lock0);
            pthread_mutex_unlock(&lock0);
        } else {
            //Notify next thread to exit
            printf("thread1 pthread_mutex_lock lID=%p\n",&lock1);
            pthread_mutex_lock(&lock1);
            printf("thread1 pthread_cond_signal condID=%p\n",&cond1);
            pthread_cond_signal(&cond1);
            printf("thread1 pthread_mutex_unlock lID=%p\n",&lock1);
            pthread_mutex_unlock(&lock1);
            printf("thread1 pthread_mutex_unlock lID=%p\n",&lock0);
            pthread_mutex_unlock(&lock0);
            break;
        }
    }
    return nullptr;
}

void *testThread2(void *data) {
    while (true) {
        printf("thread2 pthread_mutex_lock lID=%p\n",&lock1);
        pthread_mutex_lock(&lock1);
        while (turn % 3 != 1 && turn < 6) {
            printf("thread2 pthread_cond_wait condID=%p lID=%p\n",&cond1,&lock1);
            pthread_cond_wait(&cond1, &lock1);
        }
        if (turn < 6) {
            int randNum = rand() % 100 + 1;
            void *memAddr = malloc(1024);
            fprintf(stderr, "B Rand:%d %p\n", randNum, memAddr);

            ++turn;

            printf("thread2 pthread_mutex_lock lID=%p\n",&lock2);
            pthread_mutex_lock(&lock2);
            printf("thread2 pthread_cond_signal condID=%p\n",&cond2);
            pthread_cond_signal(&cond2);
            printf("thread2 pthread_mutex_unlock lID=%p\n",&lock2);
            pthread_mutex_unlock(&lock2);
            printf("thread2 pthread_mutex_unlock lID=%p\n",&lock1);
            pthread_mutex_unlock(&lock1);
        } else {
            //Notify next thread to exit
            printf("thread2 pthread_mutex_lock lID=%p\n",&lock2);
            pthread_mutex_lock(&lock2);
            printf("thread2 pthread_cond_signal condID=%p\n",&cond2);
            pthread_cond_signal(&cond2);
            printf("thread2 pthread_mutex_unlock lID=%p\n",&lock2);
            pthread_mutex_unlock(&lock2);
            printf("thread2 pthread_mutex_unlock lID=%p\n",&lock1);
            pthread_mutex_unlock(&lock1);
            break;
        }
    }
    return nullptr;
}

void *testThread3(void *data) {
    while (true) {
        printf("thread3 pthread_mutex_lock lID=%p\n",&lock2);
        pthread_mutex_lock(&lock2);
        while (turn % 3 != 2 && turn < 6) {
            printf("thread3 pthread_cond_wait condID=%p lID=%p\n",&cond2,&lock2);
            pthread_cond_wait(&cond2, &lock2);
        }
        if (turn < 6) {
            int randNum = rand() % 100 + 1;
            void *memAddr = malloc(1024);
            fprintf(stderr, "C Rand:%d %p\n", randNum, memAddr);
            ++turn;

            printf("thread3 pthread_mutex_lock lID=%p\n",&lock0);
            pthread_mutex_lock(&lock0);
            printf("thread3 pthread_cond_signal condID=%p\n",&cond0);
            pthread_cond_signal(&cond0);
            printf("thread3 pthread_mutex_unlock lID=%p\n",&lock0);
            pthread_mutex_unlock(&lock0);
            printf("thread3 pthread_mutex_unlock lID=%p\n",&lock2);
            pthread_mutex_unlock(&lock2);
        } else {
            //Notify next thread to exit
            printf("thread3 pthread_mutex_lock lID=%p\n",&lock0);
            pthread_mutex_lock(&lock0);
            printf("thread3 pthread_cond_signal condID=%p\n",&cond0);
            pthread_cond_signal(&cond0);
            printf("thread3 pthread_mutex_unlock lID=%p\n",&lock0);
            pthread_mutex_unlock(&lock0);
            printf("thread3 pthread_mutex_unlock lID=%p\n",&lock2);
            pthread_mutex_unlock(&lock2);
            break;
        }
    }
    return nullptr;
}

//=========================================================================
// Debugger loop
//=========================================================================

volatile int DEBUGGER_WAIT = 1;

void test_continue() {
    DEBUGGER_WAIT = 0;
}


int main(int argc, char *argv[]) {

    pthread_t thread1, thread2, thread3;
    int iret1, iret2, iret3;

    /* Create independent threads each of which will execute function */
    iret1 = pthread_create(&thread1, NULL, testThread1, NULL);
    iret2 = pthread_create(&thread2, NULL, testThread2, NULL);
    iret3 = pthread_create(&thread3, NULL, testThread3, NULL);

    /* Wait till threads are complete before main continues. Unless we  */
    /* wait we run the risk of executing an exit which will terminate   */
    /* the process and all threads before the threads have completed.   */
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);

//    while (DEBUGGER_WAIT) {
    //Let gdb break
    //raise(SIGTRAP);
//    }

    exit(0);
    return 0;
}

