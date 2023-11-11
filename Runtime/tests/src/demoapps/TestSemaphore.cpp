/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <thread>

sem_t mutex;

void *thread(void *arg) {
    //wait
    printf("threadID=%lu sem_wait sID=%p\n", pthread_self(), &mutex);
    sem_wait(&mutex);
    printf("\nEntered..\n");

    //critical section
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    //signal
    printf("\nJust Exiting...\n");
    printf("threadID=%lu sem_post sID=%p\n", pthread_self(), &mutex);
    sem_post(&mutex);
    return nullptr;
}


int main() {
    sem_init(&mutex, 0, 1);
    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread, NULL);
    sleep(1);
    pthread_create(&t2, NULL, thread, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    sem_destroy(&mutex);
    return 0;
}