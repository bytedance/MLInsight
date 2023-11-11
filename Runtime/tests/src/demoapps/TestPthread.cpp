/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/

#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>

using namespace std;

void *print_message_function(void *ptr);

int main() {

    pthread_t thread1, thread2;
    std::string message1 = "Thread 1";
    std::string message2 = "Thread 2";
    int iret1, iret2;

    /* Create independent threads each of which will execute function */

    iret1 = pthread_create(&thread1, NULL, print_message_function, (void *) message1.c_str());
//    iret2 = pthread_create(&thread2, NULL, print_message_function, (void *) message2.c_str());

    /* Wait till threads are complete before main continues. Unless we  */
    /* wait we run the risk of executing an exit which will terminate   */
    /* the process and all threads before the threads have completed.   */

    printf("thread1 id=%lu\n", thread1);
//    printf("thread2 id=%lu\n", thread2);

    pthread_join(thread1, NULL);
//    pthread_join(thread2, NULL);

    printf("Thread 1 returns: %d\n", iret1);
//    printf("Thread 2 returns: %d\n", iret2);

//    mlinsight::ExtFuncCallHookAsm *libPltHook = mlinsight::ExtFuncCallHookAsm::getInstance();
//    libPltHook->saveCommonFuncID();
//    libPltHook->saveAllSymbolId();
}

thread_local char asdf;

void *print_message_function(void *ptr) {
    char *message;
    message = (char *) ptr;
    printf("%s \n", message);

    printf("%s %lu\n",&asdf, &asdf - (char*)pthread_self());
    return nullptr;

}