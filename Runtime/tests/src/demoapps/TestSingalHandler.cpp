/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/

#include  <stdio.h>
#include  <signal.h>
#include <cstdlib>

long prev_fact, i;

void SIGhandler(int);

void SIGhandler(int sig) {
    printf("\nReceived a SIGUSR1.  The answer is %ld! = %ld\n",
           i - 1, prev_fact);
}

int main(void) {
    long fact;

    printf("Factorial Computation:\n\n");
    signal(SIGUSR1, SIGhandler);
    for (prev_fact = i = 1;; i++, prev_fact = fact) {
        fact = prev_fact * i;
        if (fact < 0){
            printf("Raise SIGUSR1\n");
            raise(SIGUSR1);
            break;
        }
        else if (i % 3 == 0)
            printf("     %ld! = %ld\n", i, fact);
    }
    printf("Raise SIGUSR2\n");
    raise(SIGUSR2);
    printf("Raise signal alarm\n");
    raise(SIGALRM);
    printf("Raise signal term\n");
    raise(SIGTERM);
    printf("Raise complete\n");

    return 0;
}

