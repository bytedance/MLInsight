/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_ASSEMBLYTOOL_H
#define MLINSIGHT_ASSEMBLYTOOL_H

# define THREAD_SELF \
  ({ pthread_t *__self; \
     asm ("mov %%fs:%c1,%0" : "=r" (__self) \
          : "i" (0x10));\
     __self;})
#endif //MLINSIGHT_ASSEMBLYTOOL_H
