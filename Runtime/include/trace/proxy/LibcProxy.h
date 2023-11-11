/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_LIBCPROXY_H
#define MLINSIGHT_LIBCPROXY_H

extern "C" {

typedef int (*main_fn_t)(int, char **, char **);
#ifndef MANUAL_INSTALL

void exit(int __status) __attribute__ ((noreturn));


#endif
}
#endif //MLINSIGHT_LIBCPROXY_H
