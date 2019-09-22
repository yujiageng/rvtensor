#define _GNU_SOURCE
#include <stdarg.h>
#include <unistd.h>
#include <sched.h>
#include "syscall.h"

extern int __cas_clone(int (*func)(void *), void *stack, int flags, void *arg, ...);

long __syscall_ret(unsigned long r){
        if (r > -4096UL) {
                return -1;
        }
        return r;
}

int cas_clone(int (*func)(void *), void *stack, int flags, void *arg, ...){
        va_list ap;
        pid_t *ptid, *ctid;
        void  *tls;

        va_start(ap, arg);
        ptid = va_arg(ap, pid_t *);
        tls  = va_arg(ap, void *);
        ctid = va_arg(ap, pid_t *);
        va_end(ap);

        return __syscall_ret(__cas_clone(func, stack, flags, arg, ptid, tls, ctid));
}

