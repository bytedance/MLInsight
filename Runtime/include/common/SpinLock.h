#if !defined(__SPINLOCK_H__)
#define __SPINLOCK_H__

/*
 * @file:   spinlock.h
 * @brief:  spinlock used internally.
 *  Note:   Some references: http://locklessinc.com/articles/locks/
 */

class spinlock {
public:
    spinlock() { _lock = 0; }

    void init() { _lock = 0; }

    // Lock
    void lock() {
        while (__atomic_exchange_n(&_lock, 1, __ATOMIC_SEQ_CST) == 1) {
            __asm__("pause");
        }
    }

    void unlock() {
        __atomic_store_n(&_lock, 0, __ATOMIC_SEQ_CST);
    }

    bool _lock;
};

class nolock {
public:
    nolock() {}

    void init() {}

    // Lock
    void lock() {}

    void unlock() {}
};


#endif /* __SPINLOCK_H__ */
