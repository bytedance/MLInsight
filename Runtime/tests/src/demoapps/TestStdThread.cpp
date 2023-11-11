/*

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#include <iostream>
#include <thread>
#include <FuncWithDiffParms.h>
#include <CallFunctionCall.h>

using namespace std;


void threadA() {
    callFuncA();
}

void threadB() {
    callFuncA();
}

int main() {

    printf("Hello\n");
    funcA();
    printf("ThreadA\n");

    std::thread thread1(callFuncA);
    std::thread thread2(callFuncA);
//    printf("Thread1.join()\n");
    thread1.join();
    thread2.join();
//    printf("Thread execution complete\n");

    return 0;
}


/*
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <memory>
using namespace std;

struct A
{
    A(int&& n)
    {
        cout << "rvalue overload, n=" << n << endl;
    }
    A(int& n)
    {
        cout << "lvalue overload, n=" << n << endl;
    }
};

class B
{
public:
    template<class T1, class T2, class T3>
    B(T1 && t1, T2 && t2, T3 && t3) :
            a1_(std::forward<T1>(t1)),
            a2_(std::forward<T2>(t2)),
            a3_(std::forward<T3>(t3))
    {

    }
private:
    A a1_, a2_, a3_;
};

template <class T, class U>
std::unique_ptr<T> make_unique1(U&& u)
{
    //return std::unique_ptr<T>(new T(std::forward<U>(u)));
    return std::unique_ptr<T>(new T(std::move(u)));
}

template <class T, class... U>
std::unique_ptr<T> make_unique(U&&... u)
{
    //return std::unique_ptr<T>(new T(std::forward<U>(u)...));
    return std::unique_ptr<T>(new T(std::move(u)...));
}

int main()
{
    std::string asdf="1234";


    auto p1 = make_unique1<A>(2);

}
*/
