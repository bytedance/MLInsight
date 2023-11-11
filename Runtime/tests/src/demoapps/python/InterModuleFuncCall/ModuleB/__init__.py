import ModuleA
def funcB():
    print("FuncB")

def callFuncA():
    print("callFuncA")
    ModuleA.funcA()

def callFuncB():
    print("callFuncB")
    funcB()