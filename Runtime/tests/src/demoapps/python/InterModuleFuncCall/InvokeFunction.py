import mlinsight
import ModuleA,ModuleB
import testextensionmodule

mlinsight.install()
ModuleA.funcA()
ModuleB.funcB()
ModuleB.callFuncA()
ModuleB.callFuncB()
testextensionmodule.callNativeFuncA()
testextensionmodule.callNativeFuncAThroughDlSym()
